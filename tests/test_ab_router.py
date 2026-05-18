"""Tests for the A/B testing framework (ab_router.py)."""

from __future__ import annotations

import dataclasses
import json
from unittest.mock import MagicMock, patch

import pytest

from llm_inference_benchmarking.ab_router import ABRouter, run_ab
from llm_inference_benchmarking.types import ABResult, ABVariantResult

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_mock_invoke_result(content: str, latency_ms: int = 400, cost: float = 0.0001, model: str = "gpt-mock"):
    usage = MagicMock()
    usage.latency_ms = latency_ms
    usage.estimated_cost_usd = cost
    res = MagicMock()
    res.content = content
    res.usage = usage
    res.model = model
    return res


_PROMPTS = [
    {"id": "t001", "task_type": "qa", "prompt": "What is 2+2?", "reference": "4"},
    {"id": "t002", "task_type": "qa", "prompt": "What is 3+3?", "reference": "6"},
    {"id": "t003", "task_type": "qa", "prompt": "What is 4+4?", "reference": "8"},
]


# ── ABResult schema ───────────────────────────────────────────────────────────


def test_abresult_fields_all_present():
    result = ABResult(
        variant_a=ABVariantResult(
            tier="cheap", model="gpt-a", avg_score=7.0, avg_latency_ms=300.0, total_cost_usd=0.001
        ),
        variant_b=ABVariantResult(
            tier="balanced", model="gpt-b", avg_score=8.0, avg_latency_ms=500.0, total_cost_usd=0.003
        ),
        win_rate_a=0.333,
        n_prompts=3,
        judge_model="gpt-judge",
    )
    d = dataclasses.asdict(result)
    assert "variant_a" in d
    assert "variant_b" in d
    assert "win_rate_a" in d
    assert "n_prompts" in d
    assert "judge_model" in d
    assert d["variant_a"]["tier"] == "cheap"
    assert d["variant_b"]["tier"] == "balanced"


# ── ABRouter.run — patch at method level (thread-safe) ────────────────────────


def _make_router() -> ABRouter:
    router = ABRouter.__new__(ABRouter)
    router._judge_tier = "cheap"
    router._judge_model = ""
    router._client = MagicMock()
    return router


@patch.object(ABRouter, "_judge")
@patch.object(ABRouter, "_invoke_variant")
def test_abresult_win_rate_zero_when_b_always_wins(mock_invoke, mock_judge):
    # A scores 5, B scores 9 for every prompt
    mock_invoke.side_effect = lambda text, tier, mo, pid, label: (
        ("mediocre", 400.0, 0.0001, "gpt-a") if label == "A" else ("excellent", 400.0, 0.0001, "gpt-b")
    )
    mock_judge.side_effect = lambda tt, ref, resp: 5 if resp == "mediocre" else 9

    router = _make_router()
    result = router.run(_PROMPTS, {"tier": "cheap"}, {"tier": "balanced"})
    assert result.win_rate_a == 0.0
    assert result.variant_b.avg_score > result.variant_a.avg_score


@patch.object(ABRouter, "_judge")
@patch.object(ABRouter, "_invoke_variant")
def test_abresult_win_rate_one_when_a_always_wins(mock_invoke, mock_judge):
    mock_invoke.side_effect = lambda text, tier, mo, pid, label: (
        ("excellent", 400.0, 0.0001, "gpt-a") if label == "A" else ("mediocre", 400.0, 0.0001, "gpt-b")
    )
    mock_judge.side_effect = lambda tt, ref, resp: 9 if resp == "excellent" else 4

    router = _make_router()
    result = router.run(_PROMPTS, {"tier": "cheap"}, {"tier": "balanced"})
    assert result.win_rate_a == 1.0


@patch.object(ABRouter, "_judge", return_value=7)
@patch.object(ABRouter, "_invoke_variant", return_value=("answer", 400.0, 0.0001, "gpt-mock"))
def test_n_prompts_matches_input(mock_invoke, mock_judge):
    router = _make_router()
    result = router.run(_PROMPTS, {"tier": "cheap"}, {"tier": "balanced"})
    assert result.n_prompts == len(_PROMPTS)


@patch.object(ABRouter, "_judge", return_value=6)
@patch.object(ABRouter, "_invoke_variant", return_value=("answer", 400.0, 0.0001, "gpt-mock"))
def test_judge_model_propagated(mock_invoke, mock_judge):
    router = _make_router()
    router._judge_model = "gpt-judge-4o"  # set directly since _judge is patched
    result = router.run(_PROMPTS, {"tier": "cheap"}, {"tier": "balanced"})
    assert result.judge_model == "gpt-judge-4o"


# ── run_ab dry_run ────────────────────────────────────────────────────────────


def test_run_ab_dry_run_returns_none(tmp_path, capsys):
    prompts_path = tmp_path / "p.json"
    prompts_path.write_text(json.dumps(_PROMPTS))
    result = run_ab(
        prompts_path=prompts_path,
        variant_a={"tier": "cheap"},
        variant_b={"tier": "balanced"},
        output_path=None,
        dry_run=True,
    )
    assert result is None
    captured = capsys.readouterr()
    assert "dry-run" in captured.out


# ── run_ab writes output ──────────────────────────────────────────────────────


@patch("llm_inference_benchmarking.ab_router.ABRouter")
def test_run_ab_writes_json(mock_router_cls, tmp_path):
    mock_router = MagicMock()
    mock_router_cls.return_value = mock_router
    mock_router.run.return_value = ABResult(
        variant_a=ABVariantResult(tier="cheap", model="a", avg_score=7.0, avg_latency_ms=300.0, total_cost_usd=0.001),
        variant_b=ABVariantResult(
            tier="balanced", model="b", avg_score=8.0, avg_latency_ms=500.0, total_cost_usd=0.003
        ),
        win_rate_a=0.333,
        n_prompts=3,
        judge_model="judge",
    )

    prompts_path = tmp_path / "p.json"
    prompts_path.write_text(json.dumps(_PROMPTS))
    out_path = tmp_path / "ab_out.json"

    result = run_ab(
        prompts_path=prompts_path,
        variant_a={"tier": "cheap"},
        variant_b={"tier": "balanced"},
        output_path=out_path,
    )

    assert result is not None
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["n_prompts"] == 3
    assert data["win_rate_a"] == pytest.approx(0.333)


# ── model override warning ────────────────────────────────────────────────────


def test_invoke_variant_warns_on_model_override(capsys):
    router = _make_router()
    router._client.invoke.return_value = MagicMock(
        content="response",
        usage=MagicMock(latency_ms=300, estimated_cost_usd=0.0001),
        model="gpt-mock",
    )
    router._invoke_variant("hello", "cheap", "gpt-4o", "p001", "A")
    captured = capsys.readouterr()
    assert "[WARN" in captured.out
    assert "gpt-4o" in captured.out


def test_invoke_variant_no_warning_without_override(capsys):
    router = _make_router()
    router._client.invoke.return_value = MagicMock(
        content="response",
        usage=MagicMock(latency_ms=300, estimated_cost_usd=0.0001),
        model="gpt-mock",
    )
    router._invoke_variant("hello", "cheap", "", "p001", "A")
    captured = capsys.readouterr()
    assert "[WARN" not in captured.out
