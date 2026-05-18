"""Tests for the LLM eval harness (eval.py)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from llm_inference_benchmarking.eval import (
    _REGRESSION_THRESHOLD,
    build_judge_prompt,
    parse_judge_response,
    print_regression_report,
    run_eval,
)

# ── build_judge_prompt ────────────────────────────────────────────────────────


def test_build_judge_prompt_contains_all_fields():
    prompt = build_judge_prompt("reasoning", "Yes, by transitivity.", "Yes it is true.")
    assert "reasoning" in prompt
    assert "Yes, by transitivity." in prompt
    assert "Yes it is true." in prompt


def test_build_judge_prompt_task_types():
    for task_type in ("summarization", "reasoning", "code", "qa", "instruction_following"):
        p = build_judge_prompt(task_type, "ref", "resp")
        assert task_type in p


# ── parse_judge_response ──────────────────────────────────────────────────────


def test_parse_judge_response_valid():
    score, reason = parse_judge_response('{"score": 8, "reason": "Good answer"}')
    assert score == 8
    assert reason == "Good answer"


def test_parse_judge_response_clamps_score():
    score, _ = parse_judge_response('{"score": 15, "reason": "too high"}')
    assert score == 10
    score, _ = parse_judge_response('{"score": -3, "reason": "negative"}')
    assert score == 0


def test_parse_judge_response_strips_markdown_fences():
    raw = '```json\n{"score": 7, "reason": "ok"}\n```'
    score, reason = parse_judge_response(raw)
    assert score == 7
    assert reason == "ok"


def test_parse_judge_response_malformed_returns_zero():
    score, reason = parse_judge_response("not json at all")
    assert score == 0
    assert reason == "parse error"


def test_parse_judge_response_missing_reason():
    score, reason = parse_judge_response('{"score": 5}')
    assert score == 5
    assert reason == ""


# ── regression detection ──────────────────────────────────────────────────────


def _make_result(by_type: dict, run_id: str = "eval_run") -> dict:
    return {
        "run_id": run_id,
        "tier": "cheap",
        "model": "gpt-test",
        "summary": {
            "avg_score": 7.0,
            "avg_latency_ms": 500,
            "total_cost_usd": 0.001,
            "by_task_type": by_type,
        },
        "results": [],
    }


def test_regression_detection_fires_on_drop(capsys):
    current = _make_result({"reasoning": 5.0}, "eval_new")
    prior = _make_result({"reasoning": 7.0}, "eval_old")
    print_regression_report(current, prior)
    captured = capsys.readouterr()
    assert "REGRESSION" in captured.out


def test_regression_detection_silent_on_improvement(capsys):
    current = _make_result({"reasoning": 8.0}, "eval_new")
    prior = _make_result({"reasoning": 7.0}, "eval_old")
    print_regression_report(current, prior)
    captured = capsys.readouterr()
    assert "REGRESSION" not in captured.out
    assert "No regressions detected" in captured.out


def test_regression_threshold_boundary(capsys):
    # Exactly at threshold — should not fire
    current = _make_result({"qa": 7.0 - _REGRESSION_THRESHOLD}, "eval_new")
    prior = _make_result({"qa": 7.0}, "eval_old")
    print_regression_report(current, prior)
    captured = capsys.readouterr()
    assert "REGRESSION" not in captured.out


# ── run_eval dry_run ──────────────────────────────────────────────────────────


def test_run_eval_dry_run_returns_empty(tmp_path, capsys):
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(
        json.dumps(
            [
                {"id": "t001", "task_type": "qa", "prompt": "What is 2+2?", "reference": "4"},
                {"id": "t002", "task_type": "qa", "prompt": "What is 3+3?", "reference": "6"},
            ]
        )
    )
    result = run_eval(tier="cheap", prompts_path=prompts_path, results_dir=tmp_path / "results", dry_run=True)
    assert result == {}
    captured = capsys.readouterr()
    assert "dry-run" in captured.out


# ── run_eval with mocked GatewayClient ───────────────────────────────────────


def _make_mock_result(content: str, latency_ms: int = 500, cost: float = 0.0001):
    usage = MagicMock()
    usage.latency_ms = latency_ms
    usage.estimated_cost_usd = cost
    res = MagicMock()
    res.content = content
    res.usage = usage
    res.model = "gpt-mock"
    return res


@patch("llm_inference_benchmarking.eval.GatewayClient")
def test_run_eval_writes_json(mock_client_cls, tmp_path):
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps([{"id": "q001", "task_type": "qa", "prompt": "What is 1+1?", "reference": "2"}]))

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    # First call: model response; second call: judge response
    mock_client.invoke.side_effect = [
        _make_mock_result("The answer is 2."),
        _make_mock_result('{"score": 9, "reason": "Correct and concise"}'),
    ]

    results_dir = tmp_path / "results"
    result = run_eval(tier="cheap", prompts_path=prompts_path, results_dir=results_dir)

    assert result["tier"] == "cheap"
    assert len(result["results"]) == 1
    assert result["results"][0]["score"] == 9
    assert result["summary"]["avg_score"] == 9.0

    # Result file written
    written = list(results_dir.glob("eval_*.json"))
    assert len(written) == 1
    on_disk = json.loads(written[0].read_text())
    assert on_disk["summary"]["avg_score"] == 9.0


@patch("llm_inference_benchmarking.eval.GatewayClient")
def test_run_eval_handles_model_error_gracefully(mock_client_cls, tmp_path):
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps([{"id": "q001", "task_type": "qa", "prompt": "Fail me", "reference": "n/a"}]))

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.invoke.side_effect = RuntimeError("API error")

    results_dir = tmp_path / "results"
    result = run_eval(tier="cheap", prompts_path=prompts_path, results_dir=results_dir)

    assert result["results"][0]["score"] == 0


# ── judge_tier regression ─────────────────────────────────────────────────────


@patch("llm_inference_benchmarking.eval.GatewayClient")
def test_run_eval_uses_judge_tier(mock_client_cls, tmp_path):
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(json.dumps([{"id": "q001", "task_type": "qa", "prompt": "What is 1+1?", "reference": "2"}]))

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    captured_requests: list = []

    def _capture_invoke(req):
        captured_requests.append(req)
        result = MagicMock()
        result.content = '{"score": 8, "reason": "ok"}'
        result.usage = MagicMock(latency_ms=100, estimated_cost_usd=0.0001)
        result.model = "gpt-mock"
        return result

    mock_client.invoke.side_effect = _capture_invoke

    run_eval(tier="cheap", prompts_path=prompts_path, results_dir=tmp_path / "results", judge_tier="balanced")

    # Second invoke is the judge call; its tier must match judge_tier="balanced"
    assert len(captured_requests) == 2
    assert captured_requests[1].tier == "balanced"
