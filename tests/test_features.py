"""Feature tests: rate limiter, SLA, quality router, Pareto, modal modes, TCO, cache benchmark."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


def test_rate_limiter_token_bucket_allows_within_limit():
    with patch.dict(os.environ, {"GATEWAY_RATE_LIMIT_RPM": "10", "GATEWAY_RATE_LIMIT_ALGO": "token_bucket"}):
        from llm_inference_benchmarking.rate_limiter import RateLimiter

        rl = RateLimiter()
        assert all(rl.is_allowed("test_client") for _ in range(10))


def test_rate_limiter_token_bucket_blocks_over_limit():
    with patch.dict(os.environ, {"GATEWAY_RATE_LIMIT_RPM": "3", "GATEWAY_RATE_LIMIT_ALGO": "token_bucket"}):
        from llm_inference_benchmarking.rate_limiter import RateLimiter

        rl = RateLimiter()
        results = [rl.is_allowed("client_x") for _ in range(5)]
        assert results[:3] == [True, True, True]
        assert False in results[3:]


def test_rate_limiter_sliding_window_allows_within_limit():
    with patch.dict(os.environ, {"GATEWAY_RATE_LIMIT_RPM": "5", "GATEWAY_RATE_LIMIT_ALGO": "sliding_window"}):
        from llm_inference_benchmarking.rate_limiter import RateLimiter

        rl = RateLimiter()
        assert all(rl.is_allowed("sw_client") for _ in range(5))


def test_rate_limiter_sliding_window_blocks_over_limit():
    with patch.dict(os.environ, {"GATEWAY_RATE_LIMIT_RPM": "2", "GATEWAY_RATE_LIMIT_ALGO": "sliding_window"}):
        from llm_inference_benchmarking.rate_limiter import RateLimiter

        rl = RateLimiter()
        results = [rl.is_allowed("sw_client2") for _ in range(4)]
        assert results[0] and results[1]
        assert not results[2]


def test_rate_limiter_disabled_when_rpm_zero():
    with patch.dict(os.environ, {"GATEWAY_RATE_LIMIT_RPM": "0"}):
        from llm_inference_benchmarking.rate_limiter import RateLimiter

        rl = RateLimiter()
        assert all(rl.is_allowed("any") for _ in range(1000))


def test_rate_limiter_per_client_isolation():
    with patch.dict(os.environ, {"GATEWAY_RATE_LIMIT_RPM": "2", "GATEWAY_RATE_LIMIT_ALGO": "token_bucket"}):
        from llm_inference_benchmarking.rate_limiter import RateLimiter

        rl = RateLimiter()
        [rl.is_allowed("client_a") for _ in range(3)]  # exhaust client_a's bucket
        assert rl.is_allowed("client_b")


# ---------------------------------------------------------------------------
# SLATracker
# ---------------------------------------------------------------------------


def test_sla_tracker_no_cap_returns_same_tier():
    with patch.dict(
        os.environ,
        {
            "GATEWAY_SLA_P99_CHEAP_MS": "0",
            "GATEWAY_SLA_P99_BALANCED_MS": "0",
            "GATEWAY_SLA_P99_PREMIUM_MS": "0",
        },
    ):
        from llm_inference_benchmarking.sla import SLATracker

        tracker = SLATracker()
        assert tracker.check("premium") == "premium"
        assert tracker.check("balanced") == "balanced"


def test_sla_tracker_within_cap_no_change():
    with patch.dict(os.environ, {"GATEWAY_SLA_P99_BALANCED_MS": "5000", "GATEWAY_SLA_WINDOW": "10"}):
        from llm_inference_benchmarking.sla import SLATracker

        tracker = SLATracker()
        for _ in range(10):
            tracker.record("balanced", 2000)
        assert tracker.check("balanced") == "balanced"


def test_sla_tracker_breach_premium_downgrades_to_balanced():
    with patch.dict(os.environ, {"GATEWAY_SLA_P99_PREMIUM_MS": "3000", "GATEWAY_SLA_WINDOW": "10"}):
        from llm_inference_benchmarking.sla import SLATracker

        tracker = SLATracker()
        for _ in range(10):
            tracker.record("premium", 9000)
        assert tracker.check("premium") == "balanced"


def test_sla_tracker_breach_balanced_downgrades_to_cheap():
    with patch.dict(os.environ, {"GATEWAY_SLA_P99_BALANCED_MS": "1000", "GATEWAY_SLA_WINDOW": "10"}):
        from llm_inference_benchmarking.sla import SLATracker

        tracker = SLATracker()
        for _ in range(10):
            tracker.record("balanced", 5000)
        assert tracker.check("balanced") == "cheap"


def test_sla_tracker_breach_cheap_raises():
    with patch.dict(os.environ, {"GATEWAY_SLA_P99_CHEAP_MS": "500", "GATEWAY_SLA_WINDOW": "10"}):
        from llm_inference_benchmarking.sla import SLATracker, SLAViolationError

        tracker = SLATracker()
        for _ in range(10):
            tracker.record("cheap", 2000)
        with pytest.raises(SLAViolationError):
            tracker.check("cheap")


def test_sla_tracker_p99_no_samples_returns_none():
    with patch.dict(os.environ, {"GATEWAY_SLA_P99_BALANCED_MS": "1000"}):
        from llm_inference_benchmarking.sla import SLATracker

        tracker = SLATracker()
        assert tracker.p99("balanced") is None
        assert tracker.check("balanced") == "balanced"


# ---------------------------------------------------------------------------
# quality_router
# ---------------------------------------------------------------------------


def test_quality_router_no_data_returns_none(tmp_path: Path):
    from llm_inference_benchmarking.quality_router import pick_cheapest_qualified

    candidates = [("cheap", "openai", "gpt-5.4-mini"), ("balanced", "openai", "gpt-5.4")]
    assert pick_cheapest_qualified(candidates, min_score=0.70, results_dir=tmp_path) is None


def test_quality_router_picks_cheapest_qualified(tmp_path: Path):
    from llm_inference_benchmarking.quality_router import pick_cheapest_qualified

    data = [
        {"model_id": "org/cheap-model", "mode": "fp16", "quality": {"mmlu_accuracy": 0.72}},
        {"model_id": "org/expensive-model", "mode": "fp16", "quality": {"mmlu_accuracy": 0.85}},
    ]
    (tmp_path / "modal_quant_benchmark_a10g.json").write_text(json.dumps(data))
    candidates = [("cheap", "openai", "org/cheap-model"), ("balanced", "openai", "org/expensive-model")]
    result = pick_cheapest_qualified(candidates, min_score=0.70, results_dir=tmp_path)
    assert result == ("cheap", "openai", "org/cheap-model")


def test_quality_router_skips_below_threshold(tmp_path: Path):
    from llm_inference_benchmarking.quality_router import pick_cheapest_qualified

    data = [
        {"model_id": "org/cheap-model", "mode": "fp16", "quality": {"mmlu_accuracy": 0.55}},
        {"model_id": "org/good-model", "mode": "fp16", "quality": {"mmlu_accuracy": 0.80}},
    ]
    (tmp_path / "modal_quant_benchmark_a10g.json").write_text(json.dumps(data))
    candidates = [("cheap", "openai", "org/cheap-model"), ("balanced", "openai", "org/good-model")]
    result = pick_cheapest_qualified(candidates, min_score=0.70, results_dir=tmp_path)
    assert result == ("balanced", "openai", "org/good-model")


def test_quality_router_none_when_all_below_threshold(tmp_path: Path):
    from llm_inference_benchmarking.quality_router import pick_cheapest_qualified

    data = [{"model_id": "org/bad-model", "mode": "fp16", "quality": {"mmlu_accuracy": 0.40}}]
    (tmp_path / "modal_quant_benchmark_a10g.json").write_text(json.dumps(data))
    assert pick_cheapest_qualified([("cheap", "openai", "org/bad-model")], min_score=0.70, results_dir=tmp_path) is None


# ---------------------------------------------------------------------------
# Pareto chart (data loading and frontier algorithm only — no matplotlib)
# ---------------------------------------------------------------------------


def test_pareto_load_points(tmp_path: Path):
    from llm_inference_benchmarking.viz import _load_points

    data = [
        {
            "model_id": "org/modelA",
            "mode": "fp16",
            "gpu": "A10G",
            "quality": {"mmlu_accuracy": 0.80},
            "cost": {"self_hosted_per_1k_output_usd": 0.005},
        },
        {
            "model_id": "org/modelB",
            "mode": "nf4",
            "gpu": "A10G",
            "quality": {"mmlu_accuracy": 0.72},
            "cost": {"self_hosted_per_1k_output_usd": 0.002},
        },
    ]
    (tmp_path / "modal_quant_benchmark_a10g.json").write_text(json.dumps(data))
    points = _load_points(tmp_path)
    assert len(points) == 2
    costs = {p["model"]: p["cost_per_1k"] for p in points}
    assert costs["org/modelA"] == 0.005
    assert costs["org/modelB"] == 0.002


def test_pareto_frontier_correctness():
    from llm_inference_benchmarking.viz import _pareto_frontier

    points = [
        {"label": "A", "cost_per_1k": 0.001, "accuracy": 0.60},
        {"label": "B", "cost_per_1k": 0.003, "accuracy": 0.75},
        {"label": "C", "cost_per_1k": 0.005, "accuracy": 0.70},  # dominated by B
        {"label": "D", "cost_per_1k": 0.008, "accuracy": 0.85},
    ]
    labels = {p["label"] for p in _pareto_frontier(points)}
    assert {"A", "B", "D"} == labels


def test_pareto_empty_dir(tmp_path: Path):
    from llm_inference_benchmarking.viz import _load_points

    assert _load_points(tmp_path) == []


# ---------------------------------------------------------------------------
# modal_benchmark: quantization modes
# ---------------------------------------------------------------------------


def test_new_modes_in_all_modes():
    from llm_inference_benchmarking.modal_benchmark import _ALL_MODES

    for mode in (
        "gptq",
        "fp8",
        "flash-attn",
        "torch-compile",
        "tensor-parallel",
        "continuous-batching",
        "tgi",
        "cpu-q2k",
        "cpu-q4km",
        "cpu-q5km",
        "cpu-q8_0",
    ):
        assert mode in _ALL_MODES, f"Mode {mode!r} missing from _ALL_MODES"


def test_default_gptq_model_constant():
    from llm_inference_benchmarking.modal_benchmark import _DEFAULT_GPTQ_MODEL

    assert _DEFAULT_GPTQ_MODEL
    assert "/" in _DEFAULT_GPTQ_MODEL, "Expected 'org/repo' format"


def test_all_modes_no_duplicates():
    from llm_inference_benchmarking.modal_benchmark import _ALL_MODES

    assert len(_ALL_MODES) == len(set(_ALL_MODES))


def test_mode_notes_cover_new_modes():
    from llm_inference_benchmarking.modal_benchmark import _MODE_NOTES

    for mode in (
        "gptq",
        "fp8",
        "flash-attn",
        "torch-compile",
        "tensor-parallel",
        "continuous-batching",
        "tgi",
        "cpu-q2k",
        "cpu-q4km",
        "cpu-q5km",
        "cpu-q8_0",
    ):
        assert mode in _MODE_NOTES, f"No note for new mode {mode!r}"
        assert len(_MODE_NOTES[mode]) > 20, f"Note for {mode!r} is too short"


def test_multi_gpu_modes_and_cpu_modes_constants():
    from llm_inference_benchmarking.modal_benchmark import (
        _ALL_MODES,
        _CPU_MODES,
        _MULTI_GPU_MODES,
        _TGI_MODES,
    )

    assert "tensor-parallel" in _MULTI_GPU_MODES
    assert "tgi" in _TGI_MODES
    for m in ("cpu-q2k", "cpu-q4km", "cpu-q5km", "cpu-q8_0"):
        assert m in _CPU_MODES
    # All special modes must still appear in _ALL_MODES
    for m in _MULTI_GPU_MODES | _CPU_MODES | _TGI_MODES:
        assert m in _ALL_MODES


def test_run_quant_benchmark_has_model_id_param():
    from llm_inference_benchmarking.modal_benchmark import run_quant_benchmark

    fn = getattr(run_quant_benchmark, "_raw_f", None) or run_quant_benchmark
    try:
        sig = inspect.signature(fn)
        assert "model_id" in sig.parameters
        assert sig.parameters["model_id"].default == ""
    except (ValueError, TypeError):
        pytest.skip("Cannot inspect Modal-wrapped function signature")


def test_gpu_output_path_all_gpus():
    from llm_inference_benchmarking.modal_benchmark import _GPU_COST_PER_HR, _gpu_output_path

    for gpu in _GPU_COST_PER_HR:
        p = _gpu_output_path("results/modal_quant_benchmark.json", gpu)
        assert p.suffix == ".json"
        assert gpu.lower().replace("-", "_") in str(p)


# ---------------------------------------------------------------------------
# cost.py: TCO model
# ---------------------------------------------------------------------------


def test_tco_self_hosted_formula():
    from llm_inference_benchmarking.cost import compute_tco

    result = compute_tco(gpu_cost_per_hr=1.10, output_tps=30.0, api_price_per_1k_out=0.015)
    expected = round((1.10 / 3600) / 30.0 * 1000, 4)
    assert abs(result["self_hosted_per_1k_output_usd"] - expected) < 0.0001
    assert result["api_per_1k_output_usd"] == 0.015


def test_tco_self_hosted_cheaper():
    from llm_inference_benchmarking.cost import compute_tco

    result = compute_tco(gpu_cost_per_hr=1.10, output_tps=200.0, api_price_per_1k_out=0.015)
    assert result["self_hosted_cheaper"] is True
    assert result["savings_pct"] > 0


def test_tco_api_cheaper():
    from llm_inference_benchmarking.cost import compute_tco

    result = compute_tco(gpu_cost_per_hr=6.45, output_tps=3.0, api_price_per_1k_out=0.015)
    assert result["self_hosted_cheaper"] is False
    assert result["savings_pct"] < 0


def test_tco_utilization_doubles_cost_at_half():
    from llm_inference_benchmarking.cost import compute_tco

    full = compute_tco(gpu_cost_per_hr=1.10, output_tps=30.0, api_price_per_1k_out=0.015, utilization=1.0)
    half = compute_tco(gpu_cost_per_hr=1.10, output_tps=30.0, api_price_per_1k_out=0.015, utilization=0.5)
    assert abs(half["self_hosted_per_1k_output_usd"] - full["self_hosted_per_1k_output_usd"] * 2) < 0.001


def test_tco_breakeven_tps():
    from llm_inference_benchmarking.cost import compute_tco

    result = compute_tco(gpu_cost_per_hr=1.10, output_tps=30.0, api_price_per_1k_out=0.015)
    expected_breakeven = (1.10 / 3600) / (0.015 / 1000)
    assert abs(result["breakeven_output_tps"] - round(expected_breakeven, 1)) < 0.5


def test_tco_invalid_tps():
    from llm_inference_benchmarking.cost import compute_tco

    with pytest.raises(ValueError, match="output_tps"):
        compute_tco(gpu_cost_per_hr=1.10, output_tps=0.0, api_price_per_1k_out=0.015)


def test_tco_invalid_utilization():
    from llm_inference_benchmarking.cost import compute_tco

    with pytest.raises(ValueError, match="utilization"):
        compute_tco(gpu_cost_per_hr=1.10, output_tps=30.0, api_price_per_1k_out=0.015, utilization=0.0)


# ---------------------------------------------------------------------------
# benchmark.py: cache token extraction and cache benchmark schema
# ---------------------------------------------------------------------------


def test_extract_cache_tokens_claude_format():
    from llm_inference_benchmarking.benchmark import _extract_cache_tokens

    assert _extract_cache_tokens({"usage": {"cache_read_input_tokens": 512, "input_tokens": 600}}) == 512


def test_extract_cache_tokens_openai_format():
    from llm_inference_benchmarking.benchmark import _extract_cache_tokens

    assert _extract_cache_tokens({"prompt_tokens_details": {"cached_tokens": 256}}) == 256


def test_extract_cache_tokens_no_cache():
    from llm_inference_benchmarking.benchmark import _extract_cache_tokens

    assert _extract_cache_tokens({}) == 0
    assert _extract_cache_tokens({"usage": {}}) == 0


def test_extract_cache_tokens_non_dict():
    from llm_inference_benchmarking.benchmark import _extract_cache_tokens

    assert _extract_cache_tokens(None) == 0  # type: ignore[arg-type]


def test_run_cache_benchmark_output_schema(tmp_path: Path):
    from llm_inference_benchmarking.benchmark import run_cache_benchmark
    from llm_inference_benchmarking.types import GatewayResult, GatewayUsage

    fake_usage = GatewayUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        estimated_cost_usd=0.001,
        latency_ms=500,
    )
    fake_raw = MagicMock()
    fake_raw.content = "test response"
    fake_raw.response_metadata = {"usage": {"cache_read_input_tokens": 0}}
    fake_result = GatewayResult(
        content="test response",
        backend="openai",
        model="gpt-5.4-mini",
        tier="cheap",
        usage=fake_usage,
        raw=fake_raw,
    )

    out = tmp_path / "cache_bench.json"
    with patch("llm_inference_benchmarking.benchmark.GatewayClient") as MockClient:
        MockClient.return_value.invoke.return_value = fake_result
        run_cache_benchmark(out)

    assert out.exists()
    rows = json.loads(out.read_text())
    assert isinstance(rows, list) and len(rows) > 0
    required_keys = {
        "tier",
        "backend",
        "model",
        "cold_latency_ms",
        "warm_latency_ms",
        "latency_reduction_pct",
        "cold_cost_usd",
        "warm_cost_usd",
        "cold_cache_read_tokens",
        "warm_cache_read_tokens",
    }
    for row in rows:
        assert required_keys.issubset(row.keys())


# ---------------------------------------------------------------------------
# flops.py: FLOPs funnel and roofline
# ---------------------------------------------------------------------------


def test_model_flops_per_token_known_value():
    from llm_inference_benchmarking.flops import model_flops_per_token

    # Llama-3.1-8B: 6 * 8B = 48B FLOPs for FFN+projections per token
    flops = model_flops_per_token(num_params=8_000_000_000, num_layers=32, seq_len=512, hidden_dim=4096)
    # Should be >= 48B (attention adds a bit on top)
    assert flops >= 48e9
    # Sanity: shouldn't be more than 2x the base estimate
    assert flops < 100e9


def test_roofline_returns_correct_bound_memory():
    from llm_inference_benchmarking.flops import roofline

    # Very low arithmetic intensity → memory bound
    result = roofline(flops_per_token=1e9, dtype_bytes=2.0, achieved_tps=50.0, gpu="A10G")
    assert result["bound"] == "memory"
    assert "achieved_mfu_pct" in result
    assert result["achieved_mfu_pct"] >= 0


def test_roofline_unknown_gpu_returns_safe_default():
    from llm_inference_benchmarking.flops import roofline

    result = roofline(flops_per_token=1e12, dtype_bytes=2.0, achieved_tps=100.0, gpu="UnknownGPU-9000")
    assert result["bound"] == "unknown"
    assert "note" in result


def test_build_flops_funnel_attaches_to_result():
    from llm_inference_benchmarking.flops import build_flops_funnel

    mode_result = {"throughput": {"output_tokens_per_sec": 50.0}, "quant_mode": "fp16"}
    cfg = {"num_params": 8_000_000_000, "num_layers": 32, "hidden": 4096, "seq_len": 512}
    out = build_flops_funnel(mode_result, cfg, "A10G", "fp16")
    assert "flops_funnel" in out
    ff = out["flops_funnel"]
    assert "achieved_mfu_pct" in ff
    assert "bound" in ff
    assert ff["achieved_tps"] == 50.0


def test_build_flops_funnel_safe_on_zero_tps():
    from llm_inference_benchmarking.flops import build_flops_funnel

    mode_result = {"throughput": {"output_tokens_per_sec": 0.0}}
    out = build_flops_funnel(mode_result, {}, "A10G", "fp16")
    # Should return unchanged result without crashing
    assert "flops_funnel" not in out


# ---------------------------------------------------------------------------
# autoscaler.py + cost.py: latency-cost signal
# ---------------------------------------------------------------------------


def test_compute_serving_cost_returns_expected_keys():
    from llm_inference_benchmarking.cost import compute_serving_cost

    r = compute_serving_cost(latency_ms=200, output_tps=50.0, batch_size=4, gpu_cost_per_hr=1.10, utilization=0.8)
    assert {"cost_per_request_usd", "latency_cost_score", "latency_ms", "batch_size"}.issubset(r.keys())


def test_compute_serving_cost_higher_latency_increases_score():
    from llm_inference_benchmarking.cost import compute_serving_cost

    low = compute_serving_cost(latency_ms=100, output_tps=50.0, batch_size=1, gpu_cost_per_hr=1.10)
    high = compute_serving_cost(latency_ms=4000, output_tps=50.0, batch_size=1, gpu_cost_per_hr=1.10)
    assert high["latency_cost_score"] > low["latency_cost_score"]


def test_autoscaler_signal_scale_up_on_low_load():
    from llm_inference_benchmarking.autoscaler import autoscaler_signal

    sig = autoscaler_signal(
        {"p99_latency_ms": 50, "output_tps": 200, "batch_size": 1, "utilization": 0.05, "gpu_cost_per_hr": 1.10}
    )
    assert sig["scale_direction"] == "up"
    assert sig["recommended_batch_size"] > 1


def test_autoscaler_signal_scale_down_on_high_load():
    from llm_inference_benchmarking.autoscaler import autoscaler_signal

    # Latency at ceiling (norm=1.0) + high GPU cost + low utilization → score > 0.8 → down
    sig = autoscaler_signal(
        {"p99_latency_ms": 5000, "output_tps": 5, "batch_size": 16, "utilization": 0.1, "gpu_cost_per_hr": 10.0}
    )
    assert sig["scale_direction"] == "down"
    assert sig["recommended_batch_size"] < 16


def test_autoscaler_signal_hold_in_nominal_range():
    from llm_inference_benchmarking.autoscaler import autoscaler_signal

    # norm_lat = 3500/5000 = 0.7, cost is low -> composite ~0.5 (hold zone 0.4-0.8)
    sig = autoscaler_signal(
        {"p99_latency_ms": 3500, "output_tps": 80, "batch_size": 4, "utilization": 0.5, "gpu_cost_per_hr": 1.10}
    )
    assert sig["scale_direction"] == "hold"


def test_autoscaler_signal_all_keys_present():
    from llm_inference_benchmarking.autoscaler import autoscaler_signal

    sig = autoscaler_signal(
        {"p99_latency_ms": 300, "output_tps": 80, "batch_size": 4, "utilization": 0.55, "gpu_cost_per_hr": 1.10}
    )
    assert {"scale_direction", "score", "recommended_batch_size", "reason", "cost_per_request_usd"}.issubset(sig.keys())


def test_policy_auto_tier_uses_autoscaler_scale_up():
    from unittest.mock import patch

    from llm_inference_benchmarking.policy import RoutingPolicyEngine
    from llm_inference_benchmarking.types import GatewayRequest

    with patch("llm_inference_benchmarking.autoscaler.autoscaler_signal") as mock_sig:
        mock_sig.return_value = {
            "scale_direction": "up",
            "score": 0.2,
            "recommended_batch_size": 2,
            "reason": "low load",
            "cost_per_request_usd": 0.0001,
        }
        engine = RoutingPolicyEngine()
        req = GatewayRequest(
            prompt="test",
            tier="auto",
            metadata={
                "live_metrics": {
                    "p99_latency_ms": 50,
                    "output_tps": 200,
                    "batch_size": 1,
                    "utilization": 0.05,
                    "gpu_cost_per_hr": 1.10,
                }
            },
        )
        decision = engine.decide(req)
    assert decision.tier == "premium"


def test_policy_auto_tier_uses_autoscaler_scale_down():
    from unittest.mock import patch

    from llm_inference_benchmarking.policy import RoutingPolicyEngine
    from llm_inference_benchmarking.types import GatewayRequest

    with patch("llm_inference_benchmarking.autoscaler.autoscaler_signal") as mock_sig:
        mock_sig.return_value = {
            "scale_direction": "down",
            "score": 0.9,
            "recommended_batch_size": 2,
            "reason": "overloaded",
            "cost_per_request_usd": 0.001,
        }
        engine = RoutingPolicyEngine()
        req = GatewayRequest(
            prompt="test",
            tier="auto",
            metadata={
                "live_metrics": {
                    "p99_latency_ms": 4800,
                    "output_tps": 5,
                    "batch_size": 16,
                    "utilization": 0.98,
                    "gpu_cost_per_hr": 1.10,
                }
            },
        )
        decision = engine.decide(req)
    assert decision.tier == "cheap"
