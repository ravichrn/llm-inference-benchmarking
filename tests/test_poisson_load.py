"""Unit tests for the Poisson arrival load simulator (poisson_load.py).

These tests exercise the pure-logic functions (stats, saturation detection,
queue depth reconstruction, prompt-mix parsing) without needing a running
gateway.
"""

from __future__ import annotations

import random
import statistics

import pytest

from llm_inference_benchmarking.poisson_load import (
    ArrivalRecord,
    LevelResult,
    _compute_level_stats,
    _find_knee,
    _parse_prompt_mix,
    _pct,
    _reconstruct_queue_depth,
)

# ---------------------------------------------------------------------------
# _pct helper
# ---------------------------------------------------------------------------


def test_pct_empty_list_returns_zero():
    assert _pct([], 50) == 0.0


def test_pct_single_element():
    assert _pct([100.0], 50) == 100.0
    assert _pct([100.0], 99) == 100.0


def test_pct_basic_percentiles():
    data = sorted(float(i) for i in range(1, 101))  # 1..100
    # Floor indexing: idx = int(100 * p/100) → p=50 → idx=50 → data[50]=51.0
    assert _pct(data, 50) == 51.0
    assert _pct(data, 99) == 100.0
    assert _pct(data, 0) == 1.0


# ---------------------------------------------------------------------------
# Exponential inter-arrival mean validation
# ---------------------------------------------------------------------------


def test_expovariate_produces_correct_mean():
    """expovariate(lambda_rps) should have mean ≈ 1/lambda_rps."""
    rng = random.Random(0)
    lambda_rps = 2.0
    samples = [rng.expovariate(lambda_rps) for _ in range(10_000)]
    mean = statistics.mean(samples)
    expected_mean = 1.0 / lambda_rps
    # Within 5% of expected mean
    assert abs(mean - expected_mean) / expected_mean < 0.05, f"mean={mean:.4f}, expected≈{expected_mean:.4f}"


# ---------------------------------------------------------------------------
# _reconstruct_queue_depth
# ---------------------------------------------------------------------------


def test_queue_depth_empty():
    mean, peak = _reconstruct_queue_depth([])
    assert mean == 0.0
    assert peak == 0


def test_queue_depth_single_request():
    r = ArrivalRecord(request_id=0, prompt_class="short", scheduled_at=0.0)
    r.fired_at = 0.0
    r.completed_at = 1.0
    r.ok = True
    mean, peak = _reconstruct_queue_depth([r])
    assert mean == pytest.approx(1.0, abs=0.01)
    assert peak == 1


def test_queue_depth_sequential_requests():
    """Two non-overlapping requests should have mean queue depth 1.0 each."""
    r1 = ArrivalRecord(request_id=0, prompt_class="short", scheduled_at=0.0)
    r1.fired_at = 0.0
    r1.completed_at = 1.0
    r1.ok = True

    r2 = ArrivalRecord(request_id=1, prompt_class="short", scheduled_at=1.0)
    r2.fired_at = 1.0
    r2.completed_at = 2.0
    r2.ok = True

    mean, peak = _reconstruct_queue_depth([r1, r2])
    # Each request was in-flight for 1s out of 2s total, mean ≈ 1.0
    assert mean == pytest.approx(1.0, abs=0.01)
    assert peak == 1


def test_queue_depth_overlapping_requests():
    """Two perfectly overlapping requests → mean queue depth 2.0."""
    r1 = ArrivalRecord(request_id=0, prompt_class="short", scheduled_at=0.0)
    r1.fired_at = 0.0
    r1.completed_at = 1.0
    r1.ok = True

    r2 = ArrivalRecord(request_id=1, prompt_class="short", scheduled_at=0.0)
    r2.fired_at = 0.0
    r2.completed_at = 1.0
    r2.ok = True

    mean, peak = _reconstruct_queue_depth([r1, r2])
    assert mean == pytest.approx(2.0, abs=0.01)
    assert peak == 2


def test_little_law_queue_depth():
    """
    Little's Law: mean queue depth = lambda * mean_latency_s in steady state.

    Simulate 10 requests each with latency=0.9s, evenly spaced at 1.0 rps.
    Expected queue depth: 1.0 * 0.9 = 0.9.
    """
    records = []
    for i in range(10):
        r = ArrivalRecord(request_id=i, prompt_class="short", scheduled_at=float(i))
        r.fired_at = float(i)
        r.completed_at = float(i) + 0.9
        r.ok = True
        records.append(r)

    mean, _peak = _reconstruct_queue_depth(records)
    # Each request occupies 0.9s of a 1s window → mean ≈ 0.9
    assert mean == pytest.approx(0.9, abs=0.05)


# ---------------------------------------------------------------------------
# Saturation detection
# ---------------------------------------------------------------------------


def test_saturation_detected_high_p99_ratio():
    """p99 > 3xp50 should trigger saturation."""
    # Build records that produce p50≈800ms, p99≈3200ms
    records = []
    for i in range(100):
        r = ArrivalRecord(request_id=i, prompt_class="short", scheduled_at=float(i))
        r.fired_at = float(i)
        latency = 3200.0 if i >= 99 else 800.0  # p99 >> p50
        r.completed_at = r.fired_at + latency / 1000
        r.ok = True
        r.latency_ms = latency
        records.append(r)

    result = _compute_level_stats(records, lambda_rps=1.0, duration_s=100.0, prompt_mix={"short": 1.0})
    assert result.saturated is True


def test_saturation_detected_throughput_cap():
    """achieved_rps < 0.85 * lambda_rps should trigger saturation."""
    # 10 successful requests in 30s → 0.33 rps, lambda=2.0 → severely below 85%
    records = []
    for i in range(10):
        r = ArrivalRecord(request_id=i, prompt_class="short", scheduled_at=float(i))
        r.fired_at = float(i)
        r.completed_at = r.fired_at + 1.0
        r.ok = True
        r.latency_ms = 1000.0
        records.append(r)

    result = _compute_level_stats(records, lambda_rps=2.0, duration_s=30.0, prompt_mix={"short": 1.0})
    assert result.saturated is True


def test_no_saturation_below_capacity():
    """Uniform low latency at low arrival rate → not saturated."""
    records = []
    for i in range(50):
        r = ArrivalRecord(request_id=i, prompt_class="short", scheduled_at=float(i))
        r.fired_at = float(i)
        r.completed_at = r.fired_at + 0.5  # 500ms, well below inter-arrival gap
        r.ok = True
        r.latency_ms = 500.0
        records.append(r)

    # lambda=1.0, achieved≈1.0, p99/p50 ≈ 1.0
    result = _compute_level_stats(records, lambda_rps=1.0, duration_s=50.0, prompt_mix={"short": 1.0})
    assert result.saturated is False


# ---------------------------------------------------------------------------
# _find_knee
# ---------------------------------------------------------------------------


def test_find_knee_no_saturated_levels():
    levels = [
        LevelResult(
            0.5,
            30,
            {},
            15,
            15,
            0,
            0.5,
            {"p50": 800, "p99": 850, "mean": 800, "p95": 830, "min": 780, "max": 850},
            0.4,
            1,
            5.0,
            False,
        ),
        LevelResult(
            1.0,
            30,
            {},
            30,
            30,
            0,
            1.0,
            {"p50": 810, "p99": 870, "mean": 810, "p95": 850, "min": 800, "max": 870},
            0.8,
            1,
            5.0,
            False,
        ),
    ]
    assert _find_knee(levels) is None


def test_find_knee_returns_first_saturated():
    levels = [
        LevelResult(
            0.5,
            30,
            {},
            15,
            15,
            0,
            0.5,
            {"p50": 800, "p99": 850, "mean": 800, "p95": 830, "min": 780, "max": 850},
            0.4,
            1,
            5.0,
            False,
        ),
        LevelResult(
            2.0,
            30,
            {},
            60,
            52,
            8,
            1.7,
            {"p50": 820, "p99": 5100, "mean": 1200, "p95": 3200, "min": 800, "max": 5100},
            2.1,
            5,
            5.0,
            True,
        ),
        LevelResult(
            5.0,
            30,
            {},
            150,
            51,
            99,
            1.7,
            {"p50": 900, "p99": 18000, "mean": 3000, "p95": 9000, "min": 800, "max": 18000},
            8.4,
            15,
            5.0,
            True,
        ),
    ]
    knee = _find_knee(levels)
    assert knee is not None
    assert knee.lambda_rps == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _parse_prompt_mix
# ---------------------------------------------------------------------------


def test_parse_prompt_mix_basic():
    mix = _parse_prompt_mix("short=0.6,medium=0.3,long=0.1")
    assert set(mix.keys()) == {"short", "medium", "long"}
    assert sum(mix.values()) == pytest.approx(1.0, abs=1e-9)
    assert mix["short"] == pytest.approx(0.6, abs=1e-9)


def test_parse_prompt_mix_normalizes():
    mix = _parse_prompt_mix("short=6,medium=3,long=1")
    assert mix["short"] == pytest.approx(0.6, abs=1e-9)
    assert mix["medium"] == pytest.approx(0.3, abs=1e-9)
    assert mix["long"] == pytest.approx(0.1, abs=1e-9)


def test_parse_prompt_mix_single_class():
    mix = _parse_prompt_mix("long=1.0")
    assert mix == {"long": pytest.approx(1.0)}


def test_parse_prompt_mix_invalid_format():
    with pytest.raises(ValueError, match="Invalid prompt-mix"):
        _parse_prompt_mix("short,medium")


def test_parse_prompt_mix_zero_weight_raises():
    with pytest.raises(ValueError, match="must sum to a positive value"):
        _parse_prompt_mix("short=0,medium=0")


# ---------------------------------------------------------------------------
# _compute_level_stats edge cases
# ---------------------------------------------------------------------------


def test_compute_level_stats_all_failed():
    records = []
    for i in range(5):
        r = ArrivalRecord(request_id=i, prompt_class="short", scheduled_at=float(i))
        r.fired_at = float(i)
        r.completed_at = r.fired_at + 0.1
        r.ok = False
        r.latency_ms = 100.0
        records.append(r)

    result = _compute_level_stats(records, lambda_rps=1.0, duration_s=5.0, prompt_mix={"short": 1.0})
    assert result.successful == 0
    assert result.failed == 5
    assert result.latency_ms["mean"] == 0.0


def test_compute_level_stats_scheduler_drift():
    """scheduler_drift_ms should reflect the mean delay between scheduled and fired times."""
    records = []
    for i in range(10):
        r = ArrivalRecord(request_id=i, prompt_class="short", scheduled_at=float(i))
        r.fired_at = float(i) + 0.020  # 20ms late
        r.completed_at = r.fired_at + 0.5
        r.ok = True
        r.latency_ms = 500.0
        records.append(r)

    result = _compute_level_stats(records, lambda_rps=1.0, duration_s=10.0, prompt_mix={"short": 1.0})
    assert result.scheduler_drift_ms == pytest.approx(20.0, abs=1.0)
