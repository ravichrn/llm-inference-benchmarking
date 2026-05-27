"""Latency-cost autoscaler signal for LLM serving.

Combines p99 latency and GPU utilisation into a composite score that
recommends whether to scale up, scale down, or hold steady.

Usage:
    from llm_inference_benchmarking.autoscaler import autoscaler_signal

    sig = autoscaler_signal({
        "p99_latency_ms": 250,
        "output_tps": 80.0,
        "batch_size": 4,
        "utilization": 0.6,
        "gpu_cost_per_hr": 1.10,
    })
    # {"scale_direction": "hold", "score": 0.52, "recommended_batch_size": 4, "reason": "..."}

Thresholds (env-configurable):
    AUTOSCALER_UP_THRESHOLD   (default 0.4): score below this → scale up
    AUTOSCALER_DOWN_THRESHOLD (default 0.8): score above this → scale down
"""

from __future__ import annotations

import os

from llm_inference_benchmarking.cost import compute_serving_cost

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_DEFAULT_UP_THRESHOLD = 0.4
_DEFAULT_DOWN_THRESHOLD = 0.8

# Batch size step for recommendations
_BATCH_STEP_UP = 2
_BATCH_STEP_DOWN = 2
_BATCH_MIN = 1
_BATCH_MAX = 64


def _up_threshold() -> float:
    return float(os.getenv("AUTOSCALER_UP_THRESHOLD", str(_DEFAULT_UP_THRESHOLD)))


def _down_threshold() -> float:
    return float(os.getenv("AUTOSCALER_DOWN_THRESHOLD", str(_DEFAULT_DOWN_THRESHOLD)))


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


def autoscaler_signal(metrics: dict) -> dict:
    """Compute a scale direction from live serving metrics.

    Args:
        metrics: dict with keys:
            p99_latency_ms   — p99 request latency in milliseconds
            output_tps       — current output tokens per second
            batch_size       — current dynamic batch size
            utilization      — GPU utilisation fraction [0, 1]
            gpu_cost_per_hr  — GPU instance cost in USD/hr

    Returns:
        dict with keys:
            scale_direction  — "up" | "down" | "hold"
            score            — composite score in [0, 1]
            recommended_batch_size
            reason           — human-readable explanation
    """
    p99_ms = float(metrics.get("p99_latency_ms", 0))
    output_tps = float(metrics.get("output_tps", 1))
    batch_size = int(metrics.get("batch_size", 1))
    utilization = float(metrics.get("utilization", 1.0))
    gpu_cost_per_hr = float(metrics.get("gpu_cost_per_hr", 1.10))

    cost_result = compute_serving_cost(
        latency_ms=p99_ms,
        output_tps=output_tps,
        batch_size=batch_size,
        gpu_cost_per_hr=gpu_cost_per_hr,
        utilization=max(utilization, 1e-6),
    )
    score = cost_result["latency_cost_score"]

    up_thresh = _up_threshold()
    down_thresh = _down_threshold()

    if score < up_thresh:
        direction = "up"
        new_batch = min(batch_size + _BATCH_STEP_UP, _BATCH_MAX)
        reason = (
            f"Score {score:.3f} < up_threshold {up_thresh} — "
            f"system is underloaded (p99={p99_ms:.0f}ms, util={utilization:.0%}). "
            f"Increase batch size to {new_batch} or route more traffic here."
        )
    elif score > down_thresh:
        direction = "down"
        new_batch = max(batch_size - _BATCH_STEP_DOWN, _BATCH_MIN)
        reason = (
            f"Score {score:.3f} > down_threshold {down_thresh} — "
            f"system is overloaded or cost-inefficient (p99={p99_ms:.0f}ms, util={utilization:.0%}). "
            f"Reduce batch size to {new_batch} or shed load to a cheaper tier."
        )
    else:
        direction = "hold"
        new_batch = batch_size
        reason = (
            f"Score {score:.3f} in [{up_thresh}, {down_thresh}] — "
            f"serving is within nominal operating range (p99={p99_ms:.0f}ms, util={utilization:.0%})."
        )

    return {
        "scale_direction": direction,
        "score": score,
        "recommended_batch_size": new_batch,
        "reason": reason,
        "cost_per_request_usd": cost_result["cost_per_request_usd"],
    }
