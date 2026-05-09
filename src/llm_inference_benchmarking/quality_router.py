"""Quality-aware cheapest-model router.

Ranks candidate models by cost; returns the first whose historical quality
score (MMLU accuracy from benchmark results) meets the configured bar.

Env vars:
  GATEWAY_QUALITY_MIN_SCORE   -- minimum MMLU accuracy [0.0-1.0], default 0.70
  GATEWAY_QUALITY_RESULTS_DIR — directory containing modal_benchmark JSON files,
                                 default "results/"

Falls back to the policy-selected model when no benchmark data is available.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

_log = logging.getLogger(__name__)

# Cost tier ordering: prefer cheaper models first
_TIER_ORDER = ["cheap", "balanced", "premium"]


def _load_quality_scores(results_dir: Path) -> dict[str, float]:
    """Read all modal_benchmark JSON files and extract model→quality_score."""
    scores: dict[str, float] = {}
    if not results_dir.is_dir():
        return scores
    for fpath in results_dir.glob("modal_quant_benchmark*.json"):
        try:
            data = json.loads(fpath.read_text())
            # Modal writes {"results": [...], "gpu_cost_per_hr_usd": ...}; tests write a bare list
            if isinstance(data, dict):
                entries = data.get("results", [])
            elif isinstance(data, list):
                entries = data
            else:
                continue
            for entry in entries:
                model = entry.get("model_id", "")
                q = entry.get("quality", {})
                accuracy = q.get("mmlu_accuracy") if isinstance(q, dict) else None
                if model and accuracy is not None:
                    # keep the best score seen for a model across all result files
                    scores[model] = max(scores.get(model, 0.0), float(accuracy))
        except Exception as exc:
            _log.warning("Skipping %s: %s", fpath.name, exc)
            continue
    return scores


def pick_cheapest_qualified(
    candidates: list[tuple[str, str, str]],  # [(tier, backend, model_id), ...]
    min_score: float | None = None,
    results_dir: Path | None = None,
) -> tuple[str, str, str] | None:
    """Return the cheapest candidate whose quality score meets ``min_score``.

    Args:
        candidates: list of (tier, backend, model_id) tuples, already ordered
                    cheap→expensive by the caller.
        min_score:  minimum acceptable MMLU accuracy. Defaults to
                    GATEWAY_QUALITY_MIN_SCORE env var (0.70 if unset).
        results_dir: directory containing benchmark JSON files.

    Returns:
        The first qualifying (tier, backend, model_id), or None if no data.
    """
    if min_score is None:
        min_score = float(os.getenv("GATEWAY_QUALITY_MIN_SCORE", "0.70") or "0.70")
    if results_dir is None:
        results_dir = Path(os.getenv("GATEWAY_QUALITY_RESULTS_DIR", "results"))

    scores = _load_quality_scores(results_dir)
    if not scores:
        return None  # no benchmark data; caller falls back to policy

    for tier, backend, model_id in candidates:
        score = scores.get(model_id)
        if score is not None and score >= min_score:
            return tier, backend, model_id

    return None
