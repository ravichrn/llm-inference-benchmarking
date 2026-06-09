"""Unit tests for the adaptive routing classifier (classifier_router.py)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from llm_inference_benchmarking.classifier_router import (
    MIN_TRAINING_SAMPLES,
    AdaptiveRouter,
    LogisticClassifier,
    _downgrade_tier,
    _ledger_row_to_features,
    _prompt_to_features,
    retrain_from_judge_scores,
)

# ---------------------------------------------------------------------------
# _prompt_to_features
# ---------------------------------------------------------------------------


def test_prompt_features_length():
    features = _prompt_to_features("Hello world")
    assert len(features) == 4


def test_prompt_features_log_scaling():
    short = _prompt_to_features("Hi")
    long = _prompt_to_features("A" * 2000)
    assert long[0] > short[0]  # log(estimated_tokens) for long > short
    assert long[1] > short[1]  # log(char_count) for long > short


def test_prompt_features_analysis_keyword():
    f = _prompt_to_features("Compare these two approaches")
    assert f[2] == 1.0  # has_analysis
    assert f[3] == 0.0  # not has_simple


def test_prompt_features_simple_keyword():
    f = _prompt_to_features("Classify this text")
    assert f[2] == 0.0  # not has_analysis
    assert f[3] == 1.0  # has_simple


def test_prompt_features_no_keywords():
    f = _prompt_to_features("Tell me about the weather")
    assert f[2] == 0.0
    assert f[3] == 0.0


def test_ledger_row_features_length():
    features = _ledger_row_to_features(100)
    assert len(features) == 4


def test_ledger_row_features_monotone():
    f1 = _ledger_row_to_features(10)
    f2 = _ledger_row_to_features(1000)
    assert f2[0] > f1[0]  # log(input_tokens) grows with token count


# ---------------------------------------------------------------------------
# LogisticClassifier
# ---------------------------------------------------------------------------


def test_logistic_classifier_trivial_fit():
    """Classifier should learn a linearly separable 1-feature case."""
    clf = LogisticClassifier(n_features=1, n_classes=3, lr=0.1, epochs=500)
    # Class 0 at x≈0, class 1 at x≈5, class 2 at x≈10
    X = [[0.0]] * 20 + [[5.0]] * 20 + [[10.0]] * 20
    y = [0] * 20 + [1] * 20 + [2] * 20
    clf.fit(X, y)
    assert clf.predict([0.0]) == "cheap"
    assert clf.predict([10.0]) == "premium"


def test_logistic_classifier_predict_proba_sums_to_one():
    clf = LogisticClassifier(n_features=2)
    probs = clf.predict_proba([1.0, 2.0])
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-9


def test_logistic_classifier_confidence():
    clf = LogisticClassifier(n_features=1, n_classes=3, lr=0.1, epochs=500)
    X = [[0.0]] * 30 + [[10.0]] * 30
    y = [0] * 30 + [2] * 30
    clf.fit(X, y)
    tier, conf = clf.confidence([0.0])
    assert tier in ("cheap", "balanced", "premium")
    assert 0.0 < conf <= 1.0


def test_logistic_classifier_empty_fit_does_not_crash():
    clf = LogisticClassifier(n_features=4)
    clf.fit([], [])  # should not raise
    probs = clf.predict_proba([1.0, 2.0, 0.0, 0.0])
    assert len(probs) == 3


# ---------------------------------------------------------------------------
# AdaptiveRouter — fallback when no ledger
# ---------------------------------------------------------------------------


def test_adaptive_router_returns_none_when_no_db(tmp_path):
    router = AdaptiveRouter(db_path=tmp_path / "nonexistent.db")
    result = router.predict_tier("Any prompt")
    assert result is None


def test_adaptive_router_returns_none_when_empty_db(tmp_path):
    db = tmp_path / "empty.db"
    con = sqlite3.connect(str(db))
    con.execute(
        """CREATE TABLE gateway_usage (
            id INTEGER PRIMARY KEY,
            ts TEXT,
            tier TEXT,
            input_tokens INTEGER,
            ok INTEGER
        )"""
    )
    con.close()
    router = AdaptiveRouter(db_path=db)
    result = router.predict_tier("Any prompt")
    assert result is None


def test_adaptive_router_returns_none_below_min_samples(tmp_path):
    db = _create_ledger(tmp_path, n_cheap=10, n_balanced=5, n_premium=5)  # 20 < 50
    router = AdaptiveRouter(db_path=db)
    result = router.predict_tier("Any prompt")
    assert result is None


# ---------------------------------------------------------------------------
# AdaptiveRouter — trains when data is available
# ---------------------------------------------------------------------------


def test_adaptive_router_predicts_valid_tier_after_training(tmp_path):
    db = _create_ledger(tmp_path, n_cheap=30, n_balanced=30, n_premium=30)
    router = AdaptiveRouter(db_path=db)
    result = router.predict_tier("What is 2 plus 2?")
    assert result in ("cheap", "balanced", "premium")


def test_adaptive_router_predicts_with_confidence(tmp_path):
    db = _create_ledger(tmp_path, n_cheap=30, n_balanced=30, n_premium=30)
    router = AdaptiveRouter(db_path=db)
    result = router.predict_with_confidence("What is 2 plus 2?")
    assert result is not None
    tier, conf = result
    assert tier in ("cheap", "balanced", "premium")
    assert 0.0 < conf <= 1.0


def test_adaptive_router_retrains_after_threshold(tmp_path):
    """After MIN_TRAINING_SAMPLES rows, first predict triggers training.
    After RETRAIN_INTERVAL more rows, next predict triggers retraining.
    """
    db = _create_ledger(tmp_path, n_cheap=30, n_balanced=20, n_premium=20)  # 70 >= 50
    router = AdaptiveRouter(db_path=db)

    # First prediction — triggers training
    router.predict_tier("hello")
    first_n = router._trained_on_n
    assert first_n >= MIN_TRAINING_SAMPLES

    # Add RETRAIN_INTERVAL more rows to ledger
    con = sqlite3.connect(str(db))
    for i in range(100):
        con.execute(
            "INSERT INTO gateway_usage (ts, tier, input_tokens, ok) VALUES (?, ?, ?, 1)",
            ("2025-01-01", "cheap", 50 + i),
        )
    con.commit()
    con.close()

    # Next prediction — should retrain (row count grew by >= RETRAIN_INTERVAL)
    router.predict_tier("hello again")
    assert router._trained_on_n > first_n


# ---------------------------------------------------------------------------
# Integration: policy.py uses classifier with fallback
# ---------------------------------------------------------------------------


def test_policy_falls_back_to_heuristics_when_no_classifier(tmp_path, monkeypatch):
    """When AdaptiveRouter returns None, policy falls back to keyword heuristics."""
    monkeypatch.setenv("GATEWAY_LEDGER_DB", str(tmp_path / "empty.db"))
    # Reset the singleton so it picks up the new db path
    import llm_inference_benchmarking.policy as policy_mod

    policy_mod._adaptive_router = None

    from llm_inference_benchmarking.policy import RoutingPolicyEngine
    from llm_inference_benchmarking.types import GatewayRequest

    engine = RoutingPolicyEngine()
    req = GatewayRequest(prompt="Summarize this document in one paragraph.", tier="auto")
    decision = engine.decide(req)
    assert decision.tier in ("cheap", "balanced", "premium")

    # Reset singleton after test
    policy_mod._adaptive_router = None


# ---------------------------------------------------------------------------
# _power_metrics (tested via the pure helper, imported lazily to avoid modal)
# ---------------------------------------------------------------------------


def _get_power_metrics():
    """Import _power_metrics without triggering the modal top-level import.

    Stubs modal only for this import, then restores sys.modules so other tests
    that check for a real modal import are not affected.
    """
    import importlib
    import sys
    import unittest.mock as mock

    modal_was_present = "modal" in sys.modules
    modal_orig = sys.modules.get("modal")
    mb_was_present = "llm_inference_benchmarking.modal_benchmark" in sys.modules
    mb_orig = sys.modules.get("llm_inference_benchmarking.modal_benchmark")

    try:
        if not modal_was_present:
            sys.modules["modal"] = mock.MagicMock()
        mod = importlib.import_module("llm_inference_benchmarking.modal_benchmark")
        return mod._power_metrics
    finally:
        # Restore original state so the mock doesn't leak into other tests
        if not modal_was_present:
            del sys.modules["modal"]
        elif modal_orig is not None:
            sys.modules["modal"] = modal_orig
        if not mb_was_present:
            sys.modules.pop("llm_inference_benchmarking.modal_benchmark", None)
        elif mb_orig is not None:
            sys.modules["llm_inference_benchmarking.modal_benchmark"] = mb_orig


def test_power_metrics_basic():
    _power_metrics = _get_power_metrics()
    samples = [100.0, 120.0, 110.0]
    result = _power_metrics(samples, output_tokens=1000, elapsed_s=10.0)
    assert result["mean_power_w"] == pytest.approx(110.0, abs=0.1)
    assert result["total_energy_j"] == pytest.approx(1100.0, abs=0.1)
    assert result["tokens_per_joule"] == pytest.approx(1000 / 1100.0, rel=0.01)


def test_power_metrics_empty_samples():
    _power_metrics = _get_power_metrics()
    result = _power_metrics([], output_tokens=100, elapsed_s=5.0)
    assert result == {}


def test_power_metrics_zero_elapsed():
    _power_metrics = _get_power_metrics()
    result = _power_metrics([100.0], output_tokens=100, elapsed_s=0.0)
    assert result["tokens_per_joule"] == 0.0


# ---------------------------------------------------------------------------
# _downgrade_tier
# ---------------------------------------------------------------------------


def test_downgrade_tier_premium_to_balanced():
    assert _downgrade_tier("premium") == "balanced"


def test_downgrade_tier_balanced_to_cheap():
    assert _downgrade_tier("balanced") == "cheap"


def test_downgrade_tier_cheap_stays_cheap():
    assert _downgrade_tier("cheap") == "cheap"


# ---------------------------------------------------------------------------
# retrain_from_judge_scores
# ---------------------------------------------------------------------------


def test_retrain_from_judge_scores_no_files(tmp_path):
    n = retrain_from_judge_scores(eval_dir=tmp_path, db_path=tmp_path / "test.db")
    assert n == 0


def test_retrain_from_judge_scores_inserts_rows(tmp_path):
    eval_file = tmp_path / "eval_test.json"
    eval_file.write_text(
        json.dumps(
            {
                "tier": "balanced",
                "results": [
                    {"score": 9, "latency_ms": 500},
                    {"score": 8, "latency_ms": 300},
                    {"score": 4, "latency_ms": 200},  # below threshold → downgraded to cheap
                ],
            }
        )
    )
    db = tmp_path / "test.db"
    n = retrain_from_judge_scores(eval_dir=tmp_path, min_score=7.0, db_path=db)
    assert n == 3

    import sqlite3

    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT tier FROM gateway_usage ORDER BY rowid").fetchall()
    con.close()
    tiers = [r[0] for r in rows]
    assert tiers.count("balanced") == 2  # score >= 7 → keep tier
    assert tiers.count("cheap") == 1  # score 4 < 7 → downgraded


def test_retrain_from_judge_scores_skips_unknown_tier(tmp_path):
    eval_file = tmp_path / "eval_test.json"
    eval_file.write_text(
        json.dumps(
            {
                "tier": "unknown_tier",
                "results": [{"score": 9, "latency_ms": 100}],
            }
        )
    )
    n = retrain_from_judge_scores(eval_dir=tmp_path, db_path=tmp_path / "test.db")
    assert n == 0


def test_retrain_from_judge_scores_skips_missing_score(tmp_path):
    eval_file = tmp_path / "eval_test.json"
    eval_file.write_text(
        json.dumps(
            {
                "tier": "cheap",
                "results": [{"latency_ms": 100}],  # no score field
            }
        )
    )
    n = retrain_from_judge_scores(eval_dir=tmp_path, db_path=tmp_path / "test.db")
    assert n == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_ledger(tmp_path: Path, n_cheap: int, n_balanced: int, n_premium: int) -> Path:
    """Create a test ledger DB with the given number of rows per tier."""
    db = tmp_path / "gateway_usage.db"
    con = sqlite3.connect(str(db))
    con.execute(
        """CREATE TABLE gateway_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT '2025-01-01',
            tier TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            ok INTEGER NOT NULL DEFAULT 1
        )"""
    )
    rows = (
        [("cheap", 50 + i) for i in range(n_cheap)]
        + [("balanced", 300 + i) for i in range(n_balanced)]
        + [("premium", 1500 + i) for i in range(n_premium)]
    )
    con.executemany("INSERT INTO gateway_usage (tier, input_tokens) VALUES (?, ?)", rows)
    con.commit()
    con.close()
    return db
