"""Unit tests for the adaptive routing classifier (classifier_router.py)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from llm_inference_benchmarking.classifier_router import (
    MIN_TRAINING_SAMPLES,
    AdaptiveRouter,
    LogisticClassifier,
    _ledger_row_to_features,
    _prompt_to_features,
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
