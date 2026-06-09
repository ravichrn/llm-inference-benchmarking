"""Adaptive routing classifier trained on SQLite ledger history.

Trains a multinomial logistic regression on past routing decisions from the
gateway_usage ledger to predict the optimal tier (cheap/balanced/premium)
for new requests.

Falls back silently to keyword heuristics when fewer than MIN_TRAINING_SAMPLES
rows exist in the ledger. Retrains every RETRAIN_INTERVAL new rows.

Usage (internal — called from policy.py):
    router = AdaptiveRouter()
    tier = router.predict_tier(prompt_text)  # returns None if no training data
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path

MIN_TRAINING_SAMPLES = 50
RETRAIN_INTERVAL = 100

_CLASSES = ["cheap", "balanced", "premium"]
_CLASS_INDEX = {c: i for i, c in enumerate(_CLASSES)}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _prompt_to_features(prompt: str) -> list[float]:
    """Extract numeric features from a raw prompt string.

    Features (all log-scaled to handle wide dynamic range):
      0: log(estimated_tokens + 1)   — prompt length proxy
      1: log(char_count + 1)         — raw length signal
      2: has_analysis_keywords        — 1.0 if premium keywords found, else 0.0
      3: has_simple_keywords          — 1.0 if cheap keywords found, else 0.0
    """
    text = str(prompt).lower()
    char_count = len(text)
    estimated_tokens = char_count / 4.0  # rough token estimate

    analysis_kw = ("compare", "digest", "summarize", "analysis", "analyze", "evaluate", "assess")
    simple_kw = ("yes or no", "classify", "grade", "rewrite", "define", "what is", "list")

    has_analysis = 1.0 if any(k in text for k in analysis_kw) else 0.0
    has_simple = 1.0 if any(k in text for k in simple_kw) else 0.0

    return [
        math.log(estimated_tokens + 1),
        math.log(char_count + 1),
        has_analysis,
        has_simple,
    ]


def _ledger_row_to_features(input_tokens: int) -> list[float]:
    """Extract features from a ledger row (prediction-compatible subset).

    Uses input_tokens as a proxy for estimated tokens at routing time.
    """
    estimated_chars = input_tokens * 4.0
    return [
        math.log(input_tokens + 1),
        math.log(estimated_chars + 1),
        0.0,  # keyword signals not available from ledger (retroactive)
        0.0,
    ]


# ---------------------------------------------------------------------------
# Logistic regression (pure stdlib, multinomial, SGD)
# ---------------------------------------------------------------------------


class LogisticClassifier:
    """Multinomial logistic regression trained via mini-batch SGD.

    Attributes:
        W: weight matrix, shape (n_classes, n_features)
        b: bias vector, shape (n_classes,)
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        lr: float = 0.05,
        epochs: int = 200,
        l2: float = 1e-4,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.W: list[list[float]] = [[0.0] * n_features for _ in range(n_classes)]
        self.b: list[float] = [0.0] * n_classes

    def _softmax(self, logits: list[float]) -> list[float]:
        max_l = max(logits)
        exps = [math.exp(lg - max_l) for lg in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def _forward(self, x: list[float]) -> list[float]:
        logits = [sum(self.W[c][j] * x[j] for j in range(self.n_features)) + self.b[c] for c in range(self.n_classes)]
        return self._softmax(logits)

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        """Train via full-batch gradient descent with L2 regularisation."""
        n = len(X)
        if n == 0:
            return

        for _ in range(self.epochs):
            # Accumulate gradients over all samples
            dW = [[0.0] * self.n_features for _ in range(self.n_classes)]
            db = [0.0] * self.n_classes

            for xi, yi in zip(X, y, strict=False):
                probs = self._forward(xi)
                for c in range(self.n_classes):
                    err = probs[c] - (1.0 if c == yi else 0.0)
                    for j in range(self.n_features):
                        dW[c][j] += err * xi[j]
                    db[c] += err

            # Update weights with L2 regularisation
            for c in range(self.n_classes):
                for j in range(self.n_features):
                    self.W[c][j] -= self.lr * (dW[c][j] / n + self.l2 * self.W[c][j])
                self.b[c] -= self.lr * (db[c] / n)

    def predict_proba(self, x: list[float]) -> list[float]:
        return self._forward(x)

    def predict(self, x: list[float]) -> str:
        probs = self._forward(x)
        return _CLASSES[probs.index(max(probs))]

    def confidence(self, x: list[float]) -> tuple[str, float]:
        probs = self._forward(x)
        best_idx = probs.index(max(probs))
        return _CLASSES[best_idx], probs[best_idx]


# ---------------------------------------------------------------------------
# AdaptiveRouter
# ---------------------------------------------------------------------------


class AdaptiveRouter:
    """Adaptive tier router that learns from the gateway_usage SQLite ledger.

    Wraps LogisticClassifier with lazy training, periodic retraining, and a
    clean fallback to None when insufficient data is available.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _default_ledger_path()
        self._classifier: LogisticClassifier | None = None
        self._trained_on_n: int = 0
        self._last_row_count: int = 0

    def _current_row_count(self) -> int:
        if not Path(self._db_path).exists():
            return 0
        try:
            con = sqlite3.connect(str(self._db_path))
            (n,) = con.execute("SELECT COUNT(*) FROM gateway_usage WHERE ok = 1").fetchone()
            con.close()
            return int(n)
        except Exception:
            return 0

    def _load_training_data(self) -> tuple[list[list[float]], list[int]]:
        """Query ledger for tier routing history; return (X_features, y_labels)."""
        try:
            con = sqlite3.connect(str(self._db_path))
            rows = con.execute(
                "SELECT input_tokens, tier FROM gateway_usage WHERE ok = 1 ORDER BY ts DESC LIMIT 5000"
            ).fetchall()
            con.close()
        except Exception:
            return [], []

        X: list[list[float]] = []
        y: list[int] = []
        for input_tokens, tier in rows:
            if tier not in _CLASS_INDEX:
                continue
            X.append(_ledger_row_to_features(int(input_tokens or 0)))
            y.append(_CLASS_INDEX[tier])
        return X, y

    def _maybe_retrain(self) -> None:
        n = self._current_row_count()
        if n < MIN_TRAINING_SAMPLES:
            self._classifier = None
            return
        if self._classifier is not None and (n - self._last_row_count) < RETRAIN_INTERVAL:
            return

        X, y = self._load_training_data()
        if len(X) < MIN_TRAINING_SAMPLES:
            self._classifier = None
            return

        clf = LogisticClassifier(n_features=len(X[0]))
        clf.fit(X, y)
        self._classifier = clf
        self._trained_on_n = len(X)
        self._last_row_count = n

        import logging

        logging.getLogger(__name__).info("[classifier] retrained on %d samples", len(X))

    def predict_tier(self, prompt: str) -> str | None:
        """Return predicted tier, or None if insufficient training data.

        None signals the caller to fall back to keyword heuristics.
        """
        self._maybe_retrain()
        if self._classifier is None:
            return None
        features = _prompt_to_features(prompt)
        return self._classifier.predict(features)

    def predict_with_confidence(self, prompt: str) -> tuple[str, float] | None:
        """Return (tier, confidence_probability) or None."""
        self._maybe_retrain()
        if self._classifier is None:
            return None
        features = _prompt_to_features(prompt)
        return self._classifier.confidence(features)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_ledger_path() -> Path:
    return Path(os.getenv("GATEWAY_LEDGER_DB", "gateway_usage.db"))


def retrain_from_judge_scores(
    eval_dir: Path | str = "results",
    min_score: float = 7.0,
    db_path: Path | None = None,
) -> int:
    """Retrain the AdaptiveRouter using LLM-as-judge quality scores from eval result files.

    Reads all eval_*.json files in eval_dir. Each file contains a list of tasks with
    a ``score`` (0-10) and the ``tier`` that was used. Tasks whose score >= min_score
    are treated as positive training examples; tasks below the threshold are treated as
    negative (downgrade signal).

    Returns the number of samples written to the ledger, or 0 if nothing was found.

    Why this matters: the classifier normally trains on past routing decisions that were
    themselves produced by keyword heuristics, making training circular. Judge scores are
    an independent quality signal — they let the router learn whether a tier was
    *actually sufficient*, not just which tier the heuristic chose.
    """
    eval_path = Path(eval_dir)
    rows: list[tuple[int, str]] = []  # (estimated_input_tokens, tier)

    for f in sorted(eval_path.glob("eval_*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        tier = data.get("tier")
        if tier not in _CLASS_INDEX:
            continue
        for task in data.get("results", []):
            score = task.get("score")
            if score is None:
                continue
            # Use latency_ms as a rough proxy for prompt complexity when tokens unknown
            lat = task.get("latency_ms") or 0
            estimated_tokens = max(10, int(lat / 10))  # crude but consistent proxy
            effective_tier = tier if float(score) >= min_score else _downgrade_tier(tier)
            rows.append((estimated_tokens, effective_tier))

    if not rows:
        return 0

    db = db_path or _default_ledger_path()
    try:
        con = sqlite3.connect(str(db))
        con.execute(
            """CREATE TABLE IF NOT EXISTS gateway_usage (
                ts TEXT, tier TEXT, input_tokens INTEGER, output_tokens INTEGER,
                latency_ms REAL, cost_usd REAL, ok INTEGER DEFAULT 1
            )"""
        )
        con.executemany(
            "INSERT INTO gateway_usage (ts, tier, input_tokens, output_tokens, latency_ms, cost_usd, ok) "
            "VALUES (datetime('now'), ?, ?, 0, 0, 0, 1)",
            [(tier, tokens) for tokens, tier in rows],
        )
        con.commit()
        con.close()
    except Exception:
        return 0

    import logging

    logging.getLogger(__name__).info("[classifier] inserted %d judge-score samples into ledger", len(rows))
    return len(rows)


def _downgrade_tier(tier: str) -> str:
    """Return the next cheaper tier (premium→balanced→cheap)."""
    order = ["cheap", "balanced", "premium"]
    idx = order.index(tier) if tier in order else 2
    return order[max(0, idx - 1)]
