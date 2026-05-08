"""Per-tier p99 latency SLA enforcement.

Configured via env vars:
  GATEWAY_SLA_P99_CHEAP_MS      — p99 cap for cheap tier (ms), 0 = disabled
  GATEWAY_SLA_P99_BALANCED_MS   — p99 cap for balanced tier (ms), 0 = disabled
  GATEWAY_SLA_P99_PREMIUM_MS    — p99 cap for premium tier (ms), 0 = disabled
  GATEWAY_SLA_WINDOW            — number of recent requests to consider (default 100)

When the observed p99 exceeds the cap:
  - cheap tier: raise SLAViolationError (caller should return 503)
  - balanced tier: downgrade to cheap
  - premium tier: downgrade to balanced
"""

from __future__ import annotations

import os
import threading
import typing
from collections import deque


class SLAViolationError(Exception):
    pass


class SLATracker:
    """Tracks recent latency samples per tier using a sliding deque."""

    _TIER_ENV: typing.ClassVar[dict[str, str]] = {
        "cheap": "GATEWAY_SLA_P99_CHEAP_MS",
        "balanced": "GATEWAY_SLA_P99_BALANCED_MS",
        "premium": "GATEWAY_SLA_P99_PREMIUM_MS",
    }

    def __init__(self):
        window = int(os.getenv("GATEWAY_SLA_WINDOW", "100") or "100")
        self._caps: dict[str, int] = {}
        for tier, env in self._TIER_ENV.items():
            v = int(os.getenv(env, "0") or "0")
            if v > 0:
                self._caps[tier] = v
        self._samples: dict[str, deque[int]] = {t: deque(maxlen=window) for t in self._TIER_ENV}
        self._lock = threading.Lock()

    def record(self, tier: str, latency_ms: int) -> None:
        """Record a completed request latency."""
        with self._lock:
            if tier in self._samples:
                self._samples[tier].append(latency_ms)

    def p99(self, tier: str) -> int | None:
        """Return p99 latency for a tier, or None if no samples."""
        with self._lock:
            samples = list(self._samples.get(tier, []))
        if not samples:
            return None
        samples.sort()
        idx = int(len(samples) * 0.99)
        return samples[min(idx, len(samples) - 1)]

    def check(self, tier: str) -> str:
        """Check SLA before routing. Returns the (possibly downgraded) tier.

        Raises SLAViolationError for cheap tier if p99 cap is breached.
        """
        cap = self._caps.get(tier)
        if cap is None:
            return tier
        observed = self.p99(tier)
        if observed is None or observed <= cap:
            return tier
        # SLA breach: escalate / degrade
        if tier == "cheap":
            raise SLAViolationError(f"cheap tier p99 {observed}ms exceeds cap {cap}ms; no cheaper fallback.")
        if tier == "balanced":
            return "cheap"
        if tier == "premium":
            return "balanced"
        return tier

    @property
    def caps(self) -> dict[str, int]:
        return dict(self._caps)
