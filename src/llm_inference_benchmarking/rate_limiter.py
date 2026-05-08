"""Token-bucket rate limiter for the inference gateway.

Per-IP (or global) sliding-window and token-bucket implementations.
Configured via env vars:
  GATEWAY_RATE_LIMIT_RPM   — max requests per minute per client (default 60)
  GATEWAY_RATE_LIMIT_ALGO  — "token_bucket" (default) or "sliding_window"
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque


class _TokenBucket:
    """Thread-safe token bucket. Refills at rate=capacity/60 tokens/sec."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._tokens = float(capacity)
        self._refill_rate = capacity / 60.0  # tokens per second
        self._lock = threading.Lock()
        self._last = time.monotonic()

    def consume(self) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._refill_rate)
            self._last = now
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False


class _SlidingWindow:
    """Thread-safe sliding-window counter keyed by 1-second buckets."""

    def __init__(self, limit: int):
        self._limit = limit
        self._window = 60.0  # seconds
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def consume(self) -> bool:
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._window
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) < self._limit:
                self._timestamps.append(now)
                return True
            return False


class RateLimiter:
    """Per-client rate limiter. ``client_id`` is typically a remote IP or API key."""

    def __init__(self):
        self._rpm = int(os.getenv("GATEWAY_RATE_LIMIT_RPM", "60") or "60")
        algo = os.getenv("GATEWAY_RATE_LIMIT_ALGO", "token_bucket").strip().lower()
        self._algo = algo
        self._buckets: dict[str, _TokenBucket | _SlidingWindow] = {}
        self._lock = threading.Lock()

    def _get_bucket(self, client_id: str) -> _TokenBucket | _SlidingWindow:
        with self._lock:
            if client_id not in self._buckets:
                if self._algo == "sliding_window":
                    self._buckets[client_id] = _SlidingWindow(self._rpm)
                else:
                    self._buckets[client_id] = _TokenBucket(self._rpm)
            return self._buckets[client_id]

    def is_allowed(self, client_id: str = "global") -> bool:
        """Return True if the request is within the rate limit."""
        if self._rpm <= 0:
            return True
        return self._get_bucket(client_id).consume()

    @property
    def rpm_limit(self) -> int:
        return self._rpm
