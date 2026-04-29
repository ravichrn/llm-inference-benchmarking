from prometheus_client import Counter, Histogram

_REQS = Counter(
    "gateway_requests_total",
    "Gateway requests by tier/backend/model/status",
    ["tier", "backend", "model", "status"],
)
_LATENCY = Histogram(
    "gateway_latency_seconds",
    "Gateway end-to-end latency in seconds",
    ["tier", "backend", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
_ESTIMATED_COST = Counter(
    "gateway_estimated_cost_usd_total",
    "Cumulative estimated gateway cost in USD",
    ["tier", "backend", "model"],
)


def observe_success(
    tier: str, backend: str, model: str, latency_ms: int, estimated_cost_usd: float
) -> None:
    _REQS.labels(tier=tier, backend=backend, model=model, status="ok").inc()
    _LATENCY.labels(tier=tier, backend=backend, model=model).observe(latency_ms / 1000.0)
    _ESTIMATED_COST.labels(tier=tier, backend=backend, model=model).inc(
        max(estimated_cost_usd, 0.0)
    )


def observe_error(tier: str, backend: str, model: str) -> None:
    _REQS.labels(tier=tier, backend=backend, model=model, status="error").inc()
