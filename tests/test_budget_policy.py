import pytest

from llm_inference_benchmarking.client import _apply_budget_policy
from llm_inference_benchmarking.types import GatewayDecision


class _PolicyStub:
    def resolve_tier(self, tier: str) -> GatewayDecision:
        return GatewayDecision(tier=tier, backend="openai", model="gpt-4o", reason="stub")


def test_hard_cap_blocks_requests(monkeypatch):
    monkeypatch.setattr(
        "llm_inference_benchmarking.client.get_today_success_cost_usd", lambda: 20.0
    )
    monkeypatch.setenv("GATEWAY_DAILY_USD_HARD_CAP", "15")
    monkeypatch.setenv("GATEWAY_DAILY_USD_SOFT_CAP", "5")
    with pytest.raises(RuntimeError):
        _apply_budget_policy(
            _PolicyStub(),
            GatewayDecision(tier="balanced", backend="openai", model="gpt-4o", reason="x"),
        )


def test_soft_cap_downgrades_premium(monkeypatch):
    monkeypatch.setattr("llm_inference_benchmarking.client.get_today_success_cost_usd", lambda: 6.0)
    monkeypatch.setenv("GATEWAY_DAILY_USD_HARD_CAP", "15")
    monkeypatch.setenv("GATEWAY_DAILY_USD_SOFT_CAP", "5")
    out = _apply_budget_policy(
        _PolicyStub(),
        GatewayDecision(tier="premium", backend="claude", model="claude-opus-4-6", reason="x"),
    )
    assert out.tier == "balanced"
