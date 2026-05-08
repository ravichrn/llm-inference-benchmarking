"""Routing policy: tier selection, budget caps, and cloud-fallback controls."""

from __future__ import annotations

import pytest

from llm_inference_benchmarking.client import _apply_budget_policy
from llm_inference_benchmarking.policy import RoutingPolicyEngine
from llm_inference_benchmarking.types import GatewayDecision, GatewayRequest


class _PolicyStub:
    def resolve_tier(self, tier: str) -> GatewayDecision:
        return GatewayDecision(tier=tier, backend="openai", model="gpt-4o", reason="stub")


# ---------------------------------------------------------------------------
# Budget policy
# ---------------------------------------------------------------------------


def test_hard_cap_blocks_requests(monkeypatch):
    monkeypatch.setattr("llm_inference_benchmarking.client.get_today_success_cost_usd", lambda: 20.0)
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


# ---------------------------------------------------------------------------
# Policy controls (Ollama / cloud fallback)
# ---------------------------------------------------------------------------


def test_cheap_falls_back_to_cloud_when_local_unavailable(monkeypatch):
    monkeypatch.setattr("llm_inference_benchmarking.policy._check_ollama", lambda: False)
    monkeypatch.delenv("GATEWAY_CHEAP_NO_CLOUD_FALLBACK", raising=False)
    engine = RoutingPolicyEngine()
    decision = engine.decide(GatewayRequest(prompt="test", tier="cheap"))
    assert decision.backend == "openai"


def test_cheap_raises_when_cloud_fallback_disabled(monkeypatch):
    monkeypatch.setattr("llm_inference_benchmarking.policy._check_ollama", lambda: False)
    monkeypatch.setenv("GATEWAY_CHEAP_NO_CLOUD_FALLBACK", "1")
    engine = RoutingPolicyEngine()
    with pytest.raises(RuntimeError):
        engine.decide(GatewayRequest(prompt="test", tier="cheap"))
