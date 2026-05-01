import pytest

from llm_inference_benchmarking.policy import RoutingPolicyEngine
from llm_inference_benchmarking.types import GatewayRequest


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
