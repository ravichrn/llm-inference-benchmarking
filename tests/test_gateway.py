"""Gateway HTTP layer: auth/security and library boundary enforcement."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from llm_inference_benchmarking.client import BudgetExceededError
from llm_inference_benchmarking.gateway import app
from llm_inference_benchmarking.types import ABResult, ABVariantResult, GatewayResult, GatewayUsage


def _fake_gateway_result() -> GatewayResult:
    return GatewayResult(
        content="hello",
        backend="openai",
        model="gpt-mock",
        tier="cheap",
        usage=GatewayUsage(input_tokens=10, output_tokens=5, total_tokens=15, estimated_cost_usd=0.001, latency_ms=200),
    )


def _fake_ab_result() -> ABResult:
    variant = ABVariantResult(tier="cheap", model="gpt-mock", avg_score=8.0, avg_latency_ms=300.0, total_cost_usd=0.01)
    return ABResult(variant_a=variant, variant_b=variant, win_rate_a=0.5, n_prompts=3, judge_model="gpt-judge")


# ---------------------------------------------------------------------------
# Auth / security
# ---------------------------------------------------------------------------


def test_health_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    client = TestClient(app)
    assert client.get("/health").status_code == 200


def test_generate_requires_api_key(monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "secret")
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    client = TestClient(app)
    assert client.post("/generate", json={"prompt": "hi"}).status_code == 401


def test_generate_wrong_api_key_returns_401(monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "correct-key")
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    client = TestClient(app)
    assert client.post("/generate", json={"prompt": "hi"}, headers={"x-api-key": "wrong-key"}).status_code == 401


@patch("llm_inference_benchmarking.gateway._api_client")
def test_generate_success(mock_client, monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "secret")
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    mock_client.invoke.return_value = _fake_gateway_result()
    client = TestClient(app)
    resp = client.post("/generate", json={"prompt": "hello"}, headers={"x-api-key": "secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert "content" in body
    assert "backend" in body
    assert "model" in body
    assert "tier" in body
    assert "usage" in body
    usage = body["usage"]
    assert "input_tokens" in usage
    assert "estimated_cost_usd" in usage
    assert "latency_ms" in usage


@patch("llm_inference_benchmarking.gateway._api_client")
def test_generate_budget_exceeded_returns_402(mock_client, monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "secret")
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    mock_client.invoke.side_effect = BudgetExceededError("hard cap reached")
    client = TestClient(app)
    resp = client.post("/generate", json={"prompt": "hello"}, headers={"x-api-key": "secret"})
    assert resp.status_code == 402


def test_generate_rate_limit_returns_429(monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "secret")
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    client = TestClient(app)
    with patch("llm_inference_benchmarking.gateway._rate_limiter") as mock_rl:
        mock_rl.is_allowed.return_value = False
        mock_rl.rpm_limit = 60
        resp = client.post("/generate", json={"prompt": "hello"}, headers={"x-api-key": "secret"})
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_ab_requires_api_key(monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "secret")
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    client = TestClient(app)
    resp = client.post("/ab", json={"prompts": [{"prompt": "hi"}]})
    assert resp.status_code == 401


def test_ab_success_and_judge_tier(monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "secret")
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    captured_judge_tier: list[str] = []

    class _FakeABRouter:
        def __init__(self, judge_tier: str = "cheap"):
            captured_judge_tier.append(judge_tier)

        def run(self, *args, **kwargs):
            return _fake_ab_result()

    client = TestClient(app)
    with patch("llm_inference_benchmarking.gateway.ABRouter", _FakeABRouter):
        resp = client.post(
            "/ab",
            json={"prompts": [{"prompt": "What is 2+2?", "reference": "4"}], "judge_tier": "balanced"},
            headers={"x-api-key": "secret"},
        )
    assert resp.status_code == 200
    assert captured_judge_tier == ["balanced"]
    body = resp.json()
    assert "variant_a" in body
    assert "win_rate_a" in body
    assert "n_prompts" in body


def test_sla_status_success(monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "secret")
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    client = TestClient(app)
    resp = client.get("/sla/status", headers={"x-api-key": "secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert "sla" in body
    for tier in ("cheap", "balanced", "premium"):
        assert tier in body["sla"]
        assert "p99_ms" in body["sla"][tier]
        assert "cap_ms" in body["sla"][tier]
        assert "breached" in body["sla"][tier]


def test_metrics_requires_api_key(monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "secret")
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    client = TestClient(app)
    assert client.get("/metrics").status_code == 401


# ---------------------------------------------------------------------------
# Library boundary — no coupling to application-layer modules
# ---------------------------------------------------------------------------

_FORBIDDEN = (
    "from agents",
    "import agents",
    "from ingestion",
    "import ingestion",
    "from databases",
    "import databases",
    "from guardrails",
    "import guardrails",
)

_SRC = Path(__file__).resolve().parent.parent / "src" / "llm_inference_benchmarking"


def test_library_has_no_app_coupling_strings():
    py_files = list(_SRC.glob("*.py"))
    assert py_files, f"No Python files found in {_SRC}"
    combined = "\n".join(f.read_text() for f in py_files)
    violations = [t for t in _FORBIDDEN if t in combined]
    assert not violations, "Library must not reference application modules:\n" + "\n".join(violations)
