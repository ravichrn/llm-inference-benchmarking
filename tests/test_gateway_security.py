from fastapi.testclient import TestClient

from llm_inference_benchmarking.gateway import app


def test_health_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200


def test_generate_requires_api_key(monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "secret")
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    client = TestClient(app)
    response = client.post("/generate", json={"prompt": "hi"})
    assert response.status_code == 401


def test_metrics_requires_api_key(monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "secret")
    monkeypatch.delenv("GATEWAY_AUTH_DISABLED", raising=False)
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 401
