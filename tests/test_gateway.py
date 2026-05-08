"""Gateway HTTP layer: auth/security and library boundary enforcement."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from llm_inference_benchmarking.gateway import app

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
