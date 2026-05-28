"""Modal deployment for the inference gateway.

Deploys the FastAPI gateway (gateway.py) as a live HTTPS endpoint on Modal.
The gateway routes requests across cheap/balanced/premium tiers using OpenAI
and Anthropic backends, with cost tracking, SLA enforcement, and rate limiting.

Deploy:
    modal serve src/llm_inference_benchmarking/modal_gateway.py   # live reload
    modal deploy src/llm_inference_benchmarking/modal_gateway.py  # permanent

Env vars (set via Modal secrets or .env):
    OPENAI_API_KEY              required for openai backend
    ANTHROPIC_API_KEY           required for claude backend
    GATEWAY_API_KEY             shared secret for /generate auth
    GATEWAY_AUTH_DISABLED=1     skip auth (dev only, blocked in production)
    GATEWAY_LEDGER_DB           SQLite path (defaults to /data/gateway_usage.db)
    GATEWAY_DAILY_USD_HARD_CAP  daily spend cap in USD (optional)
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.111.0",
        "uvicorn>=0.30.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0",
        "langchain-core>=0.2.0",
        "langchain-openai>=0.1.0",
        "langchain-anthropic>=0.1.0",
        "langchain-ollama>=0.1.0",
        "prometheus-client>=0.20.0",
        "httpx>=0.27.0",
    )
    .add_local_python_source("llm_inference_benchmarking")
)

# Persistent volume so the SQLite ledger survives container restarts
_ledger_vol = modal.Volume.from_name("gateway-ledger", create_if_missing=True)

_secrets = [
    modal.Secret.from_name("openai-secret"),
    modal.Secret.from_name("anthropic-secret"),
    modal.Secret.from_name("gateway-secret"),
]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("inference-gateway")


@app.function(
    image=_image,
    secrets=_secrets,
    volumes={"/data": _ledger_vol},
    min_containers=1,
)
@modal.asgi_app()
def gateway():
    import os

    # Point ledger at the persistent volume so usage survives restarts
    os.environ.setdefault("GATEWAY_LEDGER_DB", "/data/gateway_usage.db")

    from llm_inference_benchmarking.gateway import app as fastapi_app
    from llm_inference_benchmarking.ledger import init_ledger

    init_ledger()
    return fastapi_app
