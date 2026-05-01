"""FastAPI application: /health, /generate, /usage/summary, /metrics endpoints."""

from __future__ import annotations

import os
import sqlite3

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from llm_inference_benchmarking.client import GatewayClient
from llm_inference_benchmarking.ledger import get_ledger_db_path
from llm_inference_benchmarking.types import GatewayRequest

app = FastAPI(title="Inference Gateway", version="0.1.0")
_api_client = GatewayClient()


class _GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    tier: str = Field(default="auto")
    role: str = Field(default="agent")


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if os.getenv("GATEWAY_AUTH_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
        return
    expected = os.getenv("GATEWAY_API_KEY", "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Gateway API key is not configured on the server.",
        )
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "gateway"}


@app.post("/generate")
def generate(req: _GenerateRequest, x_api_key: str | None = Header(default=None)) -> dict:
    _require_api_key(x_api_key)
    try:
        res = _api_client.invoke(GatewayRequest(prompt=req.prompt, tier=req.tier, role=req.role))
        return {
            "content": str(res.content),
            "backend": res.backend,
            "model": res.model,
            "tier": res.tier,
            "usage": {
                "input_tokens": res.usage.input_tokens,
                "output_tokens": res.usage.output_tokens,
                "total_tokens": res.usage.total_tokens,
                "estimated_cost_usd": res.usage.estimated_cost_usd,
                "latency_ms": res.usage.latency_ms,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Gateway inference failed.") from exc


@app.get("/usage/summary")
def usage_summary(x_api_key: str | None = Header(default=None)) -> dict:
    _require_api_key(x_api_key)
    try:
        db_path = get_ledger_db_path()
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    date(ts) AS date,
                    tier,
                    model,
                    COUNT(*) AS requests,
                    SUM(input_tokens) AS input_tokens,
                    SUM(output_tokens) AS output_tokens,
                    ROUND(SUM(estimated_cost_usd), 6) AS total_cost_usd,
                    ROUND(AVG(latency_ms), 0) AS avg_latency_ms
                FROM gateway_usage
                WHERE ok = 1
                GROUP BY date(ts), tier, model
                ORDER BY date DESC, total_cost_usd DESC
                """
            ).fetchall()
        return {"summary": [dict(r) for r in rows]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Ledger query failed.") from exc


@app.get("/metrics")
def metrics_endpoint(x_api_key: str | None = Header(default=None)) -> Response:
    _require_api_key(x_api_key)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
