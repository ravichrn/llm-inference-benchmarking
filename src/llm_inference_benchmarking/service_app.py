"""Standalone gateway service entrypoint."""

import sqlite3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_inference_benchmarking.client import GatewayClient
from llm_inference_benchmarking.ledger import get_ledger_db_path
from llm_inference_benchmarking.types import GatewayRequest

app = FastAPI(title="Inference Gateway", version="0.1.0")
client = GatewayClient()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    tier: str = Field(default="auto")
    role: str = Field(default="agent")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "gateway"}


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    try:
        res = client.invoke(GatewayRequest(prompt=req.prompt, tier=req.tier, role=req.role))
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
        raise HTTPException(status_code=500, detail=f"Gateway inference failed: {exc}") from exc


@app.get("/usage/summary")
def usage_summary() -> dict:
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
        raise HTTPException(status_code=500, detail=f"Ledger query failed: {exc}") from exc
