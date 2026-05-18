"""FastAPI application: /health, /generate, /ab, /usage/summary, /metrics endpoints."""

from __future__ import annotations

import dataclasses
import hmac
import logging
import os
import sqlite3

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from llm_inference_benchmarking.ab_router import ABRouter
from llm_inference_benchmarking.client import BudgetExceededError, GatewayClient
from llm_inference_benchmarking.ledger import get_ledger_db_path
from llm_inference_benchmarking.rate_limiter import RateLimiter
from llm_inference_benchmarking.sla import SLAViolationError
from llm_inference_benchmarking.types import GatewayRequest

load_dotenv()

_log = logging.getLogger(__name__)

app = FastAPI(title="Inference Gateway", version="0.1.0")
_api_client = GatewayClient()
_rate_limiter = RateLimiter()
_ab_router = ABRouter()


_VALID_TIERS = {"cheap", "balanced", "premium", "auto"}


class _GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    tier: str = Field(default="auto")
    role: str = Field(default="agent")

    def model_post_init(self, __context: object) -> None:
        if self.tier not in _VALID_TIERS:
            raise ValueError(f"tier must be one of {sorted(_VALID_TIERS)}, got {self.tier!r}")


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if os.getenv("GATEWAY_AUTH_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
        if os.getenv("ENV", "").strip().lower() == "production":
            raise HTTPException(status_code=500, detail="GATEWAY_AUTH_DISABLED is not permitted in production.")
        return
    expected = os.getenv("GATEWAY_API_KEY", "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Gateway API key is not configured on the server.",
        )
    if not hmac.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "gateway"}


@app.post("/generate")
def generate(request: Request, req: _GenerateRequest, x_api_key: str | None = Header(default=None)) -> dict:
    _require_api_key(x_api_key)
    client_ip = _client_ip(request)
    if not _rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {_rate_limiter.rpm_limit} requests/minute.",
            headers={"Retry-After": "60"},
        )
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
    except SLAViolationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BudgetExceededError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception("Unhandled error in /generate")
        raise HTTPException(status_code=500, detail="Gateway inference failed.") from exc


class _ABPrompt(BaseModel):
    prompt: str
    reference: str = ""
    task_type: str = "qa"
    id: str = ""


class _ABRequest(BaseModel):
    prompts: list[_ABPrompt] = Field(..., min_length=1)
    variant_a: dict = Field(default={"tier": "cheap"})
    variant_b: dict = Field(default={"tier": "balanced"})
    judge_tier: str = Field(default="cheap")

    def model_post_init(self, __context: object) -> None:
        for field, val in (("judge_tier", self.judge_tier),):
            if val not in _VALID_TIERS:
                raise ValueError(f"{field} must be one of {sorted(_VALID_TIERS)}, got {val!r}")


@app.post("/ab")
def ab_test(req: _ABRequest, x_api_key: str | None = Header(default=None)) -> dict:
    _require_api_key(x_api_key)
    prompt_dicts = [
        {
            "id": p.id or f"req_{i}",
            "task_type": p.task_type,
            "prompt": p.prompt,
            "reference": p.reference,
        }
        for i, p in enumerate(req.prompts)
    ]
    try:
        router = ABRouter(judge_tier=req.judge_tier)
        result = router.run(prompt_dicts, req.variant_a, req.variant_b)
        return dataclasses.asdict(result)
    except Exception as exc:
        _log.exception("Unhandled error in /ab")
        raise HTTPException(status_code=500, detail="A/B test failed.") from exc


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
        _log.exception("Unhandled error in /usage/summary")
        raise HTTPException(status_code=500, detail="Ledger query failed.") from exc


@app.get("/sla/status")
def sla_status(x_api_key: str | None = Header(default=None)) -> dict:
    _require_api_key(x_api_key)
    tiers = ["cheap", "balanced", "premium"]
    status = {}
    for tier in tiers:
        cap = _api_client.sla.caps.get(tier)
        p99 = _api_client.sla.p99(tier)
        status[tier] = {
            "p99_ms": p99,
            "cap_ms": cap,
            "breached": (p99 is not None and cap is not None and p99 > cap),
        }
    return {"sla": status}


@app.get("/metrics")
def metrics_endpoint(x_api_key: str | None = Header(default=None)) -> Response:
    _require_api_key(x_api_key)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
