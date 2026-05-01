import os
import time
from typing import Any

from langchain_openai import ChatOpenAI

from llm_inference_benchmarking.cost import enrich_cost
from llm_inference_benchmarking.ledger import get_today_success_cost_usd, log_usage
from llm_inference_benchmarking.metrics import observe_error, observe_success
from llm_inference_benchmarking.policy import RoutingPolicyEngine
from llm_inference_benchmarking.types import (
    GatewayDecision,
    GatewayRequest,
    GatewayResult,
    GatewayUsage,
)


class GatewayClient:
    """Provider-agnostic inference client with tiered routing + telemetry."""

    def __init__(self):
        self.policy = RoutingPolicyEngine()

    def invoke(self, request: GatewayRequest) -> GatewayResult:
        decision = self.policy.decide(request)
        llm = _build_llm(decision.backend, decision.model)
        t0 = time.perf_counter()
        try:
            raw = llm.invoke(request.prompt)
            usage = _extract_usage(raw)
            usage.latency_ms = int((time.perf_counter() - t0) * 1000)
            usage = enrich_cost(usage, decision.model)
            log_usage(decision, usage, ok=True, request_id=request.request_id)
            observe_success(
                tier=decision.tier,
                backend=decision.backend,
                model=decision.model,
                latency_ms=usage.latency_ms,
                estimated_cost_usd=usage.estimated_cost_usd,
            )
            return GatewayResult(
                content=getattr(raw, "content", raw),
                backend=decision.backend,
                model=decision.model,
                tier=decision.tier,
                usage=usage,
                raw=raw,
            )
        except Exception as exc:
            usage = GatewayUsage(latency_ms=int((time.perf_counter() - t0) * 1000))
            usage = enrich_cost(usage, decision.model)
            log_usage(decision, usage, ok=False, error=str(exc), request_id=request.request_id)
            observe_error(decision.tier, decision.backend, decision.model)
            raise


class GatewayLLM:
    """Drop-in wrapper exposing .invoke() and .bind_tools() like LangChain chat models."""

    def __init__(self, client: GatewayClient, role: str):
        self._client = client
        self._role = role
        self._bound_tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "GatewayLLM":
        clone = GatewayLLM(self._client, self._role)
        clone._bound_tools = list(tools)
        return clone

    def invoke(self, prompt: Any):
        if self._bound_tools:
            req = GatewayRequest(prompt=prompt, role=self._role)
            decision = self._client.policy.decide(req)
            llm = _build_llm(decision.backend, decision.model).bind_tools(self._bound_tools)
            t0 = time.perf_counter()
            try:
                raw = llm.invoke(prompt)
                usage = _extract_usage(raw)
                usage.latency_ms = int((time.perf_counter() - t0) * 1000)
                usage = enrich_cost(usage, decision.model)
                log_usage(decision, usage, ok=True, request_id=req.request_id)
                observe_success(
                    tier=decision.tier,
                    backend=decision.backend,
                    model=decision.model,
                    latency_ms=usage.latency_ms,
                    estimated_cost_usd=usage.estimated_cost_usd,
                )
                return raw
            except Exception as exc:
                usage = GatewayUsage(latency_ms=int((time.perf_counter() - t0) * 1000))
                usage = enrich_cost(usage, decision.model)
                log_usage(decision, usage, ok=False, error=str(exc), request_id=req.request_id)
                observe_error(decision.tier, decision.backend, decision.model)
                raise
        result = self._client.invoke(GatewayRequest(prompt=prompt, role=self._role))
        return result.raw


def _build_llm(backend: str, model: str):
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=0)
    if backend == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(  # type: ignore[call-arg]
            model=model,
            temperature=0.3,
            model_kwargs={"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}},
        )
    if backend == "vllm":
        import os

        return ChatOpenAI(
            base_url=os.getenv("VLLM_BASE_URL", "http://vllm:8080/v1"),
            api_key="EMPTY",
            model=model,
            temperature=0.3,
        )
    return ChatOpenAI(model=model, temperature=0.3)


def _apply_budget_policy(policy: Any, decision: GatewayDecision) -> GatewayDecision:
    """Enforce daily spend caps. Raises RuntimeError on hard cap; downgrades tier on soft cap."""
    hard_cap = float(os.getenv("GATEWAY_DAILY_USD_HARD_CAP", "0") or "0")
    soft_cap = float(os.getenv("GATEWAY_DAILY_USD_SOFT_CAP", "0") or "0")
    if not hard_cap and not soft_cap:
        return decision
    today_spend = get_today_success_cost_usd()
    if hard_cap and today_spend >= hard_cap:
        raise RuntimeError(
            f"Daily hard cap of ${hard_cap:.2f} reached (spent ${today_spend:.4f}). "
            "Requests blocked until tomorrow."
        )
    if soft_cap and today_spend >= soft_cap and decision.tier == "premium":
        return policy.resolve_tier("balanced")
    return decision


def _extract_usage(raw: Any) -> GatewayUsage:
    usage = GatewayUsage()
    usage_meta = getattr(raw, "usage_metadata", None) or {}
    response_meta = getattr(raw, "response_metadata", None) or {}
    token_usage = response_meta.get("token_usage", {}) if isinstance(response_meta, dict) else {}

    usage.input_tokens = int(
        usage_meta.get("input_tokens", 0)
        or token_usage.get("prompt_tokens", 0)
        or token_usage.get("input_tokens", 0)
    )
    usage.output_tokens = int(
        usage_meta.get("output_tokens", 0)
        or token_usage.get("completion_tokens", 0)
        or token_usage.get("output_tokens", 0)
    )
    usage.total_tokens = int(
        usage_meta.get("total_tokens", 0)
        or token_usage.get("total_tokens", usage.input_tokens + usage.output_tokens)
    )
    return usage
