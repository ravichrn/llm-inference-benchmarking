import logging
import os
import time
from typing import Any

from langchain_openai import ChatOpenAI

from llm_inference_benchmarking.cost import enrich_cost
from llm_inference_benchmarking.ledger import get_today_success_cost_usd, log_usage
from llm_inference_benchmarking.metrics import observe_error, observe_success
from llm_inference_benchmarking.policy import RoutingPolicyEngine
from llm_inference_benchmarking.sla import SLATracker
from llm_inference_benchmarking.types import (
    GatewayDecision,
    GatewayRequest,
    GatewayResult,
    GatewayUsage,
)

_log = logging.getLogger(__name__)


class GatewayClient:
    """Provider-agnostic inference client with tiered routing + telemetry."""

    def __init__(self):
        self.policy = RoutingPolicyEngine()
        self.sla = SLATracker()

    def _quality_route(self, decision: GatewayDecision) -> GatewayDecision:
        """Attempt to downgrade to a cheaper model that still meets quality bar."""
        from llm_inference_benchmarking.quality_router import pick_cheapest_qualified

        tiers = ["cheap", "balanced", "premium"]
        candidates = []
        for t in tiers:
            try:
                d = self.policy.resolve_tier(t)
                candidates.append((t, d.backend, d.model))
            except Exception as exc:
                _log.debug("Skipping tier %s in quality route: %s", t, exc)
        result = pick_cheapest_qualified(candidates)
        if result is None:
            return decision
        tier, backend, model = result
        return GatewayDecision(tier=tier, backend=backend, model=model, reason="quality_route")

    def invoke(self, request: GatewayRequest) -> GatewayResult:
        decision = _apply_budget_policy(self.policy, self.policy.decide(request))
        effective_tier = self.sla.check(decision.tier)
        if effective_tier != decision.tier:
            decision = self.policy.resolve_tier(effective_tier)
        decision = self._quality_route(decision)
        llm = _build_llm(decision.backend, decision.model)
        raw, usage = _invoke_tracked(llm, request.prompt, decision, request.request_id)
        self.sla.record(decision.tier, usage.latency_ms)
        return GatewayResult(
            content=getattr(raw, "content", raw),
            backend=decision.backend,
            model=decision.model,
            tier=decision.tier,
            usage=usage,
            raw=raw,
        )


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
            raw, _ = _invoke_tracked(llm, prompt, decision, req.request_id)
            return raw
        result = self._client.invoke(GatewayRequest(prompt=prompt, role=self._role))
        return result.raw


def _invoke_tracked(
    llm: Any,
    prompt: Any,
    decision: GatewayDecision,
    request_id: str = "",
) -> tuple[Any, GatewayUsage]:
    """Invoke llm, record usage/metrics, and return (raw_response, usage)."""
    t0 = time.perf_counter()
    try:
        raw = llm.invoke(prompt)
        usage = _extract_usage(raw)
        usage.latency_ms = int((time.perf_counter() - t0) * 1000)
        usage = enrich_cost(usage, decision.model)
        log_usage(decision, usage, ok=True, request_id=request_id)
        observe_success(
            tier=decision.tier,
            backend=decision.backend,
            model=decision.model,
            latency_ms=usage.latency_ms,
            estimated_cost_usd=usage.estimated_cost_usd,
        )
        return raw, usage
    except Exception as exc:
        usage = GatewayUsage(latency_ms=int((time.perf_counter() - t0) * 1000))
        usage = enrich_cost(usage, decision.model)
        log_usage(decision, usage, ok=False, error=str(exc), request_id=request_id)
        observe_error(decision.tier, decision.backend, decision.model)
        raise


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
            f"Daily hard cap of ${hard_cap:.2f} reached (spent ${today_spend:.4f}). Requests blocked until tomorrow."
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
        usage_meta.get("input_tokens", 0) or token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
    )
    usage.output_tokens = int(
        usage_meta.get("output_tokens", 0)
        or token_usage.get("completion_tokens", 0)
        or token_usage.get("output_tokens", 0)
    )
    usage.total_tokens = int(
        usage_meta.get("total_tokens", 0) or token_usage.get("total_tokens", usage.input_tokens + usage.output_tokens)
    )
    return usage
