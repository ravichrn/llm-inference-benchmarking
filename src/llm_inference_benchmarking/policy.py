import logging
import os
import time

from llm_inference_benchmarking.types import GatewayDecision, GatewayRequest

_logger = logging.getLogger(__name__)

# Lazily constructed singleton — avoids importing classifier_router at module load
_adaptive_router = None


def _get_adaptive_router():
    global _adaptive_router
    if _adaptive_router is None:
        from llm_inference_benchmarking.classifier_router import AdaptiveRouter

        _adaptive_router = AdaptiveRouter()
    return _adaptive_router


class RoutingPolicyEngine:
    """Deterministic tiered router for cost-aware model selection."""

    def decide(self, req: GatewayRequest) -> GatewayDecision:
        forced = os.getenv("GATEWAY_FORCE_TIER", "").strip().lower()
        tier = forced or req.tier
        if tier == "auto":
            tier = self._auto_tier(req)
        return self._resolve_backend(tier)

    def _auto_tier(self, req: GatewayRequest) -> str:
        # 1. Autoscaler signal (live metrics override — highest priority)
        live_metrics = req.metadata.get("live_metrics") if req.metadata else None
        if live_metrics:
            from llm_inference_benchmarking.autoscaler import autoscaler_signal

            sig = autoscaler_signal(live_metrics)
            if sig["scale_direction"] == "up":
                return "premium"
            if sig["scale_direction"] == "down":
                return "cheap"

        # 2. ML classifier — returns None when ledger has < MIN_TRAINING_SAMPLES rows
        try:
            router = _get_adaptive_router()
            result = router.predict_with_confidence(str(req.prompt))
            if result is not None:
                ml_tier, confidence = result
                _logger.debug(
                    "[auto-tier] ml_classifier predicted %r (confidence=%.2f) for prompt_len=%d",
                    ml_tier,
                    confidence,
                    len(str(req.prompt)),
                )
                return ml_tier
        except Exception as exc:
            _logger.debug("[auto-tier] classifier error, falling back to heuristics: %s", exc)

        # 3. Keyword + length heuristics (original fallback — unchanged)
        _logger.debug("[auto-tier] classifier fallback (insufficient data or error), using heuristics")
        role = req.role.lower()
        if role == "fast":
            return "cheap"
        text = str(req.prompt).lower()
        if any(k in text for k in ("rewrite", "yes or no", "grade", "classify")):
            return "cheap"
        if any(k in text for k in ("compare", "digest", "summarize", "analysis")):
            return "premium"
        prompt_len = len(text)
        if prompt_len > 4000:
            return "cheap"
        if prompt_len > 2000:
            return "balanced"
        return "balanced"

    def resolve_tier(self, tier: str) -> GatewayDecision:
        return self._resolve_backend(tier)

    def _resolve_backend(self, tier: str) -> GatewayDecision:
        if tier == "cheap":
            if _check_ollama():
                model = os.getenv("GATEWAY_CHEAP_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2"))
                return GatewayDecision(tier=tier, backend="ollama", model=model, reason="cheap_local")
            if os.getenv("GATEWAY_CHEAP_NO_CLOUD_FALLBACK", "").strip().lower() in {"1", "true", "yes"}:
                raise RuntimeError("cheap tier: Ollama unavailable and GATEWAY_CHEAP_NO_CLOUD_FALLBACK is set")
            model = os.getenv("GATEWAY_CHEAP_MODEL", "gpt-5.4-mini")
            return GatewayDecision(tier=tier, backend="openai", model=model, reason="cheap_cloud")

        if tier == "premium":
            premium_backend = os.getenv("GATEWAY_PREMIUM_BACKEND", "openai").lower()
            if premium_backend == "claude":
                model = os.getenv("GATEWAY_PREMIUM_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6"))
                return GatewayDecision(tier=tier, backend="claude", model=model, reason="premium_quality")
            model = os.getenv("GATEWAY_PREMIUM_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.5"))
            return GatewayDecision(tier=tier, backend="openai", model=model, reason="premium_default")

        backend = os.getenv("AGENT_LLM", "openai").lower()
        if backend == "vllm":
            model = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
            return GatewayDecision(tier=tier, backend="vllm", model=model, reason="balanced_vllm")
        if backend == "claude":
            model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
            return GatewayDecision(tier=tier, backend="claude", model=model, reason="balanced_claude")
        model = os.getenv("OPENAI_MODEL", "gpt-5.4")
        return GatewayDecision(tier=tier, backend="openai", model=model, reason="balanced_openai")


_ollama_cache: tuple[float, bool] = (0.0, False)


def _ollama_ttl() -> float:
    return float(os.getenv("GATEWAY_OLLAMA_HEALTH_TTL_S", "10") or "10")


def _check_ollama() -> bool:
    global _ollama_cache
    now = time.monotonic()
    if now - _ollama_cache[0] < _ollama_ttl():
        return _ollama_cache[1]
    try:
        import http.client

        conn = http.client.HTTPConnection("localhost", 11434, timeout=1)
        conn.request("HEAD", "/")
        conn.getresponse()
        result = True
    except Exception:
        result = False
    _ollama_cache = (now, result)
    return result
