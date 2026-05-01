import os

from llm_inference_benchmarking.types import GatewayDecision, GatewayRequest


class RoutingPolicyEngine:
    """Deterministic tiered router for cost-aware model selection."""

    def decide(self, req: GatewayRequest) -> GatewayDecision:
        forced = os.getenv("GATEWAY_FORCE_TIER", "").strip().lower()
        tier = forced or req.tier
        if tier == "auto":
            tier = self._auto_tier(req)
        return self._resolve_backend(tier)

    def _auto_tier(self, req: GatewayRequest) -> str:
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

    def _resolve_backend(self, tier: str) -> GatewayDecision:
        if tier == "cheap":
            if _check_ollama():
                model = os.getenv("GATEWAY_CHEAP_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2"))
                return GatewayDecision(
                    tier=tier, backend="ollama", model=model, reason="cheap_local"
                )
            if os.getenv("GATEWAY_CHEAP_NO_CLOUD_FALLBACK", "").strip():
                raise RuntimeError(
                    "cheap tier: Ollama unavailable and GATEWAY_CHEAP_NO_CLOUD_FALLBACK is set"
                )
            model = os.getenv("GATEWAY_CHEAP_MODEL", "gpt-5.4-mini")
            return GatewayDecision(tier=tier, backend="openai", model=model, reason="cheap_cloud")

        if tier == "premium":
            premium_backend = os.getenv("GATEWAY_PREMIUM_BACKEND", "openai").lower()
            if premium_backend == "claude":
                model = os.getenv(
                    "GATEWAY_PREMIUM_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
                )
                return GatewayDecision(
                    tier=tier, backend="claude", model=model, reason="premium_quality"
                )
            model = os.getenv("GATEWAY_PREMIUM_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.5"))
            return GatewayDecision(
                tier=tier, backend="openai", model=model, reason="premium_default"
            )

        backend = os.getenv("AGENT_LLM", "openai").lower()
        if backend == "vllm":
            model = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
            return GatewayDecision(tier=tier, backend="vllm", model=model, reason="balanced_vllm")
        if backend == "claude":
            model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
            return GatewayDecision(
                tier=tier, backend="claude", model=model, reason="balanced_claude"
            )
        model = os.getenv("OPENAI_MODEL", "gpt-5.4")
        return GatewayDecision(tier=tier, backend="openai", model=model, reason="balanced_openai")


def _check_ollama() -> bool:
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:11434", timeout=1)
        return True
    except Exception:
        return False
