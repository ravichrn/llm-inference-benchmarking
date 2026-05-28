"""Cost-aware inference gateway and benchmarking (standalone package)."""

from __future__ import annotations


def __getattr__(name: str):
    if name in ("GatewayClient", "GatewayLLM", "llm_agent_gateway", "llm_fast_gateway"):
        from llm_inference_benchmarking.client import GatewayClient, GatewayLLM

        globals()["GatewayClient"] = GatewayClient
        globals()["GatewayLLM"] = GatewayLLM
        globals()["llm_agent_gateway"] = GatewayLLM(GatewayClient(), role="agent")
        globals()["llm_fast_gateway"] = GatewayLLM(GatewayClient(), role="fast")
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GatewayClient",
    "GatewayLLM",
    "llm_agent_gateway",
    "llm_fast_gateway",
]
