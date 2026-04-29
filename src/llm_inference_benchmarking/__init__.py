"""Cost-aware inference gateway and benchmarking (standalone package)."""

from llm_inference_benchmarking.client import GatewayClient, GatewayLLM

_gateway = GatewayClient()

llm_agent_gateway = GatewayLLM(_gateway, role="agent")
llm_fast_gateway = GatewayLLM(_gateway, role="fast")

__all__ = [
    "GatewayClient",
    "GatewayLLM",
    "llm_agent_gateway",
    "llm_fast_gateway",
]
