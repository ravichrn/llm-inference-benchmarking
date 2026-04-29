import logging
import os

from llm_inference_benchmarking.types import GatewayUsage

_log = logging.getLogger(__name__)

_DEFAULT_PRICING = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude-opus-4-6": (0.015, 0.075),
    "claude-sonnet-4-6": (0.003, 0.015),
    "llama3.2": (0.0, 0.0),
    "meta-llama/Llama-3.1-8B-Instruct": (0.0, 0.0),
}


def enrich_cost(usage: GatewayUsage, model: str) -> GatewayUsage:
    in_price, out_price = _resolve_pricing(model)
    usage.estimated_cost_usd = round(
        (usage.input_tokens / 1000.0) * in_price + (usage.output_tokens / 1000.0) * out_price,
        8,
    )
    return usage


def _resolve_pricing(model: str) -> tuple[float, float]:
    custom_in = os.getenv(f"GATEWAY_PRICE_IN_{model.upper().replace('-', '_')}", "")
    custom_out = os.getenv(f"GATEWAY_PRICE_OUT_{model.upper().replace('-', '_')}", "")
    if custom_in and custom_out:
        return float(custom_in), float(custom_out)
    if model not in _DEFAULT_PRICING:
        _log.warning(
            "llm_inference_benchmarking.cost: no pricing entry for model %r — cost will be $0.00. "
            "Set GATEWAY_PRICE_IN_<MODEL> / GATEWAY_PRICE_OUT_<MODEL> to override.",
            model,
        )
    return _DEFAULT_PRICING.get(model, (0.0, 0.0))
