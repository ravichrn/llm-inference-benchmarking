import logging
import os

from llm_inference_benchmarking.types import GatewayUsage

_log = logging.getLogger(__name__)

_DEFAULT_PRICING = {
    # OpenAI — prices per 1k tokens (input, output)
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1-nano": (0.0001, 0.0004),
    "gpt-5.4": (0.002, 0.008),
    "gpt-5.4-mini": (0.0004, 0.0016),
    "gpt-5.5": (0.002, 0.008),
    # Anthropic
    "claude-opus-4-6": (0.015, 0.075),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-haiku-4-5-20251001": (0.0008, 0.004),
    # Local / self-hosted (no cost)
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


def compute_tco(
    gpu_cost_per_hr: float,
    output_tps: float,
    api_price_per_1k_out: float,
    utilization: float = 1.0,
) -> dict:
    """Compare self-hosted GPU vs API cost per 1k output tokens.

    Args:
        gpu_cost_per_hr:       Modal/cloud GPU cost in USD/hr (e.g. 1.10 for A10G).
        output_tps:            Measured output tokens/sec for the model+quant mode.
        api_price_per_1k_out:  Provider API price per 1k output tokens (e.g. 0.015 for claude-opus).
        utilization:           Fraction of time the GPU is serving requests (0 < u ≤ 1).
                               A GPU idle 50% of the time has effective utilization=0.5,
                               doubling the cost per token vs a fully saturated server.

    Returns dict with:
        self_hosted_per_1k_output_usd: cost at the given tps and utilization
        api_per_1k_output_usd:         provider API cost for comparison
        breakeven_output_tps:          tps at which self-hosted equals API cost
        self_hosted_cheaper:           True when self_hosted < api at current tps
        savings_pct:                   positive = self-hosted saves money; negative = API cheaper
    """
    if output_tps <= 0:
        raise ValueError("output_tps must be positive")
    if not (0 < utilization <= 1.0):
        raise ValueError("utilization must be in (0, 1]")

    effective_cost_per_sec = gpu_cost_per_hr / 3600 / utilization
    self_hosted_per_1k = effective_cost_per_sec / output_tps * 1000
    breakeven_tps = (
        (gpu_cost_per_hr / 3600 / utilization) / (api_price_per_1k_out / 1000) if api_price_per_1k_out > 0 else None
    )
    savings_pct = round((1 - self_hosted_per_1k / api_price_per_1k_out) * 100, 1) if api_price_per_1k_out > 0 else None

    return {
        "self_hosted_per_1k_output_usd": round(self_hosted_per_1k, 4),
        "api_per_1k_output_usd": round(api_price_per_1k_out, 4),
        "breakeven_output_tps": round(breakeven_tps, 1) if breakeven_tps is not None else None,
        "self_hosted_cheaper": self_hosted_per_1k < api_price_per_1k_out,
        "savings_pct": savings_pct,
    }


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
