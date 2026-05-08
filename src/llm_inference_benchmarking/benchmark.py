"""Independent benchmark runner for gateway tiers/backends."""

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from llm_inference_benchmarking.client import GatewayClient
from llm_inference_benchmarking.types import GatewayRequest

load_dotenv()

_CACHE_PROMPTS = [
    # Longer prompts increase prefix-cache hit probability (providers cache >1024 tokens)
    (
        "You are an expert AI researcher. "
        "Explain in detail how retrieval-augmented generation (RAG) reduces hallucination "
        "in large language models. Cover: (1) why LLMs hallucinate, (2) how retrieval "
        "grounds the response, (3) the role of reranking, (4) failure modes of RAG itself."
    ),
    (
        "You are an expert in GPU systems. "
        "Compare fp16, int8, nf4, and AWQ quantization for LLM inference on an A10G GPU. "
        "For each format discuss: memory footprint, throughput, perplexity impact, and "
        "when you would choose it over the others. Include a concrete decision guide."
    ),
]

_DEFAULT_PROMPTS = [
    "Summarize why retrieval-augmented generation reduces hallucination.",
    "Compare diffusion models vs autoregressive models for image generation.",
    "Rewrite this query for better retrieval: papers about robust RL transfer.",
]


def run_benchmark(output: Path, iterations: int) -> None:
    client = GatewayClient()
    rows = []
    for tier in ("cheap", "balanced", "premium"):
        for i in range(iterations):
            for prompt in _DEFAULT_PROMPTS:
                t0 = time.perf_counter()
                res = client.invoke(GatewayRequest(prompt=prompt, tier=tier, role="agent"))
                rows.append(
                    {
                        "tier": tier,
                        "backend": res.backend,
                        "model": res.model,
                        "latency_ms": res.usage.latency_ms,
                        "estimated_cost_usd": res.usage.estimated_cost_usd,
                        "tokens_total": res.usage.total_tokens,
                        "iteration": i,
                        "elapsed_ms_wall": int((time.perf_counter() - t0) * 1000),
                        "quantization": _quantization_metadata(res.backend, res.model),
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    print(f"Wrote benchmark snapshot: {output}")


def run_cache_benchmark(output: Path) -> None:
    """Measure cold vs warm latency to quantify provider-side prefix caching.

    Makes each prompt twice per tier in sequence. The second call may hit the
    provider's prefix cache (Claude: automatic for prompts >1024 tokens with the
    prompt-caching beta header; OpenAI: automatic for prompts >1024 tokens).

    Records cold/warm latency, cost delta, and cache_read_tokens when the
    provider exposes them (Claude response_metadata includes cache_read_input_tokens).
    """
    client = GatewayClient()
    rows = []
    for tier in ("cheap", "balanced", "premium"):
        for prompt in _CACHE_PROMPTS:
            # Cold call
            t0 = time.perf_counter()
            res_cold = client.invoke(GatewayRequest(prompt=prompt, tier=tier, role="agent"))
            cold_ms = int((time.perf_counter() - t0) * 1000)

            # Warm call — identical prompt, may hit provider prefix cache
            t0 = time.perf_counter()
            res_warm = client.invoke(GatewayRequest(prompt=prompt, tier=tier, role="agent"))
            warm_ms = int((time.perf_counter() - t0) * 1000)

            cold_meta = getattr(res_cold.raw, "response_metadata", {}) or {}
            warm_meta = getattr(res_warm.raw, "response_metadata", {}) or {}

            rows.append(
                {
                    "tier": tier,
                    "backend": res_cold.backend,
                    "model": res_cold.model,
                    "prompt_chars": len(prompt),
                    "cold_latency_ms": cold_ms,
                    "warm_latency_ms": warm_ms,
                    "latency_reduction_pct": round((cold_ms - warm_ms) / cold_ms * 100, 1) if cold_ms > 0 else 0.0,
                    "cold_cost_usd": res_cold.usage.estimated_cost_usd,
                    "warm_cost_usd": res_warm.usage.estimated_cost_usd,
                    "cold_cache_read_tokens": _extract_cache_tokens(cold_meta),
                    "warm_cache_read_tokens": _extract_cache_tokens(warm_meta),
                }
            )
            print(
                f"  [{tier}/{res_cold.model}] cold={cold_ms}ms  warm={warm_ms}ms  "
                f"reduction={rows[-1]['latency_reduction_pct']}%  "
                f"warm_cache_tokens={rows[-1]['warm_cache_read_tokens']}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    print(f"Wrote cache benchmark: {output}")


def _extract_cache_tokens(meta: dict) -> int:
    """Extract cache_read_input_tokens from Claude/OpenAI response metadata."""
    if not isinstance(meta, dict):
        return 0
    # Claude: response_metadata["usage"]["cache_read_input_tokens"]
    usage = meta.get("usage", {}) or {}
    if isinstance(usage, dict):
        val = usage.get("cache_read_input_tokens", 0)
        if val:
            return int(val)
    # OpenAI: response_metadata["prompt_tokens_details"]["cached_tokens"]
    details = meta.get("prompt_tokens_details", {}) or {}
    if isinstance(details, dict):
        val = details.get("cached_tokens", 0)
        if val:
            return int(val)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gateway benchmark snapshot.")
    parser.add_argument("--output", default="results/gateway_benchmark_snapshot.json")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Run prompt-caching benchmark (cold vs warm latency per tier).",
    )
    args = parser.parse_args()
    if args.cache:
        cache_out = Path(args.output).parent / "cache_benchmark_snapshot.json"
        run_cache_benchmark(cache_out)
    else:
        run_benchmark(Path(args.output), iterations=max(1, args.iterations))


def _quantization_metadata(backend: str, model: str) -> dict:
    backend = backend.lower()
    if backend == "vllm":
        return {
            "mode": os.getenv("VLLM_QUANTIZATION", ""),
            "dtype": os.getenv("VLLM_DTYPE", os.getenv("VLLM_TORCH_DTYPE", "auto")),
            "source": "env:vllm",
            "note": "Set VLLM_QUANTIZATION/VLLM_DTYPE for explicit values.",
        }

    if backend == "ollama":
        model_lower = model.lower()
        m = re.search(r"(q\d(?:_[a-z0-9]+)?)", model_lower)
        quant = m.group(1).upper() if m else _ollama_quant_from_api(model)
        return {
            "mode": quant,
            "dtype": "",
            "source": "ollama_show" if quant else "model_tag:ollama",
            "note": ("Parsed from Ollama model details." if quant else "Parsed from model tag when available."),
        }

    if backend in {"openai", "claude"}:
        return {
            "mode": "provider_managed",
            "dtype": "",
            "source": backend,
            "note": "Quantization is not user-configurable for hosted APIs in this harness.",
        }

    return {
        "mode": "",
        "dtype": "",
        "source": backend,
        "note": "Unknown backend quantization metadata.",
    }


def _ollama_quant_from_api(model: str) -> str:
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/show",
            data=json.dumps({"model": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        details = payload.get("details", {}) if isinstance(payload, dict) else {}
        return str(details.get("quantization_level", "")).upper()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
