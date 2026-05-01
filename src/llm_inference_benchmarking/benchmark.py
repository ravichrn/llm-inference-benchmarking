"""Independent benchmark runner for gateway tiers/backends."""

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from llm_inference_benchmarking.client import GatewayClient
from llm_inference_benchmarking.types import GatewayRequest

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gateway benchmark snapshot.")
    parser.add_argument("--output", default="results/gateway_benchmark_snapshot.json")
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
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
            "note": (
                "Parsed from Ollama model details."
                if quant
                else "Parsed from model tag when available."
            ),
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
