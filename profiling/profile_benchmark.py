"""Standalone (no Modal) inference benchmark for Nsight Systems profiling on Lambda.

Run with nsys:
    nsys profile --output profiling/profiles/fp16 --export sqlite --trace cuda,nvtx \\
        python profiling/profile_benchmark.py --mode fp16

Run standalone (quick sanity check, no profiler attached):
    python profiling/profile_benchmark.py --mode fp16

Supported modes: fp16  int8  nf4  vllm  gptq  sglang

Lambda setup (one-time):
    pip install torch==2.5.1 transformers accelerate bitsandbytes gptqmodel nvtx
    pip install vllm>=0.6.0
    pip install "sglang[all]>=0.4.0,<0.5.0" --extra-index-url https://flashinfer.ai/whl/cu124/torch2.5/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    import nvtx

    def _rng(name: str, color: str = "gray"):
        return nvtx.annotate(name, color=color)

except ImportError:
    import contextlib

    def _rng(name: str, color: str = "gray"):  # type: ignore[misc]
        return contextlib.nullcontext()


MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
GPTQ_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
_MODEL_REVISION = os.getenv("MODEL_REVISION", "main")

BENCH_PROMPTS = [
    "Summarize why retrieval-augmented generation reduces hallucination in large language models.",
    "Compare diffusion models versus autoregressive models for image generation. Pros and cons.",
    "Rewrite for semantic retrieval: papers about robust RL transfer learning.",
    "Explain the transformer attention mechanism to a software engineer with no ML background.",
    "What are the key trade-offs between model quantization and full-precision inference?",
]

_ITERS = 3
_MAX_NEW_TOKENS = 256


def _run_hf(mode: str, cache_dir: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = None
    if mode == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif mode == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    print(f"[{mode}] Loading {MODEL_ID} …")
    with _rng("load_model", color="blue"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir, revision=_MODEL_REVISION)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kw: dict[str, Any] = {"cache_dir": cache_dir, "device_map": "auto", "torch_dtype": torch.float16}
        if bnb_config:
            load_kw["quantization_config"] = bnb_config
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=_MODEL_REVISION, **load_kw)
        model.eval()
        load_time_s = time.perf_counter() - t0

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    with _rng("warmup", color="gray"):
        inputs = tokenizer(BENCH_PROMPTS[0], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=32, do_sample=False)

    latencies_ms: list[float] = []
    with _rng("latency", color="green"):
        for prompt in BENCH_PROMPTS * _ITERS:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    thr_tokens: list[int] = []
    thr_times: list[float] = []
    with _rng("throughput", color="orange"):
        for _ in range(2):
            inputs = tokenizer(BENCH_PROMPTS[0], return_tensors="pt").to(model.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            thr_times.append(time.perf_counter() - t0)
            thr_tokens.append(out.shape[1] - inputs["input_ids"].shape[1])

    latencies_ms.sort()
    n = len(latencies_ms)
    return {
        "mode": mode,
        "gpu": gpu_name,
        "load_time_s": round(load_time_s, 2),
        "vram_mb": round(vram_mb, 1),
        "latency_mean_ms": round(sum(latencies_ms) / n, 1),
        "latency_p50_ms": round(latencies_ms[n // 2], 1),
        "latency_p95_ms": round(latencies_ms[int(n * 0.95)], 1),
        "output_tps": round(sum(thr_tokens) / sum(thr_times), 1),
    }


def _run_vllm(cache_dir: str) -> dict[str, Any]:
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"

    import torch
    from vllm import LLM, SamplingParams

    print("[vllm] Loading via vLLM …")
    with _rng("load_model", color="blue"):
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        llm = LLM(
            model=MODEL_ID, dtype="float16", gpu_memory_utilization=0.85, max_model_len=4096, download_dir=cache_dir
        )
        load_time_s = time.perf_counter() - t0
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    gpu_name = torch.cuda.get_device_name(0)
    lat_params = SamplingParams(max_tokens=_MAX_NEW_TOKENS, temperature=0.0)
    thr_params = SamplingParams(max_tokens=512, temperature=0.0)

    with _rng("warmup", color="gray"):
        llm.generate([BENCH_PROMPTS[0]], SamplingParams(max_tokens=32, temperature=0.0), use_tqdm=False)

    latencies_ms: list[float] = []
    with _rng("latency", color="green"):
        for prompt in BENCH_PROMPTS * _ITERS:
            t0 = time.perf_counter()
            llm.generate([prompt], lat_params, use_tqdm=False)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    thr_tokens: list[int] = []
    thr_times: list[float] = []
    with _rng("throughput", color="orange"):
        for _ in range(2):
            t0 = time.perf_counter()
            outs = llm.generate([BENCH_PROMPTS[0]], thr_params, use_tqdm=False)
            thr_times.append(time.perf_counter() - t0)
            thr_tokens.append(sum(len(o.token_ids) for out in outs for o in out.outputs))

    latencies_ms.sort()
    n = len(latencies_ms)
    return {
        "mode": "vllm",
        "gpu": gpu_name,
        "load_time_s": round(load_time_s, 2),
        "vram_mb": round(vram_mb, 1),
        "latency_mean_ms": round(sum(latencies_ms) / n, 1),
        "latency_p50_ms": round(latencies_ms[n // 2], 1),
        "latency_p95_ms": round(latencies_ms[int(n * 0.95)], 1),
        "output_tps": round(sum(thr_tokens) / sum(thr_times), 1),
    }


def _run_gptq(cache_dir: str) -> dict[str, Any]:
    import gptqmodel.quantization.config as _gptq_cfg
    import torch
    from gptqmodel import GPTQModel
    from transformers import AutoTokenizer

    # gptqmodel>=1.0 rejects is_marlin_format; patch so older checkpoints load cleanly
    _orig_fqc = _gptq_cfg.QuantizeConfig.from_quant_config.__func__

    @classmethod  # type: ignore[misc]
    def _patched_fqc(cls, config_dict, fmt=None):  # type: ignore[misc]
        if config_dict.pop("is_marlin_format", False) and fmt is None:
            fmt = "marlin"
        return _orig_fqc(cls, config_dict, fmt)

    _gptq_cfg.QuantizeConfig.from_quant_config = _patched_fqc

    print("[gptq] Loading via gptqmodel …")
    with _rng("load_model", color="blue"):
        tokenizer = AutoTokenizer.from_pretrained(GPTQ_MODEL, cache_dir=cache_dir, revision=_MODEL_REVISION)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model = GPTQModel.from_quantized(GPTQ_MODEL, cache_dir=cache_dir, device="cuda:0")
        model.eval()
        load_time_s = time.perf_counter() - t0
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    gpu_name = torch.cuda.get_device_name(0)

    with _rng("warmup", color="gray"):
        inputs = tokenizer(BENCH_PROMPTS[0], return_tensors="pt").to("cuda")
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=32, do_sample=False)

    latencies_ms: list[float] = []
    with _rng("latency", color="green"):
        for prompt in BENCH_PROMPTS * _ITERS:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    thr_tokens: list[int] = []
    thr_times: list[float] = []
    with _rng("throughput", color="orange"):
        for _ in range(2):
            inputs = tokenizer(BENCH_PROMPTS[0], return_tensors="pt").to("cuda")
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            thr_times.append(time.perf_counter() - t0)
            thr_tokens.append(out.shape[1] - inputs["input_ids"].shape[1])

    latencies_ms.sort()
    n = len(latencies_ms)
    return {
        "mode": "gptq",
        "gpu": gpu_name,
        "load_time_s": round(load_time_s, 2),
        "vram_mb": round(vram_mb, 1),
        "latency_mean_ms": round(sum(latencies_ms) / n, 1),
        "latency_p50_ms": round(latencies_ms[n // 2], 1),
        "latency_p95_ms": round(latencies_ms[int(n * 0.95)], 1),
        "output_tps": round(sum(thr_tokens) / sum(thr_times), 1),
    }


def _run_sglang(cache_dir: str) -> dict[str, Any]:
    import asyncio

    import torch

    asyncio.set_event_loop(asyncio.new_event_loop())
    import sglang as sgl

    print("[sglang] Loading via SGLang …")
    with _rng("load_model", color="blue"):
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        engine = sgl.Engine(model_path=MODEL_ID, dtype="float16", mem_fraction_static=0.85, download_dir=cache_dir)
        load_time_s = time.perf_counter() - t0
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    asyncio.set_event_loop(asyncio.new_event_loop())
    gpu_name = torch.cuda.get_device_name(0)

    def _gen(prompts: list[str], max_tokens: int) -> list[dict]:
        return engine.generate(prompts, sampling_params={"max_new_tokens": max_tokens, "temperature": 0.0})

    with _rng("warmup", color="gray"):
        _gen([BENCH_PROMPTS[0]], 32)

    latencies_ms: list[float] = []
    with _rng("latency", color="green"):
        for prompt in BENCH_PROMPTS * _ITERS:
            t0 = time.perf_counter()
            _gen([prompt], _MAX_NEW_TOKENS)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    thr_tokens: list[int] = []
    thr_times: list[float] = []
    with _rng("throughput", color="orange"):
        for _ in range(2):
            t0 = time.perf_counter()
            outs = _gen([BENCH_PROMPTS[0]], 512)
            thr_times.append(time.perf_counter() - t0)
            meta = (outs[0].get("meta_info") or {}) if isinstance(outs[0], dict) else {}
            thr_tokens.append(meta.get("completion_tokens") or 512)

    latencies_ms.sort()
    n = len(latencies_ms)
    return {
        "mode": "sglang",
        "gpu": gpu_name,
        "load_time_s": round(load_time_s, 2),
        "vram_mb": round(vram_mb, 1),
        "latency_mean_ms": round(sum(latencies_ms) / n, 1),
        "latency_p50_ms": round(latencies_ms[n // 2], 1),
        "latency_p95_ms": round(latencies_ms[int(n * 0.95)], 1),
        "output_tps": round(sum(thr_tokens) / sum(thr_times), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["fp16", "int8", "nf4", "vllm", "gptq", "sglang"])
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/model-cache"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    os.environ["HF_HOME"] = args.cache_dir
    os.makedirs(args.cache_dir, exist_ok=True)

    dispatch = {
        "fp16": lambda: _run_hf("fp16", args.cache_dir),
        "int8": lambda: _run_hf("int8", args.cache_dir),
        "nf4": lambda: _run_hf("nf4", args.cache_dir),
        "vllm": lambda: _run_vllm(args.cache_dir),
        "gptq": lambda: _run_gptq(args.cache_dir),
        "sglang": lambda: _run_sglang(args.cache_dir),
    }
    result = dispatch[args.mode]()

    lat = result["latency_mean_ms"]
    tps = result["output_tps"]
    vram = result["vram_mb"]
    print(f"\n[{args.mode}] lat={lat:.0f}ms  tps={tps:.1f}  vram={vram:.0f}MB")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
