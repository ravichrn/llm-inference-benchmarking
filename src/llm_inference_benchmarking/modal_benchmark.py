"""Modal GPU quantization benchmark for Llama-3.1-8B-Instruct (ungated mirror).

Measures all key metrics across quantization modes (fp16, int8, nf4, spec-dec,
vllm, gptq, gptq-triton, awq, fp8, flash-attn, torch-compile, tensor-parallel,
continuous-batching, cpu-llama-cpp):
  - Latency: mean, p50, p95, p99, time-to-first-token (ms)
  - Throughput: output tokens/sec, total tokens/sec
  - Memory: peak GPU VRAM (MB)
  - Perplexity: on WikiText-2 test set (128-token stride)
  - Quality: zero-shot accuracy on a 50-question MMLU subset

Prerequisites:
  modal setup          # authenticate with your Modal account (one-time)

Optional .env overrides (passed via modal.Secret.from_dict from the env at launch time):
  HUGGING_FACE_HUB_TOKEN=<token>   # only needed if switching to a gated model
  QUANT_GPTQ_MODEL=<hf_model_id>   # override default GPTQ checkpoint
  GGUF_REPO=<hf_repo_id>           # override default GGUF repo for cpu-llama-cpp mode

Run:
  modal run src/llm_inference_benchmarking/modal_benchmark.py
  modal run src/llm_inference_benchmarking/modal_benchmark.py --modes fp16,nf4,gptq,flash-attn
  modal run src/llm_inference_benchmarking/modal_benchmark.py --output results/bench.json
  modal run src/llm_inference_benchmarking/modal_benchmark.py --model meta-llama/Llama-3.1-70B-Instruct --gpu H100
  modal run src/llm_inference_benchmarking/modal_benchmark.py --modes tensor-parallel --gpu A100-80GB
  modal run src/llm_inference_benchmarking/modal_benchmark.py --modes continuous-batching
  modal run src/llm_inference_benchmarking/modal_benchmark.py --modes cpu-llama-cpp
  modal run src/llm_inference_benchmarking/modal_benchmark.py --modes sglang
  modal run src/llm_inference_benchmarking/modal_benchmark.py --modes vllm,sglang --merge
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import modal

from llm_inference_benchmarking.flops import build_flops_funnel
from llm_inference_benchmarking.mmlu import (
    _load_mmlu_questions,
    _measure_quality_vllm,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ungated community mirror — identical weights to meta-llama/Llama-3.1-8B-Instruct,
# no HuggingFace token required.
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"


# Pre-quantized GPTQ checkpoint (ungated). Override via QUANT_GPTQ_MODEL in .env.
_DEFAULT_GPTQ_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

# Pre-quantized AWQ checkpoint (ungated). Override via QUANT_AWQ_MODEL in .env.
_DEFAULT_AWQ_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"


# Draft model for speculative decoding — same tokenizer vocab, ~2 GB VRAM.
DRAFT_MODEL = "unsloth/Llama-3.2-1B-Instruct"

# Model revision (git commit SHA) for reproducibility.
# Override via MODEL_REVISION env var; defaults to "main" (latest).
# To pin: set to the exact HF commit sha, e.g. "abc1234".
_MODEL_REVISION = os.getenv("MODEL_REVISION", "main")

_ALL_MODES = (
    "fp16",
    "int8",
    "nf4",
    "spec-dec",
    "vllm",
    "sglang",
    "gptq",
    "gptq-triton",
    "awq",
    "fp8",
    "flash-attn",
    "torch-compile",
    "tensor-parallel",
    "continuous-batching",
    "cpu-q4km",
)

# Modes that require multiple GPUs (dispatched to run_tp_benchmark instead of run_quant_benchmark)
_MULTI_GPU_MODES = frozenset({"tensor-parallel"})

# Modes that require a CPU-only container (dispatched to run_cpu_benchmark)
_CPU_MODES = frozenset({"cpu-q4km"})

# Modes that use the SGLang engine image (separate from vLLM to avoid flashinfer conflicts)
_SGLANG_MODES = frozenset({"sglang"})

# Default modes for a standard A10G sweep — excludes specialist modes that need
# a different GPU (fp8 needs H100, tensor-parallel needs 2xA100) or CPU-only containers.
_DEFAULT_MODES = tuple(m for m in _ALL_MODES if m not in _CPU_MODES and m not in _MULTI_GPU_MODES and m != "fp8")

# Default GGUF repo for cpu-llama-cpp modes; override via GGUF_REPO env var
_DEFAULT_GGUF_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"

# GGUF quantization levels and their mode names (ordered cheap→expensive in quality)
_GGUF_LEVELS: list[tuple[str, str]] = [
    ("Q4_K_M", "cpu-q4km"),
]

# Prompts for latency / throughput measurement
_BENCH_PROMPTS = [
    "Summarize why retrieval-augmented generation reduces hallucination in large language models.",
    "Compare diffusion models versus autoregressive models for image generation. Pros and cons.",
    "Rewrite for semantic retrieval: papers about robust RL transfer learning.",
    "Explain the transformer attention mechanism to a software engineer with no ML background.",
    "What are the key trade-offs between model quantization and full-precision inference?",
]

# Model config for FLOPs accounting (Llama-3.1-8B architecture)
_LLAMA_8B_CONFIG = {
    "num_params": 8_000_000_000,
    "num_layers": 32,
    "hidden": 4096,
    "seq_len": 512,
}


# Per-mode notes surfaced in output JSON — explains known caveats so results are self-interpreting
_MODE_NOTES: dict[str, str] = {
    "int8": (
        "bitsandbytes int8 is compute-bound on A10G: dequantize-then-multiply adds "
        "overhead vs fp16's native tensor cores. int8 saves VRAM but reduces throughput. "
        "Use fp16 if VRAM allows; use nf4 for the best speed/memory trade-off."
    ),
    "spec-dec": (
        "Speculative decoding uses a 1B draft model to propose tokens that the 8B target "
        "verifies in one parallel pass. Output is mathematically identical to target-only "
        "greedy decoding. TTFT is measured without the draft model (prefill-only baseline) "
        "so it is directly comparable to other modes. Throughput gain depends on draft "
        "acceptance rate — higher on predictable/repetitive text, lower on diverse prompts."
    ),
    "gptq": (
        "GPTQ INT4 loaded via gptqmodel with the exllama2 CUDA backend. Marlin fused kernels "
        "give fastest prefill on A10G. Compare against gptq-triton to isolate kernel backend "
        "impact, and against awq for accuracy/speed tradeoff at same bitwidth."
    ),
    "gptq-triton": (
        "GPTQ INT4 with the Triton kernel backend instead of exllama2. Triton-generated CUDA "
        "kernels are more portable across GPU architectures but typically 10-30% slower than "
        "Marlin/exllama2 on A10G Ampere. Useful for measuring the performance gap between "
        "hand-tuned CUDA kernels (exllama2/Marlin) and compiler-generated Triton kernels, and "
        "for architectures where exllama2 kernels are not available."
    ),
    "awq": (
        "AWQ (Activation-aware Weight Quantization) INT4 via vLLM backend. "
        "Calibrates quantization scales using activation statistics so salient channels are "
        "preserved at higher precision. autoawq was tested first but awq_ext kernels are not "
        "compiled for A10G Ampere, causing a silent fallback to ~161ms/token. vLLM's built-in "
        "AWQ kernels work correctly on A10G and give comparable throughput to GPTQ at INT4."
    ),
    "fp8": (
        "FP8 (8-bit floating point) quantization via vLLM's dynamic per-tensor scaling. "
        "Uses E4M3 format (4 exponent bits, 3 mantissa bits). On A10G (Ampere sm_86), "
        "FP8 runs in software emulation — no hardware FP8 tensor cores. H100 (Hopper sm_90) "
        "has native FP8 support and will show true speedup. Results here reflect SW emulation "
        "overhead; run on H100 for representative production numbers."
    ),
    "flash-attn": (
        "Flash Attention 2 rewrites the attention kernel to avoid materialising the full "
        "NxN attention matrix. Instead it tiles Q/K/V into SRAM blocks, computing "
        "softmax and matmul in a single fused pass. This reduces memory from O(n²) to O(n) "
        "and improves arithmetic intensity. Throughput gains are most visible at long "
        "sequence lengths (>2k tokens); for short prompts the difference vs fp16 is small."
    ),
    "torch-compile": (
        "torch.compile() applies TorchInductor JIT compilation to the model forward pass. "
        "The first call triggers graph capture and kernel fusion — expect a 2-5 minute "
        "warm-up penalty. Subsequent calls use the compiled graph with fused CUDA kernels. "
        "mode='reduce-overhead' minimises Python dispatch overhead. Gains are most "
        "significant for decode-heavy workloads; prefill is less affected."
    ),
    "tensor-parallel": (
        "Tensor parallelism (TP=2) splits each weight matrix column-wise across two GPUs "
        "via vLLM's tensor_parallel_size flag. Each GPU holds half the attention heads and "
        "MLP neurons; an all-reduce collective synchronises activations after each layer. "
        "Requires a multi-GPU instance (A100-80GBx2 or H100x2). The primary benefit is "
        "fitting larger models in VRAM; for 8B models on 80 GB GPUs the memory pressure is "
        "low so throughput gains come mainly from doubled memory bandwidth."
    ),
    "continuous-batching": (
        "Continuous batching (iteration-level scheduling) allows the vLLM engine to insert "
        "new requests mid-sequence rather than waiting for the whole batch to finish. "
        "This benchmark sends a stream of concurrent requests to the async vLLM engine and "
        "measures per-request latency, effective batch size over time, and GPU utilisation. "
        "Key metrics: mean/p99 request latency, queue depth at steady state, requests/sec, "
        "and effective batch size distribution. Compare against single-request vllm mode to "
        "quantify the throughput uplift from batching."
    ),
    "cpu-q4km": (
        "GGUF Q4_K_M quantization (≈4.8 bits/weight) via llama.cpp. The practical sweet spot: "
        "~4.9 GB file, quality close to fp16, throughput ~5-10 tok/s on 8 CPU cores. This is "
        "the default quantization level for consumer and edge deployments of llama.cpp."
    ),
    "cpu-q5km": (
        "GGUF Q5_K_M quantization (≈5.7 bits/weight) via llama.cpp. Slightly higher quality "
        "than Q4_K_M with ~5.7 GB file size. Recommended when Q4_K_M shows accuracy loss on "
        "domain-specific tasks. Throughput is ~10-15% lower than Q4_K_M on the same hardware."
    ),
    "cpu-q8_0": (
        "GGUF Q8_0 quantization (≈8 bits/weight) via llama.cpp — closest to fp16 quality in "
        "GGUF format, ~8.5 GB file. Minimal quality degradation vs fp16 but 2x the file size "
        "of Q4_K_M. Useful as the CPU accuracy ceiling for comparison against GPU fp16 results."
    ),
    "sglang": (
        "SGLang offline engine with RadixAttention (prefix-tree KV cache) and chunked prefill. "
        "RadixAttention reuses cached KV activations across requests that share a common prefix, "
        "reducing redundant prefill compute for multi-turn chat and RAG workloads. "
        "Compare latency/throughput directly against vllm mode — both use PagedAttention-style "
        "continuous batching, but SGLang's radix cache gives additional gains when prompts share "
        "a long system prompt or retrieval context."
    ),
}

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------

_image = (
    # CUDA 12.8 ships libnvJitLink.so.13 (SONAME bumped at 12.6+), required by bitsandbytes>=0.46.1.
    # 12.4 only has libnvJitLink.so.12 which caused "cannot open shared object" errors.
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("clang")  # gptqmodel pypcre C extension requires clang
    .pip_install("wheel", "setuptools", "packaging")
    .pip_install("torch==2.5.1", "numpy<2.0")
    .pip_install(
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        # nvidia-nvjitlink-cu12 ships libnvJitLink into site-packages/nvidia/nvjitlink/lib/;
        # bitsandbytes>=0.46.1 links against libnvJitLink.so.13 (CUDA 13 SONAME) but CUDA 12.x
        # only has .so.12 — we install the pip package so the file is guaranteed present, then
        # symlink it as .so.13 and register with ldconfig in the run_commands step below.
        "nvidia-nvjitlink-cu12",
        "bitsandbytes>=0.46.1",
        "datasets>=2.20.0",
        "huggingface_hub",
    )
    .run_commands(
        # Find libnvJitLink.so.12* from the pip nvidia package or system CUDA, create .so.13
        # symlink in the same dir, and register that dir with ldconfig so ctypes.CDLL finds it.
        "NVJIT=$(find /usr/local/lib/python3.11/site-packages/nvidia/nvjitlink/lib"
        " /usr/local/cuda/lib64 /usr/local/cuda-12.8/lib64 /usr/lib/x86_64-linux-gnu"
        " -name 'libnvJitLink.so.12*' 2>/dev/null | head -1) && "
        '[ -n "$NVJIT" ] && DIR=$(dirname "$NVJIT") && '
        'ln -sf "$NVJIT" "$DIR/libnvJitLink.so.13" && '
        'echo "$DIR" > /etc/ld.so.conf.d/nvjitlink.conf && ldconfig && '
        'echo "OK: created $DIR/libnvJitLink.so.13" '
        '|| { echo "FATAL: libnvJitLink.so.12 not found"; exit 1; }'
    )
    # gptqmodel: maintained successor to auto-gptq, no optimum dependency
    .pip_install("gptqmodel>=1.0.0")
    # AWQ quantization with fused CUDA kernels
    .pip_install("autoawq>=0.2.5")
    # NVTX markers for Nsight Systems profiling (no-op when Nsight not attached)
    .pip_install("nvtx")
    # flash-attn binary has frequent ABI issues; use torch SDPA backend instead (same kernels on A10G)
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # bitsandbytes ctypes.CDLL resolves libnvJitLink.so.13 via LD_LIBRARY_PATH at runtime.
            # The .so.13 symlink is created in the nvidia pip package dir during run_commands above.
            # ldconfig alone isn't reliable here because Modal may not persist the ld.so.cache
            # snapshot; LD_LIBRARY_PATH is checked by dlopen() directly from the process environment.
            "LD_LIBRARY_PATH": (
                "/usr/local/lib/python3.11/site-packages/nvidia/nvjitlink/lib"
                ":/usr/local/cuda/lib64"
                ":/usr/local/cuda-12.8/lib64"
            ),
        }
    )
    .pip_install("hf-transfer")
    # vLLM for the "vllm" and "fp8" benchmark modes — installed last to avoid CUDA conflicts
    .pip_install("vllm>=0.6.0")
    # Bundle the local package so flops.py, mmlu.py, etc. are importable in the container
    .add_local_python_source("llm_inference_benchmarking")
)

# CPU-only image: llama-cpp-python (pre-built wheel) + huggingface_hub for GGUF download
_cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        # --prefer-binary downloads a pre-built wheel for llama-cpp-python (no C++ compilation).
        # The wheel enables AVX2/AVX-512 VNNI kernels on modern x86 CPU containers.
        "pip install 'llama-cpp-python>=0.3.0' --prefer-binary",
        "pip install 'huggingface_hub>=0.23.0'",
    )
    .add_local_python_source("llm_inference_benchmarking")
)

# SGLang image — uses CUDA 12.4 base to match the flashinfer wheel index and avoid
# sgl_kernel picking SM100 (Blackwell) kernels when the Modal driver reports CUDA 13.x.
# Pin sglang to 0.4.x: these versions ship SM86 (A10G Ampere) kernels in sgl_kernel.
_sglang_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("libnuma-dev")
    .pip_install("wheel", "setuptools", "packaging")
    .pip_install("torch==2.5.1", "numpy<2.0")
    .pip_install(
        "sglang[all]>=0.4.0,<0.5.0",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5/",
    )
    .pip_install(
        "transformers>=4.44.0",
        "huggingface_hub",
        "hf-transfer",
        "nvtx",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("llm_inference_benchmarking")
)

# Persistent volume for caching downloaded model weights
_model_cache = modal.Volume.from_name("llm-quant-model-cache", create_if_missing=True)
_secret_payload = {k: v for k in ("HUGGING_FACE_HUB_TOKEN",) if (v := os.getenv(k, "").strip())}
_modal_secrets = [modal.Secret.from_dict(_secret_payload)] if _secret_payload else []

app = modal.App("llm-quant-benchmark")

# ---------------------------------------------------------------------------
# Remote benchmark function
# ---------------------------------------------------------------------------


@app.function(
    gpu=os.environ.get("MODAL_GPU", "A10G"),
    image=_image,
    timeout=7200,
    volumes={"/model-cache": _model_cache},
    secrets=_modal_secrets,
    memory=32768,
)
def run_quant_benchmark(quant_mode: str, model_id: str = "") -> dict[str, Any]:
    """Run all metrics for one quantization mode. Executed remotely on Modal GPU.

    Args:
        quant_mode: One of _ALL_MODES (e.g. "fp16", "gptq", "flash-attn").
        model_id:   HuggingFace model ID to benchmark. Defaults to BASE_MODEL when empty.
                    Pass a different ID to run cross-model comparisons, e.g.
                    "mistralai/Mistral-7B-Instruct-v0.3" or a 70B model on a larger GPU.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["TRANSFORMERS_CACHE"] = "/model-cache/hf"
    os.environ["HF_HOME"] = "/model-cache/hf"
    # Optional — only needed if switching to a gated model via .env
    hf_token: str | None = os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    # vllm and fp8 use the vLLM engine path; fp8 adds dynamic FP8 quantization on top
    if quant_mode == "vllm":
        return _run_vllm_benchmark(gpu_name, hf_token, model_id=model_id)
    if quant_mode == "fp8":
        return _run_vllm_benchmark(gpu_name, hf_token, model_id=model_id, quantization="fp8")
    if quant_mode == "continuous-batching":
        return _run_continuous_batching_benchmark(gpu_name, hf_token, model_id=model_id)
    if quant_mode == "awq":
        return _run_vllm_benchmark(gpu_name, hf_token, model_id=_DEFAULT_AWQ_MODEL, quantization="awq")

    model_id, bnb_config, load_kwargs = _resolve_load_config(quant_mode, hf_token, model_id)

    print(f"[{quant_mode}] Loading {model_id} …")
    t_load_start = time.perf_counter()
    load_kw = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir="/model-cache/hf", revision=_MODEL_REVISION, **load_kw
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.reset_peak_memory_stats()
    if quant_mode in ("gptq", "gptq-triton"):
        import gptqmodel.quantization.config as _gptq_cfg
        from gptqmodel import GPTQModel

        # gptqmodel>=1.0 rejected is_marlin_format; patch the classmethod so older
        # checkpoints (hugging-quants etc.) load without changing the checkpoint files.
        _orig_fqc = _gptq_cfg.QuantizeConfig.from_quant_config.__func__

        @classmethod  # type: ignore[misc]
        def _patched_fqc(cls, config_dict, fmt=None):  # type: ignore[misc]
            if config_dict.pop("is_marlin_format", False) and fmt is None:
                fmt = "marlin"
            return _orig_fqc(cls, config_dict, fmt)

        _gptq_cfg.QuantizeConfig.from_quant_config = _patched_fqc
        # gptq-triton: force Triton kernel backend instead of exllama2/Marlin CUDA kernels.
        # backend="triton" instructs gptqmodel to use Triton-generated CUDA kernels for the
        # quantized matmul ops, bypassing the hand-tuned exllama2 kernels. This lets us
        # measure the perf gap between compiler-generated (Triton) and hand-tuned (Marlin) kernels.
        gptq_kwargs: dict[str, Any] = {
            "cache_dir": "/model-cache/hf",
            "device": "cuda:0",
            **load_kw,
        }
        if quant_mode == "gptq-triton":
            gptq_kwargs["backend"] = "triton"
        model = GPTQModel.from_quantized(model_id, **gptq_kwargs)
        model.eval()
    else:
        pretrained_kwargs: dict[str, Any] = {
            "cache_dir": "/model-cache/hf",
            "device_map": "auto",
            **load_kw,
            **load_kwargs,
        }
        if bnb_config is not None:
            pretrained_kwargs["quantization_config"] = bnb_config
            # Newer transformers raises RuntimeError on CONVERSION log entries during bnb loading;
            # suppress the report — the actual quantization still runs correctly.
            try:
                import transformers.utils.loading_report as _lr

                _lr.log_state_dict_report = lambda *a, **kw: None
            except Exception:  # nosec B110 — optional monkey-patch; attribute may not exist in all versions
                pass
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=_MODEL_REVISION, **pretrained_kwargs)
        model.eval()
    # torch.compile: JIT-compile the forward pass after loading — first inference call
    # triggers graph capture (slow), subsequent calls use fused CUDA kernels.
    if quant_mode == "torch-compile":
        import torch as _torch

        # reduce-overhead uses CUDA graphs which break with growing KV cache across generate() calls;
        # default mode still fuses kernels via TorchInductor without capturing static graphs
        model.forward = _torch.compile(model.forward, mode="default")
    load_time_s = time.perf_counter() - t_load_start
    model_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[{quant_mode}] Model loaded in {load_time_s:.1f}s  ({model_vram_mb:.0f} MB VRAM)")
    _model_cache.commit()  # persist downloaded weights so next run skips the download

    # Clear max_length from the model's generation_config so it never conflicts with
    # max_new_tokens passed to model.generate() — avoids a noisy transformers warning.
    if hasattr(model, "generation_config") and getattr(model.generation_config, "max_length", None):
        model.generation_config.max_length = None

    # Load draft model for speculative decoding
    draft_model = None
    if quant_mode == "spec-dec":
        print(f"[{quant_mode}] Loading draft model {DRAFT_MODEL} …")
        draft_model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL,
            revision=_MODEL_REVISION,
            cache_dir="/model-cache/hf",
            device_map="auto",
            dtype=torch.float16,
            **load_kw,
        )
        draft_model.eval()
        draft_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[{quant_mode}] Draft model loaded  ({draft_vram_mb - model_vram_mb:.0f} MB VRAM)")
        _model_cache.commit()

    result: dict[str, Any] = {
        "quant_mode": quant_mode,
        "model_id": model_id,
        "gpu": gpu_name,
        "load_time_s": round(load_time_s, 2),
    }

    if quant_mode == "spec-dec":
        result["draft_model_id"] = DRAFT_MODEL
        result["memory"] = {
            "model_weights_mb": round(model_vram_mb, 1),
            "draft_weights_mb": round(draft_vram_mb - model_vram_mb, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        }
    else:
        result["memory"] = _measure_memory(model_vram_mb)
    result["latency"] = _measure_latency(model, tokenizer, device, assistant_model=draft_model)
    result["throughput"] = _measure_throughput(model, tokenizer, device, assistant_model=draft_model)
    # Speculative decoding's assisted_generation does not support batched inputs in transformers
    if quant_mode != "spec-dec":
        result["batch_throughput"] = _measure_batch_throughput(model, tokenizer, device)
    result["perplexity"] = _measure_perplexity(model, tokenizer, device)
    result["quality"] = _measure_quality(model, tokenizer, device)
    if quant_mode in _MODE_NOTES:
        result["notes"] = _MODE_NOTES[quant_mode]

    return result


# ---------------------------------------------------------------------------
# vLLM benchmark path
# ---------------------------------------------------------------------------
# AWQ benchmark helper
# ---------------------------------------------------------------------------


def _run_awq_benchmark(gpu_name: str, hf_token: str | None, model_id: str = "") -> dict[str, Any]:
    """Benchmark AWQ INT4 via autoawq with fused CUDA kernels."""
    import torch
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    awq_model_id = os.getenv("QUANT_AWQ_MODEL", _DEFAULT_AWQ_MODEL)
    load_kw = {"token": hf_token} if hf_token else {}

    print(f"[awq] Loading {awq_model_id} …")
    t_load_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        awq_model_id, revision=_MODEL_REVISION, cache_dir="/model-cache/hf", **load_kw
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoAWQForCausalLM.from_quantized(
        awq_model_id,
        revision=_MODEL_REVISION,
        cache_dir="/model-cache/hf",
        fuse_layers=False,  # fused kernels (awq_ext) not compiled for A10G Ampere; unfused path used
        **load_kw,
    )
    model.eval()
    load_time_s = round(time.perf_counter() - t_load_start, 2)
    model_vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    device = "cuda:0"
    latency = _measure_latency(model, tokenizer, device)
    throughput = _measure_throughput(model, tokenizer, device)
    batch_throughput = _measure_batch_throughput(model, tokenizer, device)
    quality = _measure_quality(model, tokenizer, device)

    result: dict[str, Any] = {
        "quant_mode": "awq",
        "model_id": awq_model_id,
        "gpu": gpu_name,
        "load_time_s": load_time_s,
        "memory": {"model_weights_mb": model_vram_mb, "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1)},
        "latency": latency,
        "throughput": throughput,
        "batch_throughput": batch_throughput,
        "perplexity": None,
        "quality": quality,
        "notes": _MODE_NOTES["awq"],
    }
    _model_cache.commit()
    return build_flops_funnel(result, _LLAMA_8B_CONFIG, gpu_name, "awq")


# ---------------------------------------------------------------------------


def _run_vllm_benchmark(
    gpu_name: str,
    hf_token: str | None,
    model_id: str = "",
    quantization: str | None = None,
) -> dict[str, Any]:
    """Benchmark the model via vLLM's LLM engine (PagedAttention, continuous batching).

    Args:
        model_id:     HF model to load. Defaults to BASE_MODEL when empty.
        quantization: Optional vLLM quantization scheme, e.g. "fp8" for dynamic FP8.
                      On A10G (Ampere), fp8 runs in software emulation; H100 uses hardware.
    """
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache/hf"
    os.environ["HF_HOME"] = "/model-cache/hf"
    # Must be set before vllm import — checked at import time, not engine creation
    os.environ["VLLM_USE_V1"] = "0"
    # deep_gemm not installed in image; skip its warmup pass for fp8
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"

    import torch
    from vllm import LLM, SamplingParams

    effective_model = model_id or BASE_MODEL
    quant_mode_label = quantization if quantization else "vllm"

    load_kw: dict[str, Any] = {}
    if hf_token:
        load_kw["tokenizer_revision"] = _MODEL_REVISION

    print(f"[{quant_mode_label}] Loading {effective_model} via vLLM engine …")
    t_load_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    llm_kwargs: dict[str, Any] = {
        "model": effective_model,
        "revision": _MODEL_REVISION,
        "dtype": "float16",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 4096,  # cap context to fit KV cache in remaining VRAM after fp16 weights
        "download_dir": "/model-cache/hf",
        "trust_remote_code": False,
    }
    if quantization:
        llm_kwargs["quantization"] = quantization

    llm = LLM(**llm_kwargs)
    load_time_s = time.perf_counter() - t_load_start
    model_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    print(f"[{quant_mode_label}] Engine ready in {load_time_s:.1f}s  ({model_vram_mb:.0f} MB VRAM)")
    _model_cache.commit()

    # Warmup
    _WARMUP = 1
    _ITERS = 3
    warmup_params = SamplingParams(max_tokens=32, temperature=0.0)
    for _ in range(_WARMUP):
        llm.generate([_BENCH_PROMPTS[0]], warmup_params, use_tqdm=False)

    # Latency (single request, 256 output tokens)
    lat_params = SamplingParams(max_tokens=256, temperature=0.0)
    latencies_ms: list[float] = []
    ttfts_ms: list[float] = []
    for prompt in _BENCH_PROMPTS * _ITERS:
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], lat_params, use_tqdm=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)
        # vLLM exposes per-token timing via output metrics
        out = outputs[0]
        if hasattr(out, "metrics") and out.metrics is not None and hasattr(out.metrics, "first_token_time"):
            ttft = (out.metrics.first_token_time - out.metrics.first_scheduled_time) * 1000
            ttfts_ms.append(ttft)

    latencies_ms.sort()
    lat_mean = sum(latencies_ms) / len(latencies_ms)
    lat_p50 = latencies_ms[len(latencies_ms) // 2]
    lat_p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    lat_p99 = latencies_ms[int(len(latencies_ms) * 0.99)]
    ttft_mean = sum(ttfts_ms) / len(ttfts_ms) if ttfts_ms else None

    # Throughput (512 output tokens, 2 iterations)
    thr_params = SamplingParams(max_tokens=512, temperature=0.0)
    thr_times: list[float] = []
    thr_tokens: list[int] = []
    for _ in range(2):
        t0 = time.perf_counter()
        outs = llm.generate([_BENCH_PROMPTS[0]], thr_params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        thr_times.append(elapsed)
        thr_tokens.append(sum(len(o.token_ids) for out in outs for o in out.outputs))
    output_tps = sum(thr_tokens) / sum(thr_times)

    # Batch throughput
    batch_thr: dict[str, float] = {}
    for batch_size in (1, 4, 8):
        prompts = (_BENCH_PROMPTS * batch_size)[:batch_size]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, thr_params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        tokens = sum(len(o.token_ids) for out in outs for o in out.outputs)
        batch_thr[f"batch{batch_size}_output_tokens_per_sec"] = round(tokens / elapsed, 1)

    # MMLU quality via log-prob scoring using vLLM's encode API
    quality = _measure_quality_vllm(llm)

    fp8_note = (
        " FP8 runs in software emulation on Ampere (A10G); use H100 for hardware FP8 speedup."
        if quantization == "fp8"
        else ""
    )
    result = {
        "quant_mode": quant_mode_label,
        "model_id": effective_model,
        "gpu": gpu_name,
        "load_time_s": round(load_time_s, 2),
        "memory": {
            "model_weights_mb": round(model_vram_mb, 1),
            "reserved_mb": round(reserved_mb, 1),
        },
        "latency": {
            "max_new_tokens": 256,
            "mean_ms": round(lat_mean, 1),
            "p50_ms": round(lat_p50, 1),
            "p95_ms": round(lat_p95, 1),
            "p99_ms": round(lat_p99, 1),
            "min_ms": round(min(latencies_ms), 1),
            "max_ms": round(max(latencies_ms), 1),
            "ttft_mean_ms": round(ttft_mean, 1) if ttft_mean is not None else None,
            "ttft_p95_ms": None,
            "prefill_ms": round(ttft_mean, 1) if ttft_mean is not None else None,
            "decode_ms_per_tok": round(max(lat_mean - ttft_mean, 0) / max(256 - 1, 1), 2)
            if ttft_mean is not None
            else None,
            "prefill_decode_ratio": round(ttft_mean / max(max(lat_mean - ttft_mean, 0) / max(256 - 1, 1), 0.01), 2)
            if ttft_mean is not None
            else None,
        },
        "throughput": {
            "output_tokens_per_sec": round(output_tps, 1),
            "max_new_tokens": 512,
        },
        "batch_throughput": batch_thr,
        "perplexity": None,  # vLLM does not expose per-token NLL loss
        "quality": quality,
        "notes": (
            _MODE_NOTES.get(quant_mode_label)
            or (
                "vLLM fp16 with continuous batching (PagedAttention). "
                "Perplexity is not computed — vLLM does not expose per-token NLL. "
                "Compare latency/throughput directly against the fp16 HuggingFace baseline." + fp8_note
            )
        ),
    }
    return build_flops_funnel(result, _LLAMA_8B_CONFIG, gpu_name, quant_mode_label)


# ---------------------------------------------------------------------------
# Continuous batching benchmark
# ---------------------------------------------------------------------------


def _run_continuous_batching_benchmark(
    gpu_name: str,
    hf_token: str | None,
    model_id: str = "",
) -> dict[str, Any]:
    """Measure vLLM continuous batching behaviour under concurrent load.

    Sends multiple concurrent requests to the async vLLM engine and captures:
      - per-request latency distribution (mean, p50, p95, p99)
      - effective throughput at each concurrency level (1, 4, 8, 16)
      - estimated mean batch size during decode phase
      - requests per second at steady state
    """
    import asyncio

    os.environ["TRANSFORMERS_CACHE"] = "/model-cache/hf"
    os.environ["HF_HOME"] = "/model-cache/hf"
    # Must be set before vllm import — vLLM V1 engine spawns subprocess workers
    # that fail in Modal containers and pre-allocates CUDA graphs (~2 GB overhead)
    os.environ["VLLM_USE_V1"] = "0"

    import torch
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    effective_model = model_id or BASE_MODEL
    print(f"[continuous-batching] Loading {effective_model} via AsyncLLMEngine …")

    engine_args = AsyncEngineArgs(
        model=effective_model,
        revision=_MODEL_REVISION,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,  # reduced from 4096 — saves KV cache VRAM on A10G
        enforce_eager=True,  # skip CUDA graph pre-allocation (~2 GB savings)
        download_dir="/model-cache/hf",
    )
    try:
        engine = AsyncLLMEngine.from_engine_args(engine_args)
    except (ValueError, RuntimeError) as exc:
        return {"quant_mode": "continuous-batching", "error": str(exc), "skipped": True}
    _model_cache.commit()

    sampling = SamplingParams(max_tokens=256, temperature=0.0)

    async def _send_request(req_id: str, prompt: str) -> tuple[float, int]:
        t0 = time.perf_counter()
        tokens = 0
        async for output in engine.generate(prompt, sampling, request_id=req_id):
            tokens = len(output.outputs[0].token_ids)
        return (time.perf_counter() - t0) * 1000, tokens

    async def _run_concurrency_level(concurrency: int, total: int) -> dict[str, Any]:
        sem = asyncio.Semaphore(concurrency)
        prompts = (_BENCH_PROMPTS * (total // len(_BENCH_PROMPTS) + 1))[:total]

        async def _bounded(i: int, prompt: str) -> tuple[float, int]:
            async with sem:
                return await _send_request(f"req-{i}", prompt)

        t_wall_start = time.perf_counter()
        results = await asyncio.gather(*[_bounded(i, p) for i, p in enumerate(prompts)])
        wall_s = time.perf_counter() - t_wall_start

        latencies = sorted(r[0] for r in results)
        total_tokens = sum(r[1] for r in results)
        n = len(latencies)
        return {
            "concurrency": concurrency,
            "total_requests": total,
            "wall_time_s": round(wall_s, 2),
            "requests_per_sec": round(total / wall_s, 2),
            "output_tokens_per_sec": round(total_tokens / wall_s, 1),
            "mean_latency_ms": round(sum(latencies) / n, 1),
            "p50_latency_ms": round(latencies[n // 2], 1),
            "p95_latency_ms": round(latencies[int(n * 0.95)], 1),
            "p99_latency_ms": round(latencies[int(n * 0.99)], 1),
            "estimated_mean_batch_size": round(concurrency * (sum(latencies) / n) / (wall_s * 1000 / total), 2),
        }

    async def _run_all() -> list[dict[str, Any]]:
        rows = []
        for c in (1, 4, 8, 16):
            row = await _run_concurrency_level(c, total=max(c * 3, 16))
            tps = row["output_tokens_per_sec"]
            rps = row["requests_per_sec"]
            p99 = row["p99_latency_ms"]
            print(f"  [continuous-batching] c={c:2d}  {rps:.1f} req/s  {tps:.0f} tok/s  p99={p99:.0f}ms")
            rows.append(row)
        return rows

    concurrency_results = asyncio.run(_run_all())
    model_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "quant_mode": "continuous-batching",
        "model_id": effective_model,
        "gpu": gpu_name,
        "memory": {
            "model_weights_mb": round(model_vram_mb, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        },
        "latency": concurrency_results[0],
        "throughput": {
            "output_tokens_per_sec": concurrency_results[0]["output_tokens_per_sec"],
            "max_new_tokens": 256,
        },
        "batch_throughput": {
            f"batch{r['concurrency']}_output_tokens_per_sec": r["output_tokens_per_sec"] for r in concurrency_results
        },
        "concurrency_sweep": concurrency_results,
        "perplexity": None,
        "quality": None,
        "notes": _MODE_NOTES["continuous-batching"],
    }


# ---------------------------------------------------------------------------
# Metric helpers (all run inside the Modal function)
# ---------------------------------------------------------------------------


def _resolve_load_config(quant_mode: str, hf_token: str | None, model_id: str = "") -> tuple[str, Any, dict]:
    """Return (resolved_model_id, bnb_config_or_None, extra_from_pretrained_kwargs).

    model_id: caller-supplied model override; falls back to BASE_MODEL when empty.
              AWQ and GPTQ modes always use their own pre-quantized checkpoints.
    """
    import torch
    from transformers import BitsAndBytesConfig

    base = model_id or BASE_MODEL

    if quant_mode == "fp16":
        return base, None, {"dtype": torch.float16}

    if quant_mode == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        return base, cfg, {}

    if quant_mode == "nf4":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        return base, cfg, {}

    if quant_mode == "nf4-dq":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        return base, cfg, {}

    if quant_mode == "spec-dec":
        return base, None, {"dtype": torch.float16}

    if quant_mode in ("gptq", "gptq-triton"):
        # Pre-quantized INT4 GPTQ checkpoint; gptqmodel registers the backend at load time.
        # gptq-triton uses the same checkpoint but with backend="triton" applied in run_quant_benchmark.
        gptq_id = os.environ.get("QUANT_GPTQ_MODEL", _DEFAULT_GPTQ_MODEL)
        return gptq_id, None, {}

    if quant_mode == "flash-attn":
        # torch SDPA backend uses the same FlashAttention kernels on A10G without the fragile binary
        return base, None, {"dtype": torch.float16, "attn_implementation": "sdpa"}

    if quant_mode == "torch-compile":
        # Load as fp16; torch.compile() is applied post-load in run_quant_benchmark.
        return base, None, {"dtype": torch.float16}

    raise ValueError(f"Unknown quant_mode: {quant_mode!r}")


def _measure_memory(model_vram_mb: float) -> dict[str, float]:
    import torch

    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    return {
        "model_weights_mb": round(model_vram_mb, 1),
        "reserved_mb": round(reserved_mb, 1),
    }


def _measure_latency(model: Any, tokenizer: Any, device: str, assistant_model: Any = None) -> dict[str, Any]:
    """Latency over _BENCH_PROMPTS x 5 iterations, plus TTFT."""
    import torch

    try:
        import nvtx as _nvtx
    except ImportError:
        _nvtx = None  # type: ignore[assignment]

    def _nvtx_range(name: str, color: str = "blue"):
        """Context manager that wraps nvtx.annotate when available, else no-op."""
        import contextlib

        if _nvtx is not None:
            return _nvtx.annotate(name, color=color)
        return contextlib.nullcontext()

    WARMUP = 1
    ITERS = 3
    MAX_NEW_TOKENS = 256

    gen_kw: dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_length": None,
        "do_sample": False,
    }
    if assistant_model is not None:
        gen_kw["assistant_model"] = assistant_model

    all_ms: list[float] = []
    ttft_ms_list: list[float] = []

    for prompt in _BENCH_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Warmup
        with _nvtx_range("warmup", color="gray"):
            for _ in range(WARMUP):
                with torch.no_grad():
                    model.generate(**inputs, **gen_kw)

        # TTFT: generate exactly 1 token (≈ prefill + one decode step)
        ttft_kw = {**gen_kw, "max_new_tokens": 1}
        ttft_kw.pop("assistant_model", None)  # spec-dec TTFT measured without draft for fair comparison
        with _nvtx_range("ttft", color="green"):
            for _ in range(3):
                t0 = time.perf_counter()
                with torch.no_grad():
                    model.generate(**inputs, **ttft_kw)
                ttft_ms_list.append((time.perf_counter() - t0) * 1000)

        # Full-generation latency
        with _nvtx_range("latency", color="blue"):
            for _ in range(ITERS):
                t0 = time.perf_counter()
                with torch.no_grad():
                    model.generate(**inputs, **gen_kw)
                all_ms.append((time.perf_counter() - t0) * 1000)

    all_ms.sort()
    ttft_ms_list.sort()
    n = len(all_ms)
    ttft_mean = round(statistics.mean(ttft_ms_list), 1)
    mean_total = round(statistics.mean(all_ms), 1)
    # decode phase = full generation minus prefill, divided by remaining tokens
    decode_ms_per_tok = round(max(mean_total - ttft_mean, 0) / max(MAX_NEW_TOKENS - 1, 1), 2)
    prefill_decode_ratio = round(ttft_mean / decode_ms_per_tok, 2) if decode_ms_per_tok > 0 else None
    return {
        "max_new_tokens": MAX_NEW_TOKENS,
        "mean_ms": mean_total,
        "p50_ms": round(all_ms[n // 2], 1),
        "p95_ms": round(all_ms[min(int(n * 0.95), n - 1)], 1),
        "p99_ms": round(all_ms[min(int(n * 0.99), n - 1)], 1),
        "min_ms": round(all_ms[0], 1),
        "max_ms": round(all_ms[-1], 1),
        "ttft_mean_ms": ttft_mean,
        "ttft_p95_ms": round(sorted(ttft_ms_list)[int(len(ttft_ms_list) * 0.95)], 1),
        "prefill_ms": ttft_mean,
        "decode_ms_per_tok": decode_ms_per_tok,
        "prefill_decode_ratio": prefill_decode_ratio,
    }


def _measure_throughput(model: Any, tokenizer: Any, device: str, assistant_model: Any = None) -> dict[str, float]:
    """Tokens/sec for output tokens and total tokens."""
    import torch

    MAX_NEW_TOKENS = 512
    ITERS = 2
    out_tps_list: list[float] = []
    total_tps_list: list[float] = []

    gen_kw: dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_length": None,
        "do_sample": False,
    }
    if assistant_model is not None:
        gen_kw["assistant_model"] = assistant_model

    try:
        import nvtx as _nvtx_tput

        def _tput_range(name: str, color: str = "green"):
            return _nvtx_tput.annotate(name, color=color)

    except ImportError:
        import contextlib

        def _tput_range(name: str, color: str = "green"):  # type: ignore[misc]
            return contextlib.nullcontext()

    for prompt in _BENCH_PROMPTS[:3]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        with _tput_range("throughput", color="orange"):
            for _ in range(ITERS):
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kw)
                elapsed = time.perf_counter() - t0
                output_len = out.shape[1] - input_len
                out_tps_list.append(output_len / elapsed)
                total_tps_list.append(out.shape[1] / elapsed)

    return {
        "output_tokens_per_sec": round(statistics.mean(out_tps_list), 1),
        "total_tokens_per_sec": round(statistics.mean(total_tps_list), 1),
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def _measure_batch_throughput(
    model: Any, tokenizer: Any, device: str, batch_sizes: list[int] | None = None
) -> dict[str, Any]:
    """Measure output tokens/sec at multiple batch sizes using left-padded inputs."""
    import torch

    if batch_sizes is None:
        batch_sizes = [1, 4, 8]

    MAX_NEW_TOKENS = 256
    ITERS = 2
    prompt = _BENCH_PROMPTS[0]
    gen_kw: dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_length": None,
        "do_sample": False,
    }

    tokenizer.padding_side = "left"
    results: dict[str, float] = {}

    for bs in batch_sizes:
        prompts = [prompt] * bs
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        input_len = inputs["input_ids"].shape[1]

        tps_list: list[float] = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kw)
            elapsed = time.perf_counter() - t0
            output_tokens = (out.shape[1] - input_len) * bs
            tps_list.append(output_tokens / elapsed)

        results[f"batch{bs}_output_tokens_per_sec"] = round(statistics.mean(tps_list), 1)

    return results


def _measure_perplexity(model: Any, tokenizer: Any, device: str) -> dict[str, float]:
    """Sliding-window perplexity on WikiText-2 test set."""
    import math

    import torch
    from datasets import load_dataset

    STRIDE = 128
    MAX_LEN = 1024  # tokens to evaluate (keep it fast)

    dataset = load_dataset("wikitext", revision=_MODEL_REVISION, name="wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])  # type: ignore[index]
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = min(encodings.input_ids.size(1), MAX_LEN)
    input_ids = encodings.input_ids[:, :seq_len].to(device)

    nlls: list[torch.Tensor] = []
    prev_end = 0
    for begin in range(0, seq_len, STRIDE):
        end = min(begin + tokenizer.model_max_length, seq_len)
        target_len = end - prev_end
        with torch.no_grad():
            out = model(input_ids[:, begin:end], labels=input_ids[:, begin:end])
            # scale NLL to only count the non-overlapping tokens
            nlls.append(out.loss * target_len)
        prev_end = end
        if end == seq_len:
            break

    ppl = math.exp(torch.stack(nlls).sum().item() / seq_len)
    return {
        "perplexity": round(ppl, 3),
        "eval_tokens": seq_len,
        "dataset": "wikitext-2-raw-v1/test",
    }


def _measure_quality(model: Any, tokenizer: Any, device: str) -> dict[str, Any]:
    """MMLU accuracy: zero-shot multiple-choice via log-prob scoring (200 questions)."""
    import torch

    correct = 0
    details: list[dict] = []
    questions = _load_mmlu_questions()
    subject_stats: dict[str, dict[str, int]] = {}

    for entry in questions:
        question, choices, answer_idx = entry["question"], entry["choices"], entry["answer_idx"]
        subject = entry["subject"]
        log_probs: list[float] = []

        for choice in choices:
            prompt = f"Question: {question}\nAnswer: {choice}"
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
            log_probs.append(-out.loss.item())

        predicted_idx = int(max(range(len(log_probs)), key=lambda i: log_probs[i]))
        predicted_letter = chr(ord("A") + predicted_idx)
        answer_letter = chr(ord("A") + answer_idx)
        is_correct = predicted_idx == answer_idx
        if is_correct:
            correct += 1
        s = subject_stats.setdefault(subject, {"correct": 0, "total": 0})
        s["total"] += 1
        if is_correct:
            s["correct"] += 1
        details.append(
            {
                "question": question[:60] + "…" if len(question) > 60 else question,
                "predicted": predicted_letter,
                "expected": answer_letter,
                "correct": is_correct,
                "subject": subject,
            }
        )

    accuracy = correct / len(questions)
    by_subject = {
        subj: {"accuracy": round(v["correct"] / v["total"], 4), "correct": v["correct"], "total": v["total"]}
        for subj, v in subject_stats.items()
    }
    return {
        "mmlu_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(questions),
        "mmlu_by_subject": by_subject,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


# Modal GPU pricing (USD/hr) — used for cost-per-token annotation.
_GPU_COST_PER_HR: dict[str, float] = {
    "T4": 0.59,
    "A10G": 1.10,
    "A100-40GB": 3.70,
    "A100-80GB": 4.00,
    "H100": 6.45,
}


def _gpu_output_path(base: str, gpu: str) -> Path:
    """Return a GPU-specific output path, e.g. results/modal_quant_a10g.json."""
    p = Path(base)
    slug = gpu.lower().replace("-", "_")
    # Only inject GPU slug when the caller used the default name pattern
    if p.stem == "modal_quant_benchmark" or p.stem.startswith("modal_quant_"):
        return p.parent / f"modal_quant_{slug}.json"
    return p


# ---------------------------------------------------------------------------
# Tensor parallelism benchmark (multi-GPU Modal function)
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB:2",
    image=_image,
    timeout=7200,
    volumes={"/model-cache": _model_cache},
    secrets=_modal_secrets,
    memory=65536,
)
def run_tp_benchmark(model_id: str = "") -> dict[str, Any]:
    """Benchmark tensor parallelism (TP=2) via vLLM on two A100-80GB GPUs.

    Splits weight matrices column-wise across both GPUs (vLLM tensor_parallel_size=2).
    Measures latency, throughput, and per-GPU VRAM for comparison against single-GPU
    vllm mode — quantifying the effect of doubled memory bandwidth and capacity.
    """
    import torch
    from vllm import LLM, SamplingParams

    os.environ["TRANSFORMERS_CACHE"] = "/model-cache/hf"
    os.environ["HF_HOME"] = "/model-cache/hf"
    hf_token: str | None = os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

    effective_model = model_id or BASE_MODEL
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    gpu_count = torch.cuda.device_count()
    print(f"[tensor-parallel] TP=2 on {gpu_count}x {gpu_name}  model={effective_model}")

    torch.cuda.reset_peak_memory_stats()
    t_load_start = time.perf_counter()

    llm_kwargs: dict[str, Any] = {
        "model": effective_model,
        "revision": _MODEL_REVISION,
        "dtype": "float16",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 4096,
        "download_dir": "/model-cache/hf",
        "trust_remote_code": False,
    }
    if hf_token:
        llm_kwargs["tokenizer_revision"] = _MODEL_REVISION

    llm = LLM(**llm_kwargs)
    load_time_s = time.perf_counter() - t_load_start
    # total VRAM across all GPUs
    total_vram_mb = sum(torch.cuda.max_memory_allocated(i) for i in range(gpu_count)) / 1024**2
    per_gpu_vram_mb = total_vram_mb / gpu_count
    _model_cache.commit()
    print(
        f"[tensor-parallel] Loaded in {load_time_s:.1f}s  "
        f"{total_vram_mb:.0f} MB total VRAM ({per_gpu_vram_mb:.0f} MB/GPU)"
    )

    # Warmup
    warmup_params = SamplingParams(max_tokens=32, temperature=0.0)
    for _ in range(2):
        llm.generate([_BENCH_PROMPTS[0]], warmup_params, use_tqdm=False)

    # Latency
    lat_params = SamplingParams(max_tokens=256, temperature=0.0)
    latencies_ms: list[float] = []
    ttfts_ms: list[float] = []
    for prompt in _BENCH_PROMPTS * 3:
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], lat_params, use_tqdm=False)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        out = outputs[0]
        if hasattr(out, "metrics") and out.metrics is not None and hasattr(out.metrics, "first_token_time"):
            ttfts_ms.append((out.metrics.first_token_time - out.metrics.first_scheduled_time) * 1000)

    latencies_ms.sort()
    n = len(latencies_ms)

    # Throughput
    thr_params = SamplingParams(max_tokens=512, temperature=0.0)
    thr_times, thr_tokens = [], []
    for _ in range(2):
        t0 = time.perf_counter()
        outs = llm.generate([_BENCH_PROMPTS[0]], thr_params, use_tqdm=False)
        thr_times.append(time.perf_counter() - t0)
        thr_tokens.append(sum(len(o.token_ids) for out in outs for o in out.outputs))
    output_tps = sum(thr_tokens) / sum(thr_times)

    # Batch throughput
    batch_thr: dict[str, float] = {}
    for bs in (1, 4, 8):
        prompts = (_BENCH_PROMPTS * bs)[:bs]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, thr_params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        tokens = sum(len(o.token_ids) for out in outs for o in out.outputs)
        batch_thr[f"batch{bs}_output_tokens_per_sec"] = round(tokens / elapsed, 1)

    quality = _measure_quality_vllm(llm)

    tp_result = {
        "quant_mode": "tensor-parallel",
        "model_id": effective_model,
        "gpu": f"{gpu_count}x {gpu_name}",
        "tensor_parallel_size": 2,
        "gpu_count": gpu_count,
        "load_time_s": round(load_time_s, 2),
        "memory": {
            "total_vram_mb": round(total_vram_mb, 1),
            "per_gpu_vram_mb": round(per_gpu_vram_mb, 1),
        },
        "latency": {
            "max_new_tokens": 256,
            "mean_ms": round(sum(latencies_ms) / n, 1),
            "p50_ms": round(latencies_ms[n // 2], 1),
            "p95_ms": round(latencies_ms[int(n * 0.95)], 1),
            "p99_ms": round(latencies_ms[int(n * 0.99)], 1),
            "ttft_mean_ms": round(sum(ttfts_ms) / len(ttfts_ms), 1) if ttfts_ms else None,
            "prefill_ms": round(sum(ttfts_ms) / len(ttfts_ms), 1) if ttfts_ms else None,
            "decode_ms_per_tok": round(
                max(sum(latencies_ms) / n - sum(ttfts_ms) / len(ttfts_ms), 0) / max(256 - 1, 1), 2
            )
            if ttfts_ms
            else None,
            "prefill_decode_ratio": None,
        },
        "throughput": {
            "output_tokens_per_sec": round(output_tps, 1),
            "max_new_tokens": 512,
        },
        "batch_throughput": batch_thr,
        "perplexity": None,
        "quality": quality,
        "notes": _MODE_NOTES["tensor-parallel"],
    }
    return build_flops_funnel(tp_result, _LLAMA_8B_CONFIG, gpu_name, "tensor-parallel")


# ---------------------------------------------------------------------------
# CPU llama.cpp benchmark (CPU-only Modal function)
# ---------------------------------------------------------------------------


@app.function(
    cpu=8,
    memory=32768,
    image=_cpu_image,
    timeout=3600,
    volumes={"/model-cache": _model_cache},
    secrets=_modal_secrets,
)
def run_cpu_benchmark(model_id: str = "") -> list[dict[str, Any]]:
    """Benchmark GGUF quantization levels via llama.cpp on a CPU-only Modal container.

    Sweeps Q2_K → Q4_K_M → Q5_K_M → Q8_0, returning one result dict per level.
    Uses llama-cpp-python with AVX2/AVX-512 VNNI kernels. MMLU is capped at 20
    questions (CPU runtime constraint). Perplexity and batch throughput are skipped.
    """
    import numpy as np
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    mid = model_id or BASE_MODEL
    base_name = mid.split("/")[-1]
    gguf_repo = os.getenv("GGUF_REPO", _DEFAULT_GGUF_REPO)
    cache_dir = "/model-cache/gguf"

    _ITERS = 3
    _MAX_NEW_TOKENS = 64
    _CPU_MMLU_N = 20  # keep CPU runtime under ~30 min total

    results: list[dict[str, Any]] = []

    for quant_level, mode_name in _GGUF_LEVELS:
        filename = f"{base_name}-{quant_level}.gguf"
        print(f"[{mode_name}] Downloading {gguf_repo}/{filename} …")
        t_load = time.perf_counter()
        gguf_path = hf_hub_download(
            revision=_MODEL_REVISION,
            repo_id=gguf_repo,
            filename=filename,
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        llm = Llama(
            model_path=gguf_path,
            n_ctx=512,
            n_threads=8,
            n_gpu_layers=0,
            verbose=False,
        )
        load_time_s = time.perf_counter() - t_load
        print(f"[{mode_name}] Loaded in {load_time_s:.1f}s")

        # --- Latency ---
        latencies_ms: list[float] = []
        output_tokens_list: list[int] = []
        for prompt in _BENCH_PROMPTS[:_ITERS]:
            t0 = time.perf_counter()
            out = llm.create_completion(prompt, max_tokens=_MAX_NEW_TOKENS, temperature=0.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)
            output_tokens_list.append(out["usage"]["completion_tokens"])

        latencies_ms.sort()
        n = len(latencies_ms)
        total_tps = sum(output_tokens_list) / (sum(latencies_ms) / 1000)

        # --- MMLU quality (log-prob scoring via llama_cpp eval) ---
        cpu_questions = _load_mmlu_questions(n=_CPU_MMLU_N)
        correct = 0
        for entry in cpu_questions:
            question, choices, answer_idx = entry["question"], entry["choices"], entry["answer_idx"]
            choice_str = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
            q_prompt = f"Question: {question}\n{choice_str}\nAnswer:"
            # Feed prompt tokens into the KV cache, read logits for next token
            tokens = llm.tokenize(q_prompt.encode())
            llm.reset()
            llm.eval(tokens)
            logits = np.array(llm.scores[-1])
            # Resolve token IDs for " A", " B", " C", " D"
            letter_ids = [llm.tokenize(f" {chr(65 + i)}".encode(), add_bos=False)[0] for i in range(len(choices))]
            predicted_idx = int(np.argmax([logits[tid] for tid in letter_ids]))
            if predicted_idx == answer_idx:
                correct += 1

        mmlu_acc = round(correct / len(cpu_questions), 4)

        results.append(
            {
                "quant_mode": mode_name,
                "gguf_quant_level": quant_level,
                "model_id": mid,
                "gguf_repo": gguf_repo,
                "gpu": "cpu",
                "load_time_s": round(load_time_s, 2),
                "memory": {"model_weights_mb": None},
                "latency": {
                    "max_new_tokens": _MAX_NEW_TOKENS,
                    "mean_ms": round(sum(latencies_ms) / n, 1),
                    "p50_ms": round(latencies_ms[n // 2], 1),
                    "p95_ms": round(latencies_ms[min(int(n * 0.95), n - 1)], 1),
                    "p99_ms": round(latencies_ms[min(int(n * 0.99), n - 1)], 1),
                    "ttft_mean_ms": None,
                    "prefill_ms": None,
                    "decode_ms_per_tok": None,
                },
                "throughput": {
                    "output_tokens_per_sec": round(total_tps, 1),
                    "max_new_tokens": _MAX_NEW_TOKENS,
                },
                "batch_throughput": None,
                "perplexity": None,
                "quality": {
                    "mmlu_accuracy": mmlu_acc,
                    "correct": correct,
                    "total": len(cpu_questions),
                    "note": f"20-question subset (CPU runtime constraint); {quant_level} GGUF",
                },
                "notes": _MODE_NOTES[mode_name],
            }
        )
        print(f"[{mode_name}] done — tps={total_tps:.1f}  mmlu={mmlu_acc:.0%}")

    _model_cache.commit()
    return results


# ---------------------------------------------------------------------------
# SGLang benchmark
# ---------------------------------------------------------------------------


@app.function(
    gpu=os.environ.get("MODAL_GPU", "A10G"),
    image=_sglang_image,
    timeout=3600,
    volumes={"/model-cache": _model_cache},
    secrets=_modal_secrets,
    memory=32768,
)
def run_sglang_benchmark(model_id: str = "") -> dict[str, Any]:
    """Benchmark via SGLang's offline engine (RadixAttention, chunked prefill).

    Uses sgl.Engine — the synchronous offline API analogous to vLLM's LLM class.
    Measures the same metrics as the vllm mode so results are directly comparable:
    latency, throughput, batch throughput, and zero-shot MMLU accuracy.

    Key difference from vllm: RadixAttention reuses KV cache entries across requests
    that share a common prefix (e.g. system prompt, RAG context). On diverse prompts
    with no shared prefix the throughput should be similar; the gap widens as prefix
    sharing increases.
    """
    import asyncio

    import torch

    os.environ["TRANSFORMERS_CACHE"] = "/model-cache/hf"
    os.environ["HF_HOME"] = "/model-cache/hf"

    import sglang as sgl

    effective_model = model_id or BASE_MODEL
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    print(f"[sglang] Loading {effective_model} …")
    t_load_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    engine = sgl.Engine(
        model_path=effective_model,
        dtype="float16",
        mem_fraction_static=0.85,
        download_dir="/model-cache/hf",
    )
    load_time_s = time.perf_counter() - t_load_start
    model_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    print(f"[sglang] Engine ready in {load_time_s:.1f}s  ({model_vram_mb:.0f} MB VRAM)")

    # Engine init spawns background processes that clear the main thread's event loop.
    # Re-set it here so engine.generate() (which calls asyncio.get_event_loop()) works.
    asyncio.set_event_loop(asyncio.new_event_loop())
    _model_cache.commit()

    def _gen(prompts: list[str], max_tokens: int) -> list[dict]:
        return engine.generate(prompts, sampling_params={"max_new_tokens": max_tokens, "temperature": 0.0})

    def _completion_tokens(out: dict, fallback: int) -> int:
        return (out.get("meta_info") or {}).get("completion_tokens") or fallback

    # Warmup
    _gen([_BENCH_PROMPTS[0]], 32)

    # Latency (single request, 256 output tokens)
    _ITERS = 3
    _MAX_NEW_TOKENS = 256
    latencies_ms: list[float] = []
    for prompt in _BENCH_PROMPTS * _ITERS:
        t0 = time.perf_counter()
        _gen([prompt], _MAX_NEW_TOKENS)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms.sort()
    n = len(latencies_ms)
    lat_mean = sum(latencies_ms) / n

    # Throughput (512 output tokens, 2 iterations)
    thr_times: list[float] = []
    thr_tokens: list[int] = []
    for _ in range(2):
        t0 = time.perf_counter()
        outs = _gen([_BENCH_PROMPTS[0]], 512)
        thr_times.append(time.perf_counter() - t0)
        thr_tokens.append(_completion_tokens(outs[0], 512))
    output_tps = sum(thr_tokens) / sum(thr_times)

    # Batch throughput
    batch_thr: dict[str, float] = {}
    for bs in (1, 4, 8):
        prompts = (_BENCH_PROMPTS * bs)[:bs]
        t0 = time.perf_counter()
        outs = _gen(prompts, 512)
        elapsed = time.perf_counter() - t0
        total_tok = sum(_completion_tokens(o, 512) for o in outs)
        batch_thr[f"batch{bs}_output_tokens_per_sec"] = round(total_tok / elapsed, 1)

    # MMLU: greedy decode, extract first alphabetic character
    mmlu_q = _load_mmlu_questions()
    correct = 0
    for entry in mmlu_q:
        choice_str = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(entry["choices"]))
        prompt = f"Question: {entry['question']}\n{choice_str}\nAnswer:"
        out = _gen([prompt], 4)
        text = (out[0].get("text") or "") if isinstance(out[0], dict) else ""
        predicted = next((c for c in text.strip() if c.isalpha()), "")[:1].upper()
        if predicted == chr(ord("A") + entry["answer_idx"]):
            correct += 1

    result: dict[str, Any] = {
        "quant_mode": "sglang",
        "model_id": effective_model,
        "gpu": gpu_name,
        "load_time_s": round(load_time_s, 2),
        "memory": {
            "model_weights_mb": round(model_vram_mb, 1),
            "reserved_mb": round(reserved_mb, 1),
        },
        "latency": {
            "max_new_tokens": _MAX_NEW_TOKENS,
            "mean_ms": round(lat_mean, 1),
            "p50_ms": round(latencies_ms[n // 2], 1),
            "p95_ms": round(latencies_ms[int(n * 0.95)], 1),
            "p99_ms": round(latencies_ms[int(n * 0.99)], 1),
            "min_ms": round(min(latencies_ms), 1),
            "max_ms": round(max(latencies_ms), 1),
            "ttft_mean_ms": None,
            "ttft_p95_ms": None,
            "prefill_ms": None,
            "decode_ms_per_tok": None,
            "prefill_decode_ratio": None,
        },
        "throughput": {
            "output_tokens_per_sec": round(output_tps, 1),
            "max_new_tokens": 512,
        },
        "batch_throughput": batch_thr,
        "perplexity": None,
        "quality": {
            "mmlu_accuracy": round(correct / len(mmlu_q), 4),
            "correct": correct,
            "total": len(mmlu_q),
        },
        "notes": _MODE_NOTES["sglang"],
    }
    _model_cache.commit()
    return build_flops_funnel(result, _LLAMA_8B_CONFIG, gpu_name, "sglang")


@app.local_entrypoint()
def main(
    output: str = "results/modal_quant_benchmark.json",
    gpu: str = "A10G",
    modes: str = ",".join(_DEFAULT_MODES),
    merge: bool = False,
    model: str = "",
) -> None:
    """Fan out benchmark across all quant modes in parallel, write JSON results.

    Args:
        output: Base path for the JSON results file. The GPU type is
                automatically injected into the filename so each GPU gets its
                own file (e.g. modal_quant_a10g.json, modal_quant_a100_40gb.json).
        gpu:    Modal GPU type (T4, A10G, A100-40GB, A100-80GB, H100).
        modes:  Comma-separated quant modes to run (default: all).
        merge:  If True and the GPU's output file already exists, keep results
                for modes not being re-run. Same-mode results are replaced;
                results from other modes are preserved. Different GPUs always
                write to separate files and never interfere.
        model:  HuggingFace model ID to benchmark (default: unsloth/Meta-Llama-3.1-8B-Instruct).
                Useful for cross-model comparisons, e.g. --model mistralai/Mistral-7B-Instruct-v0.3
                or cross-size runs like --model unsloth/Meta-Llama-3.1-70B-Instruct --gpu H100.
                GPTQ mode uses its own checkpoint unless QUANT_GPTQ_MODEL env var is set.

    Special modes:
        sglang:              Uses the SGLang engine image (separate from vLLM); same GPU as --gpu.
        tensor-parallel:     Always runs on 2x A100-80GB regardless of --gpu flag.
        cpu-q4km/q5km/q8_0: GGUF quant sweep on CPU-only container; --gpu ignored.
        continuous-batching: Runs on the specified --gpu using the async vLLM engine.
    """
    selected = [m.strip() for m in modes.split(",") if m.strip()]
    invalid = [m for m in selected if m not in _ALL_MODES]
    if invalid:
        raise SystemExit(f"Unknown modes: {invalid}. Valid: {list(_ALL_MODES)}")

    out_path = _gpu_output_path(output, gpu)
    gpu_cost_per_hr = _GPU_COST_PER_HR.get(gpu, 1.10)
    effective_model = model or BASE_MODEL

    print(f"Running quantization benchmark on {gpu} for modes: {selected}")
    if gpu != "A10G":
        print(
            f"Note: run_quant_benchmark is decorated for A10G. To use {gpu}, change the gpu= "
            f"parameter in the @app.function decorator and redeploy."
        )
    print(f"Model: {effective_model}")
    print(f"Output → {out_path}  (GPU cost: ${gpu_cost_per_hr}/hr)")
    print("Results will stream in as each mode completes (parallel execution).\n")

    # Partition modes by required compute type
    gpu_modes = [m for m in selected if m not in _MULTI_GPU_MODES and m not in _CPU_MODES and m not in _SGLANG_MODES]
    sglang_modes = [m for m in selected if m in _SGLANG_MODES]
    tp_modes = [m for m in selected if m in _MULTI_GPU_MODES]
    cpu_modes = [m for m in selected if m in _CPU_MODES]

    bench_fn = run_quant_benchmark

    # Load existing results for this GPU if merging
    existing: dict[str, dict] = {}
    if merge and out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
            existing = {r["quant_mode"]: r for r in prior.get("results", [])}
            kept = [m for m in existing if m not in selected]
            print(f"Merging: keeping existing results for {kept}, replacing {selected}\n")
        except Exception as e:
            print(f"Warning: could not read existing results for merge ({e}), starting fresh.\n")

    t_start = time.perf_counter()
    new_results: list[dict] = []

    def _record(result: Any) -> None:
        if isinstance(result, Exception):
            print(f"  [FAILED] {result}")
            return
        new_results.append(result)
        mode = result["quant_mode"]
        ppl_raw = result.get("perplexity")
        ppl_str = f"{ppl_raw['perplexity']:.2f}" if ppl_raw else "n/a"
        lat_info = result.get("latency") or {}
        lat = lat_info.get("mean_ms") or lat_info.get("mean_latency_ms") or 0
        tps = (result.get("throughput") or {}).get("output_tokens_per_sec", 0)
        qual = result.get("quality") or {}
        acc = qual.get("mmlu_accuracy", float("nan"))
        mem_info = result.get("memory") or {}
        mem = mem_info.get("model_weights_mb") or mem_info.get("total_vram_mb") or 0
        acc_str = f"{acc:.0%}" if acc == acc else "n/a"
        print(f"  [{mode:20s}] ppl={ppl_str}  lat={lat:.0f}ms  tps={tps:.0f}  mmlu={acc_str}  vram={mem:.0f}MB")

    # Standard GPU modes (fan-out in parallel)
    if gpu_modes:
        model_ids = [effective_model] * len(gpu_modes)
        for result in bench_fn.map(gpu_modes, model_ids, order_outputs=False, return_exceptions=True):
            _record(result)

    # SGLang modes (separate image; one call per mode)
    for _ in sglang_modes:
        _record(run_sglang_benchmark.remote(model_id=effective_model))

    # Tensor-parallel modes (each needs a dedicated 2xGPU call)
    for _ in tp_modes:
        _record(run_tp_benchmark.remote(model_id=effective_model))

    # CPU/GGUF modes — one call sweeps all four quant levels; filter to what was requested
    if cpu_modes:
        cpu_result = run_cpu_benchmark.remote(model_id=effective_model)
        if isinstance(cpu_result, list):
            requested = set(cpu_modes)
            for r in cpu_result:
                if r.get("quant_mode") in requested:
                    _record(r)
        elif not isinstance(cpu_result, Exception):
            _record(cpu_result)
        else:
            print(f"  [FAILED] cpu benchmark: {cpu_result}")

    total_s = time.perf_counter() - t_start

    # Merge new results over existing, then sort by canonical mode order.
    # Use .get() with fallback so legacy "cpu-llama-cpp" entries in old result files don't crash.
    _mode_rank = {m: i for i, m in enumerate(_ALL_MODES)}
    merged = {**existing, **{r["quant_mode"]: r for r in new_results}}
    all_results = sorted(merged.values(), key=lambda r: _mode_rank.get(r["quant_mode"], 999))
    all_modes = [r["quant_mode"] for r in all_results]

    # Annotate each result with cost per 1k output tokens using actual GPU rate
    # CPU mode uses Modal CPU pricing ($0.000164/vCPU-s x 8 vCPUs ≈ $0.0013/s)
    _CPU_COST_PER_SEC = 0.0013
    gpu_cost_per_sec = gpu_cost_per_hr / 3600
    for r in all_results:
        thr = r.get("throughput") or {}
        tps = thr.get("output_tokens_per_sec", 0)
        if tps and tps > 0:
            cost_per_sec = _CPU_COST_PER_SEC if r["quant_mode"] in _CPU_MODES else gpu_cost_per_sec
            r["cost_per_1k_output_tokens_usd"] = round(cost_per_sec / tps * 1000, 4)

    summary = {
        "benchmark": "modal_quant_benchmark",
        "base_model": effective_model,
        "gpu": gpu,
        "gpu_cost_per_hr_usd": gpu_cost_per_hr,
        "total_wall_time_s": round(total_s, 1),
        "modes_run": all_modes,
        "results": all_results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(all_results)} results → {out_path}  ({total_s:.0f}s total)")
