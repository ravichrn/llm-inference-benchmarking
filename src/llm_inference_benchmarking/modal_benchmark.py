"""Modal GPU quantization benchmark for Llama-3.1-8B-Instruct (ungated mirror).

Measures all key metrics across quantization modes (fp16, int8, nf4, nf4-dq, spec-dec,
vllm, gptq, fp8, flash-attn, torch-compile, tensor-parallel, continuous-batching,
cpu-llama-cpp):
  - Latency: mean, p50, p95, p99, time-to-first-token (ms)
  - Throughput: output tokens/sec, total tokens/sec
  - Memory: peak GPU VRAM (MB)
  - Perplexity: on WikiText-2 test set (128-token stride)
  - Quality: zero-shot accuracy on a 50-question MMLU subset

Prerequisites:
  modal setup          # authenticate with your Modal account (one-time)

Optional .env overrides (Modal reads .env automatically via Secret.from_dotenv):
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
from llm_inference_benchmarking.mmlu import _MMLU_SUBJECT_RANGES, _MMLU_SUBSET

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ungated community mirror — identical weights to meta-llama/Llama-3.1-8B-Instruct,
# no HuggingFace token required.
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"


# Pre-quantized GPTQ checkpoint (ungated). Override via QUANT_GPTQ_MODEL in .env.
_DEFAULT_GPTQ_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"


# Draft model for speculative decoding — same tokenizer vocab, ~2 GB VRAM.
DRAFT_MODEL = "unsloth/Llama-3.2-1B-Instruct"

# Model/dataset/GGUF revision for reproducibility.
# Override via env vars; set to a full hex commit SHA to pin to an immutable snapshot.
# Defaults to "main" (latest) for development convenience.
_MODEL_REVISION = os.getenv("MODEL_REVISION", "main")
_DATASET_REVISION = os.getenv("DATASET_REVISION", "main")
_GGUF_REVISION = os.getenv("GGUF_REVISION", "main")

_ALL_MODES = (
    "fp16",
    "int8",
    "nf4",
    "nf4-dq",
    "spec-dec",
    "vllm",
    "kv-fp8",
    "kv-analysis",
    "gptq",
    "gptq-triton",
    "awq",
    "fp8",
    "flash-attn",
    "torch-compile",
    "tensor-parallel",
    "continuous-batching",
    "trtllm",
    "tgi",
    "cpu-q2k",
    "cpu-q4km",
    "cpu-q5km",
    "cpu-q8_0",
    "size-sweep",
)

# Modes that require multiple GPUs (dispatched to run_tp_benchmark instead of run_quant_benchmark)
_MULTI_GPU_MODES = frozenset({"tensor-parallel"})

# KV cache analysis mode — runs a context-length sweep rather than a single config benchmark
_KV_ANALYSIS_MODES = frozenset({"kv-analysis"})

# Modes that require a CPU-only container (dispatched to run_cpu_benchmark)
_CPU_MODES = frozenset({"cpu-q2k", "cpu-q4km", "cpu-q5km", "cpu-q8_0"})

# Modes that require the TGI Docker image (dispatched to run_tgi_benchmark)
_TGI_MODES = frozenset({"tgi"})

# Modes that require the TensorRT-LLM engine build step
_TRTLLM_MODES = frozenset({"trtllm"})

# Size-vs-quant sweep: runs multiple model/quant combinations to compare tradeoffs
# across model sizes at equal VRAM, not just quant formats within one size.
_SIZE_SWEEP_MODES = frozenset({"size-sweep"})

# Each entry: (model_id, quant_mode, label, model_cfg_override)
# Pairs chosen to be roughly VRAM-comparable: 3B-fp16 ≈ 8B-nf4 ≈ 6 GB
_SIZE_SWEEP_CONFIGS: list[dict] = [
    {
        "model_id": "unsloth/Llama-3.2-3B-Instruct",
        "quant_mode": "fp16",
        "size_sweep_label": "3B-fp16",
        "model_cfg": {"num_params": 3_000_000_000, "num_layers": 28, "seq_len": 512, "hidden": 3072},
    },
    {
        "model_id": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "quant_mode": "nf4",
        "size_sweep_label": "8B-nf4",
        "model_cfg": {"num_params": 8_000_000_000, "num_layers": 32, "seq_len": 512, "hidden": 4096},
    },
    {
        "model_id": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "quant_mode": "fp16",
        "size_sweep_label": "8B-fp16",
        "model_cfg": {"num_params": 8_000_000_000, "num_layers": 32, "seq_len": 512, "hidden": 4096},
    },
]

# Default GGUF repo for cpu-llama-cpp modes; override via GGUF_REPO env var
_DEFAULT_GGUF_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"

# Model config used for FLOPs / MFU computation (Llama-3.1-8B dimensions)
_MODEL_CFG_8B: dict[str, int] = {
    "num_params": 8_000_000_000,
    "num_layers": 32,
    "seq_len": 512,
    "hidden": 4096,
}

# GGUF quantization levels and their mode names (ordered cheap→expensive in quality)
_GGUF_LEVELS: list[tuple[str, str]] = [
    ("Q2_K", "cpu-q2k"),
    ("Q4_K_M", "cpu-q4km"),
    ("Q5_K_M", "cpu-q5km"),
    ("Q8_0", "cpu-q8_0"),
]

# Prompts for latency / throughput measurement
_BENCH_PROMPTS = [
    "Summarize why retrieval-augmented generation reduces hallucination in large language models.",
    "Compare diffusion models versus autoregressive models for image generation. Pros and cons.",
    "Rewrite for semantic retrieval: papers about robust RL transfer learning.",
    "Explain the transformer attention mechanism to a software engineer with no ML background.",
    "What are the key trade-offs between model quantization and full-precision inference?",
]


def _question_subject(idx: int) -> str:
    for start, end, subj in _MMLU_SUBJECT_RANGES:
        if start <= idx <= end:
            return subj
    return "other"


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
        "GPTQ (Generative Pre-Trained Transformer Quantization) is a post-training INT4 "
        "quantization method using second-order weight updates (OBQ). Loaded from a "
        "pre-quantized checkpoint via auto-gptq. On A10G, fused CUDA kernels are used when "
        "exllama backend is available; otherwise falls back to triton (slower). Compare "
        "against AWQ — both are INT4 but GPTQ uses row-wise calibration vs AWQ's "
        "activation-aware column scaling."
    ),
    "gptq-triton": (
        "GPTQ with the Triton kernel backend forced (no exllama). Triton kernels are "
        "portable across GPU generations but are typically 10-30% slower than exllama's "
        "hand-tuned CUDA kernels on Ampere. Use this mode to isolate the kernel backend "
        "contribution to throughput when comparing against the default gptq mode."
    ),
    "awq": (
        "AWQ (Activation-aware Weight Quantization) INT4 quantization. Unlike GPTQ, AWQ "
        "identifies salient weight channels by inspecting activation magnitudes during "
        "calibration, then applies per-channel scaling before quantizing. This typically "
        "preserves more accuracy at INT4 than GPTQ. Loaded via autoawq; on A10G uses "
        "fused GEMM kernels. Compare against gptq for the accuracy/throughput trade-off "
        "at the same bit-width."
    ),
    "trtllm": (
        "TensorRT-LLM (TRTLLM) compiles the model into a TensorRT engine with fused "
        "attention, INT8/FP8 quantization, and in-flight batching. Build time is "
        "significant (10-30 min) but inference is among the fastest available on NVIDIA "
        "GPUs. Requires an engine build step per model/GPU combination; cached across "
        "runs. Compare peak throughput and TTFT against vllm and continuous-batching "
        "modes for apples-to-apples production serving cost analysis."
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
    "tgi": (
        "Text Generation Inference (TGI) by HuggingFace — production inference server with "
        "PagedAttention, continuous batching, and token streaming. TGI is the engine behind "
        "HuggingFace Inference Endpoints and is widely deployed in enterprise. This mode "
        "launches TGI inside the Modal container, measures latency, throughput, and TTFT via "
        "the streaming /generate_stream endpoint, and scores MMLU quality via greedy decode. "
        "Perplexity is not available (TGI does not expose per-token log-probs on arbitrary "
        "sequences). Compare against vllm mode to see scheduling and throughput differences."
    ),
    "cpu-q2k": (
        "GGUF Q2_K quantization (≈2.4 bits/weight) via llama.cpp on a CPU-only Modal container. "
        "Smallest file size (~2.9 GB for 8B) and fastest CPU inference, but meaningful quality "
        "degradation. Useful for edge-deployment cost analysis and as a lower bound on quality."
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
    "kv-fp8": (
        "KV cache FP8 quantization via vLLM's kv_cache_dtype='fp8_e4m3'. Unlike weight "
        "quantization (--quantization fp8), kv-fp8 quantizes only the attention key-value "
        "cache tensors stored in GPU VRAM — model weights remain in fp16. This halves KV cache "
        "memory consumption, enabling longer context windows or larger batch sizes without extra "
        "hardware. The VRAM saving grows with context length: at 8k tokens, KV cache accounts "
        "for ~2 GB of VRAM for Llama-3.1-8B; fp8 KV halves this to ~1 GB. Quality impact is "
        "typically <0.5pp MMLU drop. On A10G (Ampere sm_86) fp8 runs in software emulation; "
        "H100 (Hopper sm_90) has native hardware fp8 support and shows additional throughput gains. "
        "Compare KV cache VRAM usage against vllm mode at equivalent context lengths."
    ),
    "kv-analysis": (
        "KV cache context-length sweep: measures KV cache VRAM footprint and latency at "
        "context lengths [512, 1024, 2048, 4096, 8192] for both fp16 and fp8 KV cache. "
        "Shows how KV memory scales linearly with context length, and quantifies the fp8 "
        "memory savings ratio (expected ≈0.5x) and latency overhead across lengths. "
        "This is a diagnostic mode — it runs multiple vLLM engines sequentially and returns "
        "a context_sweep list rather than the standard benchmark result dict."
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
    .pip_install("pynvml>=11.0.0")
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
    # deep-gemm: required by vLLM >=0.7 for FP8 weight-quantised inference on H100 (Hopper sm_90).
    # PyPI tarball omits the CUTLASS submodule, so we clone with --recursive.
    # Must use GCC not clang: clang-14 (Ubuntu 22.04 default) rejects C++20 structured-binding
    # lambda captures used in deep_gemm's SM100 code; GCC 11 (already in image) handles it.
    .run_commands(
        "apt-get install -y --no-install-recommends git"
        " && git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /tmp/deep-gemm"
        " && CC=gcc CXX=g++ pip install /tmp/deep-gemm"
        " && rm -rf /tmp/deep-gemm"
    )
    # Bundle the local package so module-level imports (flops, mmlu) work in the container.
    # Modal uploads modal_benchmark.py as /root/modal_benchmark.py and imports it in every
    # container to find function definitions — without this, top-level
    # `from llm_inference_benchmarking.*` imports raise ModuleNotFoundError.
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

# TGI image: pulls the official HuggingFace Text Generation Inference Docker image
_tgi_image = (
    modal.Image.from_registry(
        "ghcr.io/huggingface/text-generation-inference:2.4.0",
        add_python="3.11",
    )
    # Clear the TGI image's baked-in ENTRYPOINT so Modal can exec Python directly.
    # Without this, Modal's `python -c "..."` becomes `text-generation-launcher python -c "..."`.
    .dockerfile_commands(["ENTRYPOINT []", "CMD []"])
    .pip_install("httpx>=0.27.0", "huggingface_hub>=0.23.0")
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
    gpu="A10G",
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
    if quant_mode == "kv-fp8":
        return _run_vllm_benchmark(gpu_name, hf_token, model_id=model_id, kv_cache_dtype="fp8_e4m3")
    if quant_mode == "kv-analysis":
        return _run_kv_analysis(gpu_name, hf_token, model_id=model_id)
    if quant_mode == "fp8":
        return _run_vllm_benchmark(gpu_name, hf_token, model_id=model_id, quantization="fp8")
    if quant_mode == "continuous-batching":
        return _run_continuous_batching_benchmark(gpu_name, hf_token, model_id=model_id)

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
    if quant_mode == "gptq":
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
        model = GPTQModel.from_quantized(
            model_id,
            cache_dir="/model-cache/hf",
            device="cuda:0",
            **load_kw,
        )
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
            cache_dir="/model-cache/hf",
            device_map="auto",
            dtype=torch.float16,
            revision=_MODEL_REVISION,
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
    # Start power-draw polling before inference; terminate after all inference calls.
    _pwr_proc, _pwr_samples = _poll_power_draw()
    _pwr_t0 = time.perf_counter()

    result["latency"] = _measure_latency(model, tokenizer, device, assistant_model=draft_model)
    result["throughput"] = _measure_throughput(model, tokenizer, device, assistant_model=draft_model)
    # Speculative decoding's assisted_generation does not support batched inputs in transformers
    if quant_mode != "spec-dec":
        result["batch_throughput"] = _measure_batch_throughput(model, tokenizer, device)
    result["perplexity"] = _measure_perplexity(model, tokenizer, device)
    result["quality"] = _measure_quality(model, tokenizer, device)

    _pwr_proc.terminate()
    _pwr_elapsed = time.perf_counter() - _pwr_t0
    _output_tokens = int((result.get("throughput") or {}).get("output_tokens_per_sec", 0) * _pwr_elapsed)
    pwr = _power_metrics(_pwr_samples, _output_tokens, _pwr_elapsed)
    if pwr:
        result["power"] = pwr

    if quant_mode in _MODE_NOTES:
        result["notes"] = _MODE_NOTES[quant_mode]

    return result


# ---------------------------------------------------------------------------
# vLLM benchmark path
# ---------------------------------------------------------------------------


def _run_vllm_benchmark(
    gpu_name: str,
    hf_token: str | None,
    model_id: str = "",
    quantization: str | None = None,
    kv_cache_dtype: str | None = None,
) -> dict[str, Any]:
    """Benchmark the model via vLLM's LLM engine (PagedAttention, continuous batching).

    Args:
        model_id:       HF model to load. Defaults to BASE_MODEL when empty.
        quantization:   Optional vLLM quantization scheme, e.g. "fp8" for dynamic FP8.
                        On A10G (Ampere), fp8 runs in software emulation; H100 uses hardware.
        kv_cache_dtype: Optional KV cache dtype, e.g. "fp8_e4m3". Quantizes only the KV
                        cache tensors (not weights), halving KV cache VRAM at the cost of
                        slight attention precision loss. Most effective at long context lengths.
    """
    import torch
    from vllm import LLM, SamplingParams

    os.environ["TRANSFORMERS_CACHE"] = "/model-cache/hf"
    os.environ["HF_HOME"] = "/model-cache/hf"
    # vLLM >=0.6 defaults to the v1 engine which spawns subprocess workers;
    # those fail in Modal containers due to IPC/shared-memory restrictions
    os.environ["VLLM_USE_V1"] = "0"

    effective_model = model_id or BASE_MODEL
    if kv_cache_dtype:
        quant_mode_label = "kv-fp8"
    elif quantization:
        quant_mode_label = quantization
    else:
        quant_mode_label = "vllm"

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
    if kv_cache_dtype:
        llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
        # FA3 (used on H100) requires BF16 model dtype when KV cache is FP8.
        # FP16 + FP8 KV raises: "For FP8 input, output must have dtype BF16".
        llm_kwargs["dtype"] = "bfloat16"

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
    return {
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
            "decode_ms_per_token": round(max(lat_mean - ttft_mean, 0) / max(256 - 1, 1), 2)
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


def _measure_quality_vllm(llm: Any) -> dict[str, Any]:
    """MMLU log-prob scoring via vLLM's generate with logprobs."""
    from vllm import SamplingParams

    correct = 0
    details: list[dict] = []
    evaluated = _MMLU_SUBSET[:50]
    subject_stats: dict[str, dict[str, int]] = {}

    # Request logprobs for the first generated token — used to score each choice
    # logprobs=20 ensures A/B/C/D tokens are included even when not in model's top-5
    score_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    for q_idx, mmlu_entry in enumerate(evaluated):
        question, choices, answer_letter = mmlu_entry[0], mmlu_entry[1], mmlu_entry[2]
        subject = _question_subject(q_idx)
        choice_labels = [chr(ord("A") + i) for i in range(len(choices))]
        prompt = f"Question: {question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:"
        out = llm.generate([prompt], score_params, use_tqdm=False)[0]
        predicted_label = "A"
        if out.outputs and out.outputs[0].logprobs:
            first_token_logprobs = out.outputs[0].logprobs[0]
            best_lp = -1e9
            for v in first_token_logprobs.values():
                tok = v.decoded_token.strip()
                if tok in choice_labels and v.logprob > best_lp:
                    best_lp = v.logprob
                    predicted_label = tok

        is_correct = predicted_label == answer_letter
        if is_correct:
            correct += 1
        s = subject_stats.setdefault(subject, {"correct": 0, "total": 0})
        s["total"] += 1
        if is_correct:
            s["correct"] += 1
        details.append(
            {
                "question": question[:60] + "…" if len(question) > 60 else question,
                "predicted": predicted_label,
                "expected": answer_letter,
                "correct": is_correct,
                "subject": subject,
            }
        )

    accuracy = correct / len(evaluated)
    by_subject = {
        subj: {"accuracy": round(v["correct"] / v["total"], 4), "correct": v["correct"], "total": v["total"]}
        for subj, v in subject_stats.items()
    }
    return {
        "mmlu_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(evaluated),
        "mmlu_by_subject": by_subject,
        "details": details,
    }


# ---------------------------------------------------------------------------
# KV cache analysis (context-length sweep)
# ---------------------------------------------------------------------------


def _run_kv_analysis(
    gpu_name: str,
    hf_token: str | None,
    model_id: str = "",
    context_lengths: list[int] | None = None,
) -> dict[str, Any]:
    """Sweep context lengths measuring KV cache VRAM for fp16 vs fp8 KV quantization.

    For each context length in [512, 1024, 2048, 4096, 8192]:
      1. Load vLLM with fp16 KV cache, generate a prompt that fills the context.
      2. Record peak VRAM delta (proxy for KV cache size) and latency.
      3. Repeat with kv_cache_dtype='fp8_e4m3'.

    Returns a result dict with a context_sweep list showing the memory savings ratio
    (expected ≈0.5x) and latency overhead at each context length.
    """
    import torch
    from vllm import LLM, SamplingParams

    os.environ["TRANSFORMERS_CACHE"] = "/model-cache/hf"
    os.environ["HF_HOME"] = "/model-cache/hf"
    os.environ["VLLM_USE_V1"] = "0"

    effective_model = model_id or BASE_MODEL
    ctx_lengths = context_lengths or [512, 1024, 2048, 4096, 8192]
    print(f"[kv-analysis] Context-length KV cache sweep for {effective_model}")
    print(f"[kv-analysis] Context lengths: {ctx_lengths}")

    # Build prompts of target lengths using a repeated token pattern
    _FILLER = "The quick brown fox jumps over the lazy dog. " * 200

    sweep_results: list[dict[str, Any]] = []
    for ctx_len in ctx_lengths:
        row: dict[str, Any] = {"context_len": ctx_len}
        # Truncate filler to approximate context length in chars (rough: 4 chars ≈ 1 token)
        prompt = _FILLER[: ctx_len * 4]

        for kv_dtype, label in ((None, "fp16"), ("fp8_e4m3", "fp8")):
            llm_kwargs: dict[str, Any] = {
                "model": effective_model,
                "revision": _MODEL_REVISION,
                "dtype": "float16",
                "gpu_memory_utilization": 0.90,
                "max_model_len": ctx_len,
                "download_dir": "/model-cache/hf",
                "trust_remote_code": False,
            }
            if kv_dtype:
                llm_kwargs["kv_cache_dtype"] = kv_dtype
                # FA3 (H100) requires BF16 model dtype when KV cache is FP8
                llm_kwargs["dtype"] = "bfloat16"

            torch.cuda.reset_peak_memory_stats()
            try:
                llm = LLM(**llm_kwargs)
                vram_after_load = torch.cuda.max_memory_allocated() / 1024**2

                params = SamplingParams(max_tokens=50, temperature=0.0)
                t0 = time.perf_counter()
                llm.generate([prompt], params, use_tqdm=False)
                lat_ms = (time.perf_counter() - t0) * 1000
                vram_peak = torch.cuda.max_memory_allocated() / 1024**2
                kv_vram_mb = max(vram_peak - vram_after_load, 0.0)

                row[f"{label}_kv_vram_mb"] = round(kv_vram_mb, 1)
                row[f"{label}_total_vram_mb"] = round(vram_peak, 1)
                row[f"{label}_lat_ms"] = round(lat_ms, 1)

                print(f"[kv-analysis] ctx={ctx_len:5d}  {label}  kv_vram={kv_vram_mb:.0f}MB  lat={lat_ms:.0f}ms")
                # Free the engine before loading the next config
                del llm
                torch.cuda.empty_cache()
            except Exception as exc:
                print(f"[kv-analysis] ctx={ctx_len} {label} FAILED: {exc}")
                row[f"{label}_kv_vram_mb"] = None
                row[f"{label}_lat_ms"] = None

        # Compute savings ratio
        fp16_vram = row.get("fp16_kv_vram_mb")
        fp8_vram = row.get("fp8_kv_vram_mb")
        if fp16_vram and fp8_vram and fp16_vram > 0:
            row["vram_ratio"] = round(fp8_vram / fp16_vram, 3)
        else:
            row["vram_ratio"] = None

        sweep_results.append(row)

    # Print summary table
    print("\n[kv-analysis] KV Cache Memory vs Context Length (fp16 vs fp8):")
    print(f"  {'ctx_len':>8}  {'fp16 KV MB':>12}  {'fp8 KV MB':>11}  {'ratio':>7}  {'fp16 lat':>10}  {'fp8 lat':>10}")
    print("  " + "─" * 65)
    for row in sweep_results:
        fp16_v = row.get("fp16_kv_vram_mb")
        fp8_v = row.get("fp8_kv_vram_mb")
        ratio = row.get("vram_ratio")
        fp16_l = row.get("fp16_lat_ms")
        fp8_l = row.get("fp8_lat_ms")
        print(
            f"  {row['context_len']:>8}  "
            f"{fp16_v or 'n/a':>12}  {fp8_v or 'n/a':>11}  "
            f"{ratio or 'n/a':>7}  "
            f"{fp16_l or 'n/a':>10}  {fp8_l or 'n/a':>10}"
        )

    _model_cache.commit()
    return {
        "quant_mode": "kv-analysis",
        "model_id": effective_model,
        "gpu": gpu_name,
        "context_sweep": sweep_results,
        "notes": _MODE_NOTES["kv-analysis"],
    }


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

    import torch
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    os.environ["TRANSFORMERS_CACHE"] = "/model-cache/hf"
    os.environ["HF_HOME"] = "/model-cache/hf"
    os.environ["VLLM_USE_V1"] = "0"

    effective_model = model_id or BASE_MODEL
    print(f"[continuous-batching] Loading {effective_model} via AsyncLLMEngine …")

    engine_args = AsyncEngineArgs(
        model=effective_model,
        revision=_MODEL_REVISION,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        download_dir="/model-cache/hf",
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
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

    if quant_mode == "gptq":
        # Pre-quantized INT4 GPTQ checkpoint; auto-gptq registers as a transformers backend.
        # For cross-model runs, caller must set QUANT_GPTQ_MODEL to a matching GPTQ checkpoint.
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


class _PowerSampler:
    """Polls GPU power draw via pynvml in a background thread."""

    def __init__(self) -> None:
        import threading

        import pynvml

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._pynvml = pynvml
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import logging

        while not self._stop.is_set():
            try:
                mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self.samples.append(mw / 1000.0)  # milliwatts → watts
            except Exception as exc:
                logging.getLogger(__name__).debug("pynvml power read failed: %s", exc)
            self._stop.wait(1.0)

    def terminate(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _poll_power_draw() -> tuple[_PowerSampler, list[float]]:
    """Start background GPU power sampling via pynvml.

    Returns (sampler, samples_list). Call sampler.terminate() when done.
    samples_list is populated in-place by a background thread.
    """
    sampler = _PowerSampler()
    return sampler, sampler.samples


def _power_metrics(samples: list[float], output_tokens: int, elapsed_s: float) -> dict[str, float]:
    """Summarise power samples into watt/joule metrics."""
    if not samples:
        return {}
    mean_w = statistics.mean(samples)
    total_j = mean_w * elapsed_s
    tok_per_joule = output_tokens / total_j if total_j > 0 else 0.0
    return {
        "mean_power_w": round(mean_w, 1),
        "total_energy_j": round(total_j, 1),
        "tokens_per_joule": round(tok_per_joule, 2),
    }


def _measure_latency(model: Any, tokenizer: Any, device: str, assistant_model: Any = None) -> dict[str, Any]:
    """Latency over _BENCH_PROMPTS x 5 iterations, plus TTFT."""
    import torch

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
        for _ in range(WARMUP):
            with torch.no_grad():
                model.generate(**inputs, **gen_kw)

        # TTFT: generate exactly 1 token (≈ prefill + one decode step)
        ttft_kw = {**gen_kw, "max_new_tokens": 1}
        ttft_kw.pop("assistant_model", None)  # spec-dec TTFT measured without draft for fair comparison
        for _ in range(3):
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, **ttft_kw)
            ttft_ms_list.append((time.perf_counter() - t0) * 1000)

        # Full-generation latency
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
        "decode_ms_per_token": decode_ms_per_tok,
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

    for prompt in _BENCH_PROMPTS[:3]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

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

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", revision=_DATASET_REVISION)
    text = "\n\n".join(dataset["text"])  # type: ignore[index]
    # Pre-slice text so tokenization stays within model_max_length. ~8 chars/token is conservative;
    # we only need MAX_LEN tokens so this avoids the "longer than max_length" warning without
    # changing which tokens are evaluated (still the first MAX_LEN of the joined corpus).
    encodings = tokenizer(text[: MAX_LEN * 8], return_tensors="pt")
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
    """MMLU subset accuracy: zero-shot multiple-choice via log-prob scoring."""
    import torch

    correct = 0
    details: list[dict] = []
    evaluated = _MMLU_SUBSET[:50]
    subject_stats: dict[str, dict[str, int]] = {}

    for q_idx, mmlu_entry in enumerate(evaluated):
        question, choices, answer_letter = mmlu_entry[0], mmlu_entry[1], mmlu_entry[2]
        subject = _question_subject(q_idx)
        answer_idx = ord(answer_letter) - ord("A")
        log_probs: list[float] = []

        for choice in choices:
            prompt = f"Question: {question}\nAnswer: {choice}"
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
            log_probs.append(-out.loss.item())

        predicted_idx = int(max(range(len(log_probs)), key=lambda i: log_probs[i]))
        predicted_letter = chr(ord("A") + predicted_idx)
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

    accuracy = correct / len(evaluated)
    by_subject = {
        subj: {"accuracy": round(v["correct"] / v["total"], 4), "correct": v["correct"], "total": v["total"]}
        for subj, v in subject_stats.items()
    }
    return {
        "mmlu_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(evaluated),
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
# Per-GPU wrapper functions (Modal 1.x removed with_options; GPU must be set
# at decoration time, so we define one thin wrapper per supported GPU type.
# Each wrapper calls run_quant_benchmark.local() which executes the full
# implementation body inside the container with that GPU attached.)
# ---------------------------------------------------------------------------


@app.function(
    gpu="T4",
    image=_image,
    timeout=7200,
    volumes={"/model-cache": _model_cache},
    secrets=_modal_secrets,
    memory=32768,
)
def _run_quant_t4(quant_mode: str, model_id: str = "") -> dict[str, Any]:
    return run_quant_benchmark.local(quant_mode, model_id=model_id)


@app.function(
    gpu="A100-40GB",
    image=_image,
    timeout=7200,
    volumes={"/model-cache": _model_cache},
    secrets=_modal_secrets,
    memory=32768,
)
def _run_quant_a100_40gb(quant_mode: str, model_id: str = "") -> dict[str, Any]:
    return run_quant_benchmark.local(quant_mode, model_id=model_id)


@app.function(
    gpu="A100-80GB",
    image=_image,
    timeout=7200,
    volumes={"/model-cache": _model_cache},
    secrets=_modal_secrets,
    memory=65536,
)
def _run_quant_a100_80gb(quant_mode: str, model_id: str = "") -> dict[str, Any]:
    return run_quant_benchmark.local(quant_mode, model_id=model_id)


@app.function(
    gpu="H100",
    image=_image,
    timeout=7200,
    volumes={"/model-cache": _model_cache},
    secrets=_modal_secrets,
    memory=65536,
)
def _run_quant_h100(quant_mode: str, model_id: str = "") -> dict[str, Any]:
    return run_quant_benchmark.local(quant_mode, model_id=model_id)


# Maps --gpu flag values to the pre-configured Modal function for that GPU.
_BENCH_FN_FOR_GPU: dict[str, Any] = {
    "A10G": run_quant_benchmark,
    "T4": _run_quant_t4,
    "A100-40GB": _run_quant_a100_40gb,
    "A100-80GB": _run_quant_a100_80gb,
    "H100": _run_quant_h100,
}

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

    return {
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
            "decode_ms_per_token": round(
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
    _MMLU_N = 20  # 20-question subset keeps CPU runtime under ~30 min total

    results: list[dict[str, Any]] = []

    for quant_level, mode_name in _GGUF_LEVELS:
        filename = f"{base_name}-{quant_level}.gguf"
        print(f"[{mode_name}] Downloading {gguf_repo}/{filename} …")
        t_load = time.perf_counter()
        gguf_path = hf_hub_download(
            repo_id=gguf_repo,
            filename=filename,
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN"),
            revision=_GGUF_REVISION,
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
        evaluated = _MMLU_SUBSET[:_MMLU_N]
        correct = 0
        for mmlu_entry in evaluated:
            question, choices, answer_letter = mmlu_entry[0], mmlu_entry[1], mmlu_entry[2]
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
            if chr(65 + predicted_idx) == answer_letter:
                correct += 1

        mmlu_acc = round(correct / len(evaluated), 4)

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
                    "decode_ms_per_token": None,  # nosec B105 — not a password; bandit false positive on key name
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
                    "total": len(evaluated),
                    "note": f"20-question subset (CPU runtime constraint); {quant_level} GGUF",
                },
                "notes": _MODE_NOTES[mode_name],
            }
        )
        print(f"[{mode_name}] done — tps={total_tps:.1f}  mmlu={mmlu_acc:.0%}")

    _model_cache.commit()
    return results


# ---------------------------------------------------------------------------
# TGI (Text Generation Inference) benchmark
# ---------------------------------------------------------------------------


@app.function(
    gpu="A10G",
    image=_tgi_image,
    timeout=1800,
    volumes={"/model-cache": _model_cache},
    secrets=_modal_secrets,
    memory=32768,
)
def run_tgi_benchmark(model_id: str = "") -> dict[str, Any]:
    """Benchmark HuggingFace TGI inference server on a Modal A10G GPU.

    Starts the TGI server as a subprocess, waits for /health, then measures
    latency, throughput, TTFT (via streaming), and MMLU quality (greedy decode).
    Perplexity is not available — TGI does not expose per-token log-probs on
    arbitrary sequences. Batch throughput is measured via concurrent requests.
    """
    import subprocess  # nosec B404 — subprocess is required to launch the TGI server process

    import httpx

    mid = model_id or BASE_MODEL
    port = 8080
    base_url = f"http://localhost:{port}"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    # TGI 2.x reads HUGGINGFACE_HUB_CACHE (not HF_HUB_CACHE) for its local weight cache
    env = {
        **os.environ,
        "HUGGING_FACE_HUB_TOKEN": hf_token,
        "HUGGINGFACE_HUB_CACHE": "/model-cache/tgi",
        "HF_HUB_CACHE": "/model-cache/tgi",
        "HF_HOME": "/model-cache/tgi",
    }

    # Pre-download model weights into the cache so the shard can load offline
    print(f"[tgi] Downloading weights for {mid} …")
    dl_result = subprocess.run(  # nosec B603 B607 — fixed CLI args, no user input interpolated
        ["text-generation-server", "download-weights", mid],
        env=env,
        capture_output=False,
        text=True,
    )
    if dl_result.returncode != 0:
        raise RuntimeError(f"text-generation-server download-weights failed (exit {dl_result.returncode})")

    cmd = [
        "text-generation-launcher",
        "--model-id",
        mid,
        "--port",
        str(port),
        # TGI 2.x flag names (3.x renamed these; we pin image to 2.4.0 for stability)
        "--max-input-length",
        "512",
        "--max-total-tokens",
        "768",
        "--max-batch-prefill-tokens",
        "512",
        "--dtype",
        "float16",
        "--num-shard",
        "1",
        # Avoids flash-attn kernel build failures; falls back to PyTorch SDPA
        "--disable-custom-kernels",
    ]
    print(f"[tgi] cmd: {' '.join(cmd)}")
    print(f"[tgi] Starting TGI server for {mid} …")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  # nosec B603

    # Collect log lines so we can include them in the error message
    import threading

    log_lines: list[str] = []

    def _drain(pipe: Any) -> None:
        for raw in pipe:
            line = raw.decode(errors="replace").rstrip()
            log_lines.append(line)
            print("[tgi-server]", line, flush=True)

    drain_thread = threading.Thread(target=_drain, args=(proc.stdout,), daemon=False)
    drain_thread.start()

    ready = False
    with httpx.Client(timeout=10.0) as hc:
        for _ in range(60):
            time.sleep(5)
            # Fast-fail if TGI process already died
            rc = proc.poll()
            if rc is not None:
                drain_thread.join(timeout=5)
                tail = "\n".join(log_lines[-30:])
                raise RuntimeError(f"TGI process exited with code {rc} before becoming ready.\n{tail}")
            try:
                r = hc.get(f"{base_url}/health")
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:  # nosec B110 — health-check poll; server not ready yet is expected
                pass
    if not ready:
        proc.kill()
        drain_thread.join(timeout=5)
        tail = "\n".join(log_lines[-30:])
        raise RuntimeError(f"TGI server did not become ready within 300 s.\n{tail}")
    print("[tgi] Server ready.")

    _ITERS = 5
    _MAX_NEW_TOKENS = 256
    _BATCH_SIZES = [1, 4, 8]

    def _generate(prompt: str, max_tokens: int = _MAX_NEW_TOKENS) -> tuple[str, int]:
        """POST /generate and return (generated_text, token_count)."""
        with httpx.Client(timeout=120.0) as hc:
            r = hc.post(
                f"{base_url}/generate",
                json={"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "do_sample": False}},
            )
            r.raise_for_status()
            body = r.json()
            text = body.get("generated_text", "")
            token_count = body.get("details", {}).get("generated_tokens", len(text.split()))
            return text, token_count

    def _ttft(prompt: str) -> float:
        """Time-to-first-token via /generate_stream (SSE)."""
        t0 = time.perf_counter()
        with httpx.Client(timeout=120.0) as hc:
            with hc.stream(
                "POST",
                f"{base_url}/generate_stream",
                json={"inputs": prompt, "parameters": {"max_new_tokens": _MAX_NEW_TOKENS, "do_sample": False}},
            ) as resp:
                for _ in resp.iter_lines():
                    return (time.perf_counter() - t0) * 1000
        return (time.perf_counter() - t0) * 1000

    # Warmup
    _generate(_BENCH_PROMPTS[0], max_tokens=32)

    # Latency
    latencies_ms: list[float] = []
    output_tokens_list: list[int] = []
    for prompt in (_BENCH_PROMPTS * _ITERS)[:_ITERS]:
        t0 = time.perf_counter()
        _, ntok = _generate(prompt)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        output_tokens_list.append(ntok)

    latencies_ms.sort()
    n = len(latencies_ms)
    total_tps = sum(output_tokens_list) / (sum(latencies_ms) / 1000)

    # TTFT
    ttft_samples = [_ttft(p) for p in _BENCH_PROMPTS[:3]]
    ttft_mean = round(sum(ttft_samples) / len(ttft_samples), 1)

    # Batch throughput
    import asyncio

    async def _batch_tps(batch_size: int) -> float:
        async with httpx.AsyncClient(timeout=120.0) as hc:
            t0 = time.perf_counter()
            tasks = [
                hc.post(
                    f"{base_url}/generate",
                    json={
                        "inputs": _BENCH_PROMPTS[i % len(_BENCH_PROMPTS)],
                        "parameters": {"max_new_tokens": _MAX_NEW_TOKENS, "do_sample": False},
                    },
                )
                for i in range(batch_size)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.perf_counter() - t0
        total_tok = sum(
            r.json().get("details", {}).get("generated_tokens", 0)
            for r in responses
            if isinstance(r, httpx.Response) and r.status_code == 200
        )
        return round(total_tok / elapsed, 1) if elapsed > 0 else 0.0

    batch_throughput: dict[str, Any] = {}
    for bs in _BATCH_SIZES:
        tps = asyncio.run(_batch_tps(bs))
        batch_throughput[f"batch_{bs}"] = {"output_tokens_per_sec": tps, "batch_size": bs}

    # MMLU quality (greedy decode, 50-question subset)
    evaluated = _MMLU_SUBSET[:50]
    correct = 0
    for mmlu_entry in evaluated:
        question, choices, answer_letter = mmlu_entry[0], mmlu_entry[1], mmlu_entry[2]
        choice_str = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        q_prompt = f"Question: {question}\n{choice_str}\nAnswer:"
        text, _ = _generate(q_prompt, max_tokens=1)
        predicted = text.strip()[:1].upper()
        if predicted == answer_letter:
            correct += 1

    mmlu_acc = round(correct / len(evaluated), 4)

    proc.kill()
    _model_cache.commit()

    return {
        "quant_mode": "tgi",
        "model_id": mid,
        "gpu": "A10G",
        "memory": {"model_weights_mb": None},
        "latency": {
            "max_new_tokens": _MAX_NEW_TOKENS,
            "mean_ms": round(sum(latencies_ms) / n, 1),
            "p50_ms": round(latencies_ms[n // 2], 1),
            "p95_ms": round(latencies_ms[min(int(n * 0.95), n - 1)], 1),
            "p99_ms": round(latencies_ms[min(int(n * 0.99), n - 1)], 1),
            "ttft_mean_ms": ttft_mean,
            "prefill_ms": ttft_mean,
            "decode_ms_per_token": round((sum(latencies_ms) / n - ttft_mean) / max(_MAX_NEW_TOKENS - 1, 1), 2),
        },
        "throughput": {
            "output_tokens_per_sec": round(total_tps, 1),
            "max_new_tokens": _MAX_NEW_TOKENS,
        },
        "batch_throughput": batch_throughput,
        "perplexity": None,
        "quality": {
            "mmlu_accuracy": mmlu_acc,
            "correct": correct,
            "total": len(evaluated),
        },
        "notes": _MODE_NOTES["tgi"],
    }


@app.local_entrypoint()
def main(
    output: str = "results/modal_quant_benchmark.json",
    gpu: str = "A10G",
    modes: str = ",".join(_ALL_MODES),
    merge: bool = True,
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
        tensor-parallel:     Always runs on 2x A100-80GB regardless of --gpu flag.
        tgi:                 Runs the TGI Docker image on A10G; --gpu is ignored.
        cpu-q2k/q4km/q5km/q8_0: GGUF quant sweep on CPU-only container; --gpu ignored.
        continuous-batching: Runs on the specified --gpu using the async vLLM engine.
        kv-fp8:              vLLM with fp8 KV cache quantization (weights stay fp16).
        kv-analysis:         Context-length sweep measuring KV cache VRAM (fp16 vs fp8).
        size-sweep:          Runs _SIZE_SWEEP_CONFIGS in parallel — compares model sizes
                             at roughly equal VRAM (3B-fp16, 8B-nf4, 8B-fp16). Reports
                             tps, MMLU, VRAM, and tokens/joule side-by-side. Results are
                             stored in the same JSON under quant_mode "size-sweep/<label>".
    """
    selected = [m.strip() for m in modes.split(",") if m.strip()]
    invalid = [m for m in selected if m not in _ALL_MODES]
    if invalid:
        raise SystemExit(f"Unknown modes: {invalid}. Valid: {list(_ALL_MODES)}")

    out_path = _gpu_output_path(output, gpu)
    gpu_cost_per_hr = _GPU_COST_PER_HR.get(gpu, 1.10)
    effective_model = model or BASE_MODEL

    print(f"Running quantization benchmark on {gpu} for modes: {selected}")
    print(f"Model: {effective_model}")
    print(f"Output → {out_path}  (GPU cost: ${gpu_cost_per_hr}/hr)")
    print("Results will stream in as each mode completes (parallel execution).\n")

    # Partition modes by required compute type
    gpu_modes = [
        m
        for m in selected
        if m not in _MULTI_GPU_MODES and m not in _CPU_MODES and m not in _TGI_MODES and m not in _SIZE_SWEEP_MODES
    ]
    tp_modes = [m for m in selected if m in _MULTI_GPU_MODES]
    cpu_modes = [m for m in selected if m in _CPU_MODES]
    tgi_modes = [m for m in selected if m in _TGI_MODES]
    size_sweep = "size-sweep" in selected

    if gpu not in _BENCH_FN_FOR_GPU:
        raise SystemExit(f"Unsupported --gpu {gpu!r}. Valid: {list(_BENCH_FN_FOR_GPU)}")
    bench_fn = _BENCH_FN_FOR_GPU[gpu]

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
        # Attach MFU / roofline analysis if not already present.
        if "flops_funnel" not in result:
            build_flops_funnel(
                result,
                model_cfg=_MODEL_CFG_8B,
                gpu_name=result.get("gpu", gpu),
                quant_mode=result.get("quant_mode", ""),
            )
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
        ff = result.get("flops_funnel") or {}
        mfu = ff.get("achieved_mfu_pct")
        mfu_str = f"{mfu:.1f}%" if mfu is not None else "n/a"
        print(
            f"  [{mode:20s}] ppl={ppl_str}  lat={lat:.0f}ms  tps={tps:.0f}"
            f"  mmlu={acc_str}  vram={mem:.0f}MB  mfu={mfu_str}"
        )

    # Standard GPU modes (fan-out in parallel)
    if gpu_modes:
        model_ids = [effective_model] * len(gpu_modes)
        for result in bench_fn.map(gpu_modes, model_ids, order_outputs=False, return_exceptions=True):
            _record(result)

    # Tensor-parallel modes (each needs a dedicated 2xGPU call)
    for _ in tp_modes:
        _record(run_tp_benchmark.remote(model_id=effective_model))

    # TGI mode (single GPU, dedicated image)
    if tgi_modes:
        _record(run_tgi_benchmark.remote(model_id=effective_model))

    # Size-vs-quant sweep — runs _SIZE_SWEEP_CONFIGS in parallel; each uses its own model/quant pair
    if size_sweep:
        print("\nRunning size-vs-quant sweep …")
        sweep_modes = [cfg["quant_mode"] for cfg in _SIZE_SWEEP_CONFIGS]
        sweep_models = [cfg["model_id"] for cfg in _SIZE_SWEEP_CONFIGS]
        for result, cfg in zip(
            bench_fn.map(sweep_modes, sweep_models, order_outputs=True, return_exceptions=True),
            _SIZE_SWEEP_CONFIGS,
            strict=False,
        ):
            if isinstance(result, Exception):
                print(f"  [FAILED size-sweep {cfg['size_sweep_label']}] {result}")
                continue
            # Tag result with the sweep label so it's distinguishable in the JSON
            result["size_sweep_label"] = cfg["size_sweep_label"]
            result["size_sweep"] = True
            # Override quant_mode to a unique key so it doesn't collide with standalone modes
            result["quant_mode"] = f"size-sweep/{cfg['size_sweep_label']}"
            model_cfg = cfg.get("model_cfg", _MODEL_CFG_8B)
            build_flops_funnel(
                result, model_cfg=model_cfg, gpu_name=result.get("gpu", gpu), quant_mode=cfg["quant_mode"]
            )
            new_results.append(result)
            label = cfg["size_sweep_label"]
            tps = (result.get("throughput") or {}).get("output_tokens_per_sec", 0)
            acc = (result.get("quality") or {}).get("mmlu_accuracy", float("nan"))
            mem = (result.get("memory") or {}).get("model_weights_mb", 0)
            pwr = (result.get("power") or {}).get("tokens_per_joule", float("nan"))
            print(f"  [size-sweep {label:12s}] tps={tps:.0f}  mmlu={acc:.0%}  vram={mem:.0f}MB  tok/J={pwr:.1f}")

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

    # Print MFU summary table sorted by utilisation descending
    mfu_rows = [
        (r["quant_mode"], (r.get("flops_funnel") or {}), (r.get("throughput") or {}))
        for r in new_results
        if (r.get("flops_funnel") or {}).get("achieved_mfu_pct") is not None
    ]
    if mfu_rows:
        print(f"\n{'─' * 72}")
        print(f"  {'Mode':<22} {'TPS':>7}  {'MFU%':>6}  {'Bound':>8}  {'Roofline TPS':>14}")
        print(f"  {'─' * 22} {'─' * 7}  {'─' * 6}  {'─' * 8}  {'─' * 14}")
        for mode, ff, thr in sorted(mfu_rows, key=lambda x: -(x[1].get("achieved_mfu_pct") or 0)):
            tps = thr.get("output_tokens_per_sec", 0)
            mfu = ff.get("achieved_mfu_pct", 0)
            bound = ff.get("bound", "n/a")
            roof = ff.get("roofline_bound_tps", 0)
            print(f"  [{mode:<20s}]  {tps:>7.0f}  {mfu:>5.1f}%  {bound:>8}  {roof:>14.0f}")
        print(f"{'─' * 72}\n")

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
