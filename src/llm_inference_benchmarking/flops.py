"""FLOPs accounting and roofline analysis for LLM inference benchmarks.

Provides:
  - model_flops_per_token: theoretical FLOPs per generated token for a transformer
  - roofline: compares achieved throughput against hardware peak (compute vs memory bound)
  - build_flops_funnel: attaches a flops_funnel sub-dict to a benchmark result

Formula reference:
  Forward pass FLOPs ≈ 6 * N * T  (N = non-embedding params, T = tokens per batch)
  Attention FLOPs per layer ≈ 4 * B * T^2 * H  (B = batch, T = seq_len, H = hidden_dim)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Hardware profiles (peak TFLOPs and memory bandwidth)
# ---------------------------------------------------------------------------

GPU_PROFILES: dict[str, dict[str, float]] = {
    "A10G": {"tflops_fp16": 125.0, "tflops_bf16": 125.0, "memory_bw_gbs": 600.0},
    "A100-40GB": {"tflops_fp16": 312.0, "tflops_bf16": 312.0, "memory_bw_gbs": 1555.0},
    "A100-80GB": {"tflops_fp16": 312.0, "tflops_bf16": 312.0, "memory_bw_gbs": 2000.0},
    "H100": {"tflops_fp16": 989.0, "tflops_bf16": 989.0, "memory_bw_gbs": 3350.0},
    "T4": {"tflops_fp16": 65.0, "tflops_bf16": 65.0, "memory_bw_gbs": 320.0},
    "V100": {"tflops_fp16": 112.0, "tflops_bf16": 112.0, "memory_bw_gbs": 900.0},
}

# Bytes per element for each dtype
DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "fp8": 1.0,
    "nf4": 0.5,
    "nf4-dq": 0.5,
    "int4": 0.5,
    "gptq": 0.5,
    "q8_0": 1.0,
    "q5km": 0.625,
    "q4km": 0.5,
    "q2k": 0.25,
}


def _normalize_gpu_name(raw: str) -> str:
    """Map a torch.cuda.get_device_name() string to a GPU_PROFILES key."""
    for key in GPU_PROFILES:
        if key.lower().replace("-", " ") in raw.lower() or key.lower() in raw.lower():
            return key
    return raw


def model_flops_per_token(
    num_params: int,
    num_layers: int,
    seq_len: int,
    hidden_dim: int,
    batch_size: int = 1,
) -> float:
    """Estimate FLOPs per generated token for a decoder-only transformer.

    Uses the standard approximation:
      - FFN + attention projections: 6 * N (non-embedding params), per token
      - Attention QK^T V computation: 4 * B * T^2 * H per layer, amortised per token
    """
    # Estimate non-embedding params (roughly 75-80% of total for 8B+ models)
    # Use num_params directly as an approximation (embeddings are ~2% of 8B)
    forward_flops = 6 * num_params
    # Attention FLOPs per token (across all layers), amortised over seq_len
    attn_flops_per_layer = 4 * batch_size * seq_len * hidden_dim
    attn_flops_total = attn_flops_per_layer * num_layers / max(seq_len, 1)
    return forward_flops + attn_flops_total


def roofline(
    flops_per_token: float,
    dtype_bytes: float,
    achieved_tps: float,
    gpu: str,
) -> dict[str, float | str]:
    """Return roofline analysis for a given GPU and achieved throughput.

    Returns:
        arithmetic_intensity: FLOPs / byte (operational intensity)
        roofline_bound: theoretical peak TPS (min of compute and memory ceiling)
        achieved_mfu_pct: model FLOPs utilisation as a percentage
        bound: 'compute' or 'memory'
        peak_tps: hardware peak TPS from the binding constraint
    """
    gpu_key = _normalize_gpu_name(gpu)
    profile = GPU_PROFILES.get(gpu_key)
    if profile is None:
        return {
            "arithmetic_intensity": 0.0,
            "roofline_bound": 0.0,
            "achieved_mfu_pct": 0.0,
            "bound": "unknown",
            "peak_tps": 0.0,
            "note": f"GPU '{gpu}' not in profiles; add to GPU_PROFILES in flops.py",
        }

    peak_flops = profile.get("tflops_fp16", profile.get("tflops_bf16", 0.0)) * 1e12
    mem_bw = profile["memory_bw_gbs"] * 1e9

    # Arithmetic intensity: FLOPs per byte of memory traffic
    # Each token requires loading ~num_params * dtype_bytes bytes (weight reads dominate)
    bytes_per_token = flops_per_token / 6 * dtype_bytes  # param bytes ≈ flops/6 * bytes/param
    arithmetic_intensity = flops_per_token / max(bytes_per_token, 1.0)

    # Roofline ceilings
    compute_ceiling_tps = peak_flops / max(flops_per_token, 1.0)
    memory_ceiling_tps = mem_bw / max(bytes_per_token, 1.0)

    peak_tps = min(compute_ceiling_tps, memory_ceiling_tps)
    bound = "compute" if compute_ceiling_tps < memory_ceiling_tps else "memory"
    mfu_pct = round(achieved_tps / max(peak_tps, 1.0) * 100, 2)

    return {
        "arithmetic_intensity": round(arithmetic_intensity, 2),
        "roofline_bound_tps": round(peak_tps, 1),
        "achieved_mfu_pct": mfu_pct,
        "bound": bound,
        "compute_ceiling_tps": round(compute_ceiling_tps, 1),
        "memory_ceiling_tps": round(memory_ceiling_tps, 1),
    }


def build_flops_funnel(
    mode_result: dict,
    model_cfg: dict,
    gpu_name: str,
    quant_mode: str = "",
) -> dict:
    """Compute and attach flops_funnel to mode_result. Returns mode_result unchanged on error."""
    try:
        tput = mode_result.get("throughput") or {}
        achieved_tps = float(tput.get("output_tokens_per_sec", 0))
        if achieved_tps <= 0:
            return mode_result

        # Infer dtype_bytes from quant_mode label
        dtype = quant_mode.lower()
        dtype_bytes = next((v for k, v in DTYPE_BYTES.items() if k in dtype), 2.0)

        flops_per_tok = model_flops_per_token(
            num_params=model_cfg.get("num_params", 8_000_000_000),
            num_layers=model_cfg.get("num_layers", 32),
            seq_len=model_cfg.get("seq_len", 512),
            hidden_dim=model_cfg.get("hidden", 4096),
        )

        rl = roofline(flops_per_tok, dtype_bytes, achieved_tps, gpu_name)
        mode_result["flops_funnel"] = {
            "flops_per_token": round(flops_per_tok / 1e9, 2),  # in GFLOPs
            "achieved_tps": achieved_tps,
            **rl,
        }
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("build_flops_funnel failed: %s", exc)
    return mode_result
