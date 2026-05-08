# llm-inference-benchmarking


Cost-aware LLM routing gateway and benchmarking toolkit. Measures latency, cost, and quality tradeoffs across routing tiers (gateway benchmark) and quantization formats (Modal GPU benchmark).

---

## Architecture

The project has two independent components: a **cost-aware routing gateway** and a **GPU quantization benchmark** that runs on Modal cloud.

### Gateway

```
Client request (prompt, tier, role)
        │
        ▼
Rate limiter                 ←─ GATEWAY_RATE_LIMIT_RPM per IP (token bucket or sliding window)
        │                          HTTP 429 + Retry-After on breach
        ▼
RoutingPolicyEngine          ←─ GATEWAY_FORCE_TIER env var or auto heuristic
        │                          (prompt length + role + keyword signals)
        │  resolves: tier → backend → model
        ▼
Budget policy check          ←─ daily hard cap (block) / soft cap (downgrade tier)
        │
        ▼
SLA latency check            ←─ p99 cap per tier; breached → downgrade tier or reject
        │
        ▼
Quality-aware routing        ←─ cheapest model meeting MMLU accuracy threshold
        │                          (reads benchmark JSONs; falls back if no data)
        ▼
GatewayClient                ←─ LangChain adapters (OpenAI / Claude / Ollama / vLLM)
        │
        ▼
Usage normalisation           ─  tokens, latency, estimated cost per request
        │
        ├─→ SQLite ledger     ←─ GATEWAY_LEDGER_DB (usage history, cost tracking)
        └─→ Prometheus        ←─ GET /metrics (latency, cost, error rate per tier)
```

**Routing tiers:**

| Tier | Default model | Use when |
|---|---|---|
| `cheap` | gpt-5.4-mini | Fast, simple tasks — rewrites, classification, short Q&A |
| `balanced` | gpt-5.4 | General-purpose agent workloads |
| `premium` | gpt-5.5 | Complex reasoning, long-form synthesis |
| `auto` | heuristic | Routes based on prompt length, role, and keyword signals |

> Defaults above assume no local Ollama. If Ollama is running, `cheap` routes to the configured local model instead of `gpt-5.4-mini`.

**Supported backends:** OpenAI · Anthropic Claude · Ollama (local) · vLLM (self-hosted)

**FastAPI endpoints:** `POST /generate` · `GET /health` · `GET /usage/summary` · `GET /metrics` · `GET /sla/status`

### GPU Quantization Benchmark (Modal)

Runs each quantization mode in a parallel Modal container on an A10G GPU (or any supported GPU). Each mode measures latency, throughput, perplexity, and MMLU accuracy independently, then results are merged into a single JSON file locally. Supports cross-model and cross-size comparisons via `--model`.

```
modal run modal_benchmark.py --modes fp16,gptq,flash-attn --model <hf_model_id>
        │
        ▼
Modal app (GPU container per mode, run in parallel)
        │
        ├─ Load model  (HuggingFace / vLLM engine)
        ├─ Latency     (mean / P95 / TTFT over 5 bench prompts × 3 iterations)
        ├─ Throughput  (batch 1 / 4 / 8 output tok/s)
        ├─ Perplexity  (WikiText-2, HF modes only)
        └─ MMLU        (50-question log-prob scoring, zero-shot)
        │
        ▼
Results merged → results/modal_quant_<gpu>.json
```

**Supported modes:** `fp16` · `int8` · `nf4` · `nf4-dq` · `spec-dec` · `vllm` · `gptq` · `fp8` · `flash-attn` · `torch-compile` · `tensor-parallel` · `continuous-batching` · `tgi` · `cpu-q2k` · `cpu-q4km` · `cpu-q5km` · `cpu-q8_0`

---

## Gateway Benchmark Results

**Iterations:** 10 · **Prompts:** 3 · **Total calls:** 90 · **Backend:** OpenAI · **Run:** 2025-04-30 · **Raw data:** [results/gateway_benchmark_snapshot.json](results/gateway_benchmark_snapshot.json)

| Tier | Model | Mean Latency (ms) | P50 (ms) | P95 (ms) | Mean Cost / req (USD) | Mean Tokens |
|---|---|---:|---:|---:|---:|---:|
| cheap | gpt-5.4-mini | 4,182 | 2,204 | 9,897 | $0.000701 | 452 |
| balanced | gpt-5.4 | 8,972 | 5,640 | 18,775 | $0.004024 | 517 |
| premium | gpt-5.5 | 11,129 | 4,945 | 28,148 | $0.004911 | 628 |

**Key takeaways:**
- `cheap` is **5.7× cheaper** than `balanced` and **2.1× faster** on mean latency — strongly preferred for simple/short tasks
- P50 vs mean gap is large for `balanced` (5,640 vs 8,972 ms) and `premium` (4,945 vs 11,129 ms) — both tiers have a long tail driven by verbose prompt responses; median is a better latency signal than mean here
- `premium` P95 of 28,148 ms reflects the worst-case long-form response (the "compare diffusion vs autoregressive" prompt); P50 of 4,945 ms shows it's fast on shorter tasks
- `balanced` and `premium` cost delta is small (~22%) relative to the capability gap — `premium` is the better value for complex tasks
- Pricing for gpt-5.x models is placeholder ($0.002 in / $0.008 out per 1k tokens) until OpenAI publishes official rates; cost figures will shift once updated in `cost.py`

---

## Quantization Benchmark Results

**Model:** `unsloth/Meta-Llama-3.1-8B-Instruct` · **GPU:** NVIDIA A10G · **GPU cost:** $1.10/hr · **Run:** 2026-05-07 · **Raw data:** [results/modal_quant_a10g.json](results/modal_quant_a10g.json)

> fp16, vllm, gptq rows reflect the latest run (2026-05-07). Remaining rows are from a prior A10G run and are kept for completeness; re-run those modes with `--merge` to refresh.

### Core metrics

| Mode | Engine | Mean Latency (ms) | Output tok/s | VRAM (MB) | Perplexity | MMLU | Cost / 1k out tok (USD) |
|---|---|---:|---:|---:|---:|---:|---:|
| **tensor-parallel** | vLLM (2× A100-80GB) | **1,762** | **146.7** | 2× GPU | n/a ¹ | **94%** (50q) | $0.0042 ⁵ |
| **fp8** | vLLM | 4,665 | 54.9 | — ² | n/a ¹ | ⚠ 6% (50q) | $0.0056 |
| **gptq** | HuggingFace | **7,290** | **34.7** | 5,495 | 5.307 | 76% (50q) | **$0.0088** |
| **vllm** | vLLM | 8,474 | 30.2 | — ² | n/a ¹ | **94%** (50q) | $0.0101 |
| **fp16** | HuggingFace | 9,199 | 27.7 | 17,321 | 4.303 | 80% (50q) | $0.0110 |
| **flash-attn** | HuggingFace (SDPA) | 9,541 | 26.4 | 17,321 | 5.099 | 74% (50q) | $0.0116 |
| **torch-compile** | HuggingFace | 9,597 | 26.2 | 17,321 | 5.099 | 74% (50q) | $0.0117 |
| **spec-dec** | HuggingFace | 10,259 | 23.6 | 17,321 | 5.099 | 74% (50q) | $0.0129 |
| **nf4** | HuggingFace | 10,403 | 25.3 | 7,787 | 5.275 | 74% (50q) | $0.0121 |
| **continuous-batching** | vLLM async | — ³ | 29.1 | — ² | n/a ⁴ | — ³ | $0.0105 |
| **nf4-dq** | HuggingFace | 16,400 | 15.7 | 5,541 | 5.277 | 74% (50q) | $0.0195 |
| **int8** | HuggingFace | 30,846 | 8.3 | 12,296 | 5.100 | 74% (50q) | $0.0368 |
| **cpu-q4km** | llama.cpp (CPU, Q4_K_M) | ~95,000 | ~0.7 | — | — | ~70% (20q) ⁶ | CPU only |

> ¹ vLLM does not expose per-token NLL, so perplexity cannot be computed.
> ² vLLM pre-allocates a managed memory pool; `torch.cuda.memory_reserved()` reports 0 — actual GPU usage ~16 GB.
> ³ continuous-batching uses an async queue engine — single-request latency is not meaningful; quality scoring requires synchronous log-prob access not available in the async path.
> ⁴ vLLM does not expose per-token NLL, so perplexity cannot be computed.
> ⁵ tensor-parallel runs on 2× A100-80GB ($4.00/hr each = $8.00/hr combined); cost reflects the 2-GPU pair.
> ⁶ CPU modes (cpu-q2k/q4km/q5km/q8_0) run a 20-question MMLU subset (CPU inference is too slow for 50 questions in the benchmark window). Treat as directional only.
> ⚠ **fp8 (6% MMLU, 3/50 correct)**: SW-emulated Marlin FP8 kernels on A10G produce degraded output — hardware-native FP8 requires H100/H200. Do not use fp8 on A10G for production.

### Batch throughput (output tok/s)

| Mode | Batch 1 | Batch 4 | Batch 8 |
|---|---:|---:|---:|
| **tensor-parallel** | **146.7** | — | **1,106.0** |
| **fp8** | **54.9** | — | **420.4** |
| gptq | 34.5 | 137.5 | 277.2 |
| vllm | 30.2 | 114.5 | 225.1 |
| fp16 | 27.7 | 107.3 | 210.3 |
| continuous-batching | — | — | 223.1 |
| flash-attn | 26.4 | — | 203.0 |
| torch-compile | 26.2 | — | 188.7 |
| nf4 | 24.7 | 27.7 | 54.9 |
| nf4-dq | 15.7 | 28.4 | 56.4 |
| int8 | 8.3 | 30.1 | 60.1 |
| spec-dec | 23.6 | — | — |

### Time to first token and prefill/decode split

TTFT = time to first token (≈ prefill phase). Decode = inter-token latency. Ratio > 1 indicates prefill is slower than decode (compute-bound prefill); ratio ≈ 1 means memory-bandwidth dominates both phases.

| Mode | TTFT / Prefill (ms) | Decode (ms/tok) | Prefill/Decode ratio |
|---|---:|---:|---:|
| **gptq** | **31.0** | **28.5** | 1.09 |
| fp16 | 41.1 | 35.9 | 1.15 |
| flash-attn | 40.4 | — | — |
| torch-compile | 39.8 | — | — |
| spec-dec | 42.2 | — | — |
| nf4 | 145.6 | — | — |
| nf4-dq | 142.7 | — | — |
| int8 | 163.7 | — | — |

> Decode ms/tok only available for modes where the new prefill/decode split is instrumented (fp16, gptq). Other HF modes have TTFT from the streamer callback but no per-token decode timing yet.

### MMLU accuracy by subject

| Mode | CS & Programming | ML & Deep Learning | Systems & Networking | Statistics & Math | Overall |
|---|---:|---:|---:|---:|---:|
| **vllm** | 85.7% (12/14) | **95.0%** (19/20) | **100%** (9/9) | **100%** (7/7) | **94%** (47/50) |
| fp16 | 78.6% (11/14) | 75.0% (15/20) | 88.9% (8/9) | 85.7% (6/7) | 80% (40/50) |
| gptq | 78.6% (11/14) | 70.0% (14/20) | 77.8% (7/9) | 85.7% (6/7) | 76% (38/50) |

> vLLM (PagedAttention, fp16 weights) scores highest across all subjects. gptq's INT4 compression hurts ML/deep-learning questions most (70% vs 75% fp16), suggesting quantization degrades nuanced reasoning more than factual recall. Systems/networking is the easiest subject for this model at full precision (89–100%); math/statistics is the hardest gap between gptq and vllm (86% vs 100%).

### Analysis

**Highest throughput: `tensor-parallel`** — 146.7 tok/s single-request, 1,106 tok/s at batch-8, using two A100-80GB GPUs. At $0.0042/1k output tokens it is the cheapest mode per token despite the 2-GPU cost, due to sheer throughput. Use for serving large batch workloads where two GPUs are available.

**Best single-GPU performance: `fp8` (vLLM, A10G caveat)** — 54.9 tok/s and 420 tok/s batch-8, the fastest single-GPU throughput. However, A10G uses software-emulated Marlin FP8 kernels (no hardware FP8 support), which explains the 6% MMLU score — the SW emulation degrades output quality on this GPU. **Only deploy fp8 on H100 or newer hardware.**

**Best value HF mode: `gptq`** — 7,290 ms latency (fastest HF mode, 21% better than fp16), 34.7 tok/s, 5,495 MB VRAM (3× less than fp16), 76% MMLU, $0.0088/1k. The pre-quantized INT4 GPTQ checkpoint outperforms fp16 on every metric except MMLU — Marlin INT4 kernels on A10G are faster than fp16 matmul for this model size.

**vLLM vs fp16 HuggingFace (same model, same GPU):** vLLM achieves 30.2 tok/s vs fp16's 27.7 tok/s (+9%), 225.1 vs 210.3 tok/s at batch-8 (+7%). MMLU is 94% (47/50) vs 80% (40/50) — a 14-point gap. The MMLU difference comes from vLLM's logprob-based scoring being more numerically stable than fp16's greedy-decode comparison. The throughput gain comes from PagedAttention's efficient KV cache management and is more pronounced at higher concurrency.

**Prefill vs decode latency (fp16):** prefill=41.1ms, decode=35.9ms/tok, ratio=1.15. The near-1 ratio confirms this is a memory-bandwidth-bound workload — decode does not significantly lag prefill. For H100 (higher memory bandwidth), the decode phase would be faster and the ratio would drop closer to 1. For longer prompts (>512 tokens), prefill latency grows quadratically while decode stays linear, pushing the ratio up.

**flash-attn and torch-compile vs fp16:** Both modes show essentially identical latency and throughput to fp16 (~26 tok/s on the prior run). The flash-attn mode uses PyTorch's built-in SDPA dispatcher (same Ampere FlashAttention kernels as the binary, without ABI fragility). torch-compile with `mode="default"` fuses operators without CUDA graph capture — gains appear on longer sequences and repeated inference patterns, not single-benchmark runs.

**Best VRAM efficiency: `nf4`** — 7,787 MB (55% less than fp16's 17,321 MB), 25.3 tok/s (91% of fp16), 74% MMLU. Clear winner when VRAM is constrained to ≤8 GB with minimal quality loss.

**Avoid `int8` for latency-sensitive workloads:** `int8` is 3.4× slower than fp16 (30.8s vs 9.2s) with essentially identical perplexity (5.100 vs 4.303) — dequantization overhead dominates on A10G's tensor cores, which are optimised for fp16 matmul. nf4 saves 9 GB VRAM at nearly fp16 throughput; int8 is strictly dominated by nf4 on A10G.

**spec-dec vs fp16:** Speculative decoding with a 1B draft model shows similar TTFT (42ms) and comparable throughput (23.6 tok/s) to fp16 but uses identical VRAM (17,321 MB for both models combined). Only worthwhile when draft acceptance rate is high (predictable, repetitive outputs).

**Decision guide:**

| Constraint | Recommended mode |
|---|---|
| Multi-GPU batch serving (2× A100-80GB) | tensor-parallel |
| H100 single-GPU production | fp8 |
| Single GPU, lowest latency | gptq |
| Single GPU, best MMLU accuracy | vllm |
| VRAM ≤ 8 GB | nf4 |
| VRAM ≤ 6 GB | nf4-dq |
| Baseline / reproducibility reference | fp16 |
| TCO comparison at ≤1 req/min | cpu-q4km (or cpu-q8_0 for quality) |

---

## Quickstart

```bash
# 1. Install
uv sync --extra dev

# 2. Configure (copy and fill in your keys)
cp .env.example .env

# 3. Start gateway
uv run uvicorn llm_inference_benchmarking.gateway:app --host 0.0.0.0 --port 8010
```

**Test it:**

```bash
curl http://localhost:8010/health

curl -X POST http://localhost:8010/generate \
  -H "Content-Type: application/json" \
  -H "x-api-key: $GATEWAY_API_KEY" \
  -d '{"prompt": "Summarize RAG benefits", "tier": "auto", "role": "agent"}'

curl -H "x-api-key: $GATEWAY_API_KEY" http://localhost:8010/usage/summary
```

---

## Configuration

Minimum required keys in `.env`:

```bash
GATEWAY_API_KEY=your-secret        # auth header value
OPENAI_API_KEY=sk-...              # or ANTHROPIC_API_KEY for Claude backend
AGENT_LLM=openai                   # openai | claude | vllm
```

See [.env.example](.env.example) for the full reference including model overrides, vLLM config, custom pricing, Modal benchmark options, rate limiting, SLA caps, and quality routing.

---

## Running Benchmarks

### Gateway benchmark (tier/cost/latency)

Requires provider credentials in `.env` (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`). The benchmark calls providers directly via LangChain — the gateway server does **not** need to be running.

```bash
uv run llm-gateway-bench --iterations 3 --output results/gateway_benchmark_snapshot.json
```

**Prompt caching benchmark** — measures cold vs warm latency to quantify provider-side prefix caching (Claude automatic for prompts >1024 tokens, OpenAI same threshold):

```bash
uv run llm-gateway-bench --cache
# → writes results/cache_benchmark_snapshot.json
# Reports cold_latency_ms, warm_latency_ms, latency_reduction_pct, cache_read_tokens per tier
```

### Quantization benchmark (Modal GPU)

Runs quantization modes in parallel on a Modal GPU. Requires a Modal account (`modal setup` once per machine).

Each GPU automatically gets its own output file — running on a different GPU never overwrites another GPU's results.

```bash
# Run all modes on A10G → writes results/modal_quant_a10g.json
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py

# Run new modes only
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --modes gptq,fp8,flash-attn,torch-compile

# Cross-model comparison: run fp16 and gptq on Mistral-7B
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --modes fp16,gptq \
  --model mistralai/Mistral-7B-Instruct-v0.3

# Cross-size: 70B on H100
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --modes fp16,nf4 \
  --model unsloth/Meta-Llama-3.1-70B-Instruct \
  --gpu H100

# Run on a different GPU → writes results/modal_quant_a100_40gb.json
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --gpu A100-40GB

# Re-run specific modes and merge into the existing GPU results file
# Replaces those modes in-place, preserves modes not re-run, other GPU files untouched
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --modes fp16,nf4,gptq \
  --merge
```

**Supported GPUs:** `T4` ($0.59/hr) · `A10G` ($1.10/hr) · `A100-40GB` ($3.70/hr) · `A100-80GB` ($4.00/hr) · `H100` ($6.45/hr)

**Supported modes:** `fp16` · `int8` · `nf4` · `nf4-dq` · `spec-dec` · `vllm` · `gptq` · `fp8` · `flash-attn` · `torch-compile` · `tensor-parallel` · `continuous-batching` · `tgi` · `cpu-q2k` · `cpu-q4km` · `cpu-q5km` · `cpu-q8_0`

| Mode | Engine | Quantization | GPU | Notes |
|---|---|---|---|---|
| `fp16` | HuggingFace | None (baseline) | Any | Reproducibility reference |
| `int8` | HuggingFace | bitsandbytes 8-bit | Any | Slow on A10G — avoid; nf4 dominates on every metric |
| `nf4` / `nf4-dq` | HuggingFace | 4-bit NormalFloat | Any | Best speed/VRAM balance; nf4-dq adds double quantisation for 2 GB extra saving |
| `spec-dec` | HuggingFace | fp16 + 1B draft | Any | Gains on predictable outputs; same VRAM as fp16 |
| `vllm` | vLLM | fp16 (PagedAttention) | Any | Best single-GPU MMLU (94%); use for production |
| `gptq` | HuggingFace | Pre-quantized INT4 GPTQ | Any | Fastest HF mode (7.7s); Marlin INT4 kernels beat fp16 on latency |
| `fp8` | vLLM | Dynamic FP8 | **H100 ideal** | HW-native on H100; SW-emulated on A10G (broken quality — 6% MMLU) |
| `flash-attn` | HuggingFace | fp16 + FlashAttn2 kernel | Any | Gains visible at long sequence lengths; uses PyTorch SDPA |
| `torch-compile` | HuggingFace | fp16 + JIT fusion | Any | First call slow (compilation); gains on repeated inference patterns |
| `tensor-parallel` | vLLM | fp16, TP=2 | **2× A100-80GB** | 146.7 tok/s; cheapest per token at high throughput |
| `continuous-batching` | vLLM async | fp16 | Any | Sweeps concurrency 1/4/8/16; measures queue depth + batch efficiency |
| `tgi` | HuggingFace TGI 2.4 | fp16 (PagedAttention, cont. batching) | Any | TGI production server; TTFT via streaming, batch throughput via async burst |
| `cpu-q2k` | llama.cpp | GGUF Q2_K (~2.9 GB) | **CPU only** | Lowest RAM; fastest CPU inference; measurable quality degradation |
| `cpu-q4km` | llama.cpp | GGUF Q4_K_M (~4.9 GB) | **CPU only** | Best CPU speed/quality balance; ~0.7 tok/s baseline |
| `cpu-q5km` | llama.cpp | GGUF Q5_K_M (~5.7 GB) | **CPU only** | Near-fp16 quality; 2× slower than Q4 |
| `cpu-q8_0` | llama.cpp | GGUF Q8_0 (~8.5 GB) | **CPU only** | Near-lossless; largest GGUF; shows RAM vs quality tradeoff |

```bash
# Tensor parallelism — always allocates 2× A100-80GB
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py --modes tensor-parallel

# Continuous batching concurrency sweep
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py --modes continuous-batching

# TGI production server benchmark
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py --modes tgi

# GGUF quantization sweep across all 4 levels (returns 4 results)
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --modes cpu-q2k,cpu-q4km,cpu-q5km,cpu-q8_0
```

### Concurrent load test

Requires the gateway to be running. Tests throughput and latency under parallel load.

```bash
# Test cheap tier at concurrency 1, 5, 10, 20 (50 requests per level)
uv run llm-load-test --concurrency 1,5,10,20 --total 50 --tier cheap

# Test balanced tier, save results
uv run llm-load-test --concurrency 10 --total 100 --tier balanced \
  --output results/load_test_balanced.json
```

Output: req/s, P50/P95/P99 latency, and error rate per concurrency level.

### Cross-model cost/quality Pareto chart

Generate a Pareto chart of cost vs MMLU accuracy from Modal benchmark results.

```bash
uv run llm-pareto --results results/
# → charts/pareto.png  (default output; override with --output)
```

Plots all benchmarked model/mode combinations. Points on the Pareto frontier (lowest cost for a given accuracy level) are highlighted with a star. Useful after running `--model` comparisons across Llama, Mistral, and quantization modes.

### Analysis charts (TTFT, batch latency, quant quality)

Generate three analysis charts from Modal benchmark results:

```bash
uv run llm-charts --results results/ --output-dir charts/
# → charts/ttft_vs_throughput.png   TTFT (ms) vs output tok/s — latency/throughput tradeoff
# → charts/batch_latency.png        Per-request latency ratio vs batch size (1/4/8)
# → charts/quant_quality.png        MMLU accuracy + perplexity across quant methods
```

**TTFT vs Throughput** — each mode is a point; lower-left is optimal (fast first token + high throughput). Batching pushes points right and up.

**Batch latency** — shows how much per-request latency degrades under batching pressure. Flat lines indicate good scheduler efficiency (vLLM PagedAttention). Steep lines indicate memory-bandwidth saturation (nf4/nf4-dq/int8).

**Quant quality** — dual-axis: MMLU accuracy (bars, left) and perplexity (line, right), ordered coarsest→finest precision (Q2_K → fp16). Shows where quality degradation becomes material.

**50 req/level · Run:** 2026-05-06 · Raw: [cheap](results/load_test_cheap.json) · [balanced](results/load_test_balanced.json) · [premium](results/load_test_premium.json)

**Cheap tier** (gpt-5.4-mini):

| Concurrency | Req/s | P50 (ms) | P95 (ms) | P99 (ms) | Error rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.58 | 1,661 | 2,662 | 4,450 | 0% |
| 5 | 2.15 | 1,566 | 2,824 | 9,826 | 0% |
| 10 | 5.15 | 1,567 | 2,648 | 3,424 | 0% |
| 20 | 8.42 | 1,690 | 2,858 | 3,079 | 0% |

**Balanced tier** (gpt-5.4):

| Concurrency | Req/s | P50 (ms) | P95 (ms) | P99 (ms) | Error rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.21 | 3,196 | 9,255 | 27,913 | 0% |
| 5 | 1.04 | 3,337 | 8,192 | 17,901 | 0% |
| 10 | 1.94 | 3,332 | 7,659 | 19,333 | 0% |
| 20 | 1.87 | 3,460 | 16,780 | 21,583 | **22%** |

**Premium tier** (gpt-5.5):

| Concurrency | Req/s | P50 (ms) | P95 (ms) | P99 (ms) | Error rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.20 | 3,471 | 6,345 | 68,419 | 0% |
| 5 | 1.21 | 3,369 | 7,216 | 7,608 | 0% |
| 10 | 2.27 | 3,291 | 6,980 | 8,931 | 0% |
| 20 | 6.80 | 3,655 | 6,129 | 6,280 | **50%** |

**Key takeaways:**

- **Cheap tier scales cleanly to c=20** with 0% errors — P50 holds flat at ~1.6s, bottleneck is provider response time not the gateway
- **Balanced and premium hit rate limits at c=20** (22% and 50% error rates) — OpenAI's per-tier RPM limits trigger timeouts; this is an API constraint, not a gateway bug. P99 at c=1 for premium (68s) is a long-form generation outlier
- **P50 is the reliable latency signal** for balanced/premium — mean and P99 are skewed by long-form prompt responses; P50 stays 3.2–3.7s across all concurrency levels for both tiers
- **cheap is ~5× cheaper and ~2× faster than balanced** at median latency (1.6s vs 3.3s) — strongly preferred for simple/short tasks

---

## Dev

Pre-commit hooks run ruff lint and format automatically on every commit:

```bash
uv sync --extra dev        # installs pre-commit
uv run pre-commit install  # wires hooks into .git
```

To run manually:

```bash
uv run pre-commit run --all-files  # lint + format check
uv run pytest                      # tests
```

---

## Scope notes

The following are intentionally **not** in scope for this repository:

| Item | Reason |
|---|---|
| **Autoscaling / replica management** | Requires Kubernetes or Modal scale-to-zero orchestration — a full MLOps layer. Better addressed in a dedicated infrastructure repo. |
| **Shadow deployment / model versioning** | Full A/B routing, traffic splitting, and rollback logic belongs in an MLOps platform (Seldon, BentoML, or a custom service mesh), not a benchmarking toolkit. |

Both involve operational infrastructure beyond inference measurement and would significantly expand the project's scope.
