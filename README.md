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
- P50 is the reliable signal for `balanced`/`premium` — both have a long tail from verbose responses; P50 stays 4.9–5.6s while mean runs 9–11s
- `balanced` and `premium` cost delta is small (~22%) — `premium` is better value for complex tasks
- Pricing for gpt-5.x models is placeholder until OpenAI publishes official rates; update in `cost.py`

---

## Quantization Benchmark Results

**Model:** `unsloth/Meta-Llama-3.1-8B-Instruct` · **GPU:** NVIDIA A10G · **GPU cost:** $1.10/hr · **Run:** 2026-05-07 · **Raw data:** [results/modal_quant_a10g.json](results/modal_quant_a10g.json)

> fp16, vllm, gptq rows reflect the latest run (2026-05-07). Remaining rows are from a prior A10G run and are kept for completeness; re-run those modes with `--merge` to refresh.

### Core metrics

| Mode | Engine | Mean Latency (ms) | Output tok/s | VRAM (MB) | Perplexity | MMLU | Cost / 1k out tok (USD) |
|---|---|---:|---:|---:|---:|---:|---:|
| **tensor-parallel** | vLLM (2× A100-80GB) | **1,762** | **146.7** | 2× GPU | n/a ¹ | **94%** (50q) | $0.0042 ⁴ |
| **fp8** | vLLM | 4,665 | 54.9 | — ² | n/a ¹ | ⚠ 6% (50q) | $0.0056 |
| **gptq** | HuggingFace | **7,290** | **34.7** | 5,495 | 5.307 | 76% (50q) | **$0.0088** |
| **vllm** | vLLM | 8,474 | 30.2 | — ² | n/a ¹ | **94%** (50q) | $0.0101 |
| **fp16** | HuggingFace | 9,199 | 27.7 | 17,321 | 4.303 | 80% (50q) | $0.0110 |
| **flash-attn** | HuggingFace (SDPA) | 9,541 | 26.4 | 17,321 | 5.099 | 74% (50q) | $0.0116 |
| **torch-compile** | HuggingFace | 9,597 | 26.2 | 17,321 | 5.099 | 74% (50q) | $0.0117 |
| **spec-dec** | HuggingFace | 10,259 | 23.6 | 17,321 | 5.099 | 74% (50q) | $0.0129 |
| **nf4** | HuggingFace | 10,403 | 25.3 | 7,787 | 5.275 | 74% (50q) | $0.0121 |
| **continuous-batching** | vLLM async | — ³ | 29.1 | — ² | n/a ¹ | — ³ | $0.0105 |
| **nf4-dq** | HuggingFace | 16,400 | 15.7 | 5,541 | 5.277 | 74% (50q) | $0.0195 |
| **int8** | HuggingFace | 30,846 | 8.3 | 12,296 | 5.100 | 74% (50q) | $0.0368 |
| **cpu-q4km** | llama.cpp (CPU, Q4_K_M) | ~95,000 | ~0.7 | — | — | ~70% (20q) ⁵ | CPU only |

> ¹ vLLM does not expose per-token NLL, so perplexity cannot be computed.
> ² vLLM pre-allocates a managed memory pool; `torch.cuda.memory_reserved()` reports 0 — actual GPU usage ~16 GB.
> ³ continuous-batching uses an async queue engine — single-request latency and quality scoring not available in the async path.
> ⁴ tensor-parallel runs on 2× A100-80GB ($8.00/hr combined); cost reflects the 2-GPU pair.
> ⁵ CPU modes run a 20-question MMLU subset (too slow for 50 questions). Treat as directional only.
> ⚠ **fp8 (6% MMLU)**: SW-emulated FP8 on A10G degrades output quality — hardware-native FP8 requires H100/H200. Do not use fp8 on A10G.

### Batch throughput (output tok/s)

> "—" = not benchmarked at that batch size. vLLM modes (tensor-parallel, fp8) run batch 1 and 8 only; continuous-batching uses an async queue so batch 1/4 are not meaningful; spec-dec skips batch 8 (draft acceptance degrades with diverse batches); flash-attn and torch-compile run batch 1 and 8 only.

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

### Time to first token (TTFT)

TTFT ≈ prefill phase duration. Quantization affects prefill differently than decode: 4-bit GPTQ has the fastest prefill (Marlin INT4 kernels); NF4/nf4-dq/int8 are slower due to dequantization overhead on the attention projection.

| Mode | TTFT / Prefill (ms) |
|---|---:|
| **gptq** | **31.0** |
| fp16 | 41.1 |
| flash-attn | 40.4 |
| torch-compile | 39.8 |
| spec-dec | 42.2 |
| nf4 | 145.6 |
| nf4-dq | 142.7 |
| int8 | 163.7 |

> Per-token decode timing is instrumented for fp16 and gptq only: fp16 decode=35.9 ms/tok (prefill/decode ratio 1.15), gptq decode=28.5 ms/tok (ratio 1.09). Both near-1 ratios confirm this is a memory-bandwidth-bound workload. Other HF modes have TTFT from the streamer callback but no per-token decode split yet.

### MMLU accuracy by subject

| Mode | CS & Programming | ML & Deep Learning | Systems & Networking | Statistics & Math | Overall |
|---|---:|---:|---:|---:|---:|
| **vllm** | 85.7% (12/14) | **95.0%** (19/20) | **100%** (9/9) | **100%** (7/7) | **94%** (47/50) |
| fp16 | 78.6% (11/14) | 75.0% (15/20) | 88.9% (8/9) | 85.7% (6/7) | 80% (40/50) |
| gptq | 78.6% (11/14) | 70.0% (14/20) | 77.8% (7/9) | 85.7% (6/7) | 76% (38/50) |

> vLLM scores highest across all subjects. GPTQ INT4 compression hurts ML/DL questions most (70% vs 75% fp16) — quantization degrades nuanced reasoning more than factual recall.

### Decision guide

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
uv sync --group dev                  # install
cp .env.example .env                 # add API keys
uv run uvicorn llm_inference_benchmarking.gateway:app --host 0.0.0.0 --port 8010

# verify
curl http://localhost:8010/health
curl -X POST http://localhost:8010/generate \
  -H "Content-Type: application/json" -H "x-api-key: $GATEWAY_API_KEY" \
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

Each GPU gets its own output file so runs on different GPUs never overwrite each other.

```bash
# Run all modes on A10G (default) → results/modal_quant_a10g.json
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py

# Run specific modes; --merge updates those rows in the existing file
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --modes fp16,gptq,nf4 --merge

# Cross-model or cross-GPU
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 --gpu A100-40GB
```

**Supported GPUs:** `T4` ($0.59/hr) · `A10G` ($1.10/hr) · `A100-40GB` ($3.70/hr) · `A100-80GB` ($4.00/hr) · `H100` ($6.45/hr)

| Mode | Engine | Quantization | GPU | Notes |
|---|---|---|---|---|
| `fp16` | HuggingFace | None | Any | Baseline reference |
| `int8` | HuggingFace | bitsandbytes 8-bit | Any | Avoid on A10G — nf4 dominates on every metric |
| `nf4` / `nf4-dq` | HuggingFace | 4-bit NormalFloat | Any | Best speed/VRAM balance; nf4-dq saves 2 GB extra |
| `spec-dec` | HuggingFace | fp16 + 1B draft | Any | Only useful for predictable, repetitive outputs |
| `vllm` | vLLM | fp16 (PagedAttention) | Any | Best single-GPU quality (94% MMLU); production default |
| `gptq` | HuggingFace | INT4 GPTQ (pre-quantized) | Any | Fastest HF mode; Marlin INT4 beats fp16 on latency |
| `fp8` | vLLM | Dynamic FP8 | **H100 ideal** | HW-native on H100; SW-emulated on A10G (6% MMLU — broken) |
| `flash-attn` | HuggingFace | fp16 + PyTorch SDPA | Any | Gains at long sequence lengths |
| `torch-compile` | HuggingFace | fp16 + JIT fusion | Any | First call slow; gains on repeated inference patterns |
| `tensor-parallel` | vLLM | fp16, TP=2 | **2× A100-80GB** | Highest throughput (146.7 tok/s); cheapest per token at scale |
| `continuous-batching` | vLLM async | fp16 | Any | Concurrency sweep 1/4/8/16; measures queue depth efficiency |
| `tgi` | HuggingFace TGI 2.4 | fp16 | Any | Production TGI server; TTFT via streaming |
| `cpu-q2k` | llama.cpp | GGUF Q2_K (~2.9 GB) | CPU only | Lowest RAM; noticeable quality loss |
| `cpu-q4km` | llama.cpp | GGUF Q4_K_M (~4.9 GB) | CPU only | Best CPU speed/quality balance |
| `cpu-q5km` | llama.cpp | GGUF Q5_K_M (~5.7 GB) | CPU only | Near-fp16 quality; 2× slower than Q4 |
| `cpu-q8_0` | llama.cpp | GGUF Q8_0 (~8.5 GB) | CPU only | Near-lossless; largest RAM footprint |

### Concurrent load test

Requires the gateway to be running. Tests throughput and latency under parallel load.

```bash
uv run llm-load-test --concurrency 1,5,10,20 --total 50 --tier cheap
uv run llm-load-test --concurrency 10 --total 100 --tier balanced \
  --output results/load_test_balanced.json
```

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

- **Cheap tier scales cleanly to c=20** (0% errors, P50 flat at ~1.6s) — bottleneck is provider response time, not the gateway
- **Balanced and premium hit rate limits at c=20** (22% / 50% errors) — OpenAI per-tier RPM caps, not a gateway bug; P50 stays 3.2–3.7s even under load

### Cross-model Pareto chart + analysis charts

```bash
uv run llm-pareto --results results/       # → charts/pareto.png (cost vs MMLU frontier)
uv run llm-charts --results results/ --output-dir charts/
# → charts/ttft_vs_throughput.png   TTFT vs output tok/s (lower-left = optimal)
# → charts/batch_latency.png        per-request latency degradation under batching
# → charts/quant_quality.png        MMLU accuracy + perplexity across quant methods
```

---

## Dev

```bash
uv sync --group dev        # install dev deps + pre-commit
uv run pre-commit install  # wire hooks into .git
make ci-test               # lint (ruff check --fix + format) + pytest
```
