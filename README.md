# llm-inference-benchmarking

[![CI](https://github.com/ravicharanketha/llm-inference-benchmarking/actions/workflows/ci.yml/badge.svg)](https://github.com/ravicharanketha/llm-inference-benchmarking/actions/workflows/ci.yml)

Cost-aware LLM routing gateway and benchmarking toolkit. Measures latency, cost, and quality tradeoffs across routing tiers (gateway benchmark) and quantization formats (Modal GPU benchmark).

---

## Architecture

```
Client request (prompt, tier, role)
        │
        ▼
RoutingPolicyEngine   ←─ GATEWAY_FORCE_TIER / auto heuristic
        │  tier → backend → model
        ▼
GatewayClient         ←─ LangChain adapters (OpenAI / Claude / Ollama / vLLM)
        │
        ▼
  Usage normalization (tokens, latency, estimated cost)
        │
        ├─→ SQLite ledger  (GATEWAY_LEDGER_DB)
        └─→ Prometheus     (GET /metrics)
```

**Tiers:**

| Tier | Default model | Use when |
|---|---|---|
| `cheap` | gpt-5.4-mini | Fast, simple tasks — rewrites, classification, short Q&A |
| `balanced` | gpt-5.4 | General-purpose agent workloads |
| `premium` | gpt-5.5 | Complex reasoning, long-form synthesis |
| `auto` | heuristic | Prompt length + role + keyword routing |

> Defaults above assume no local Ollama. If Ollama is running, `cheap` routes to the configured local model instead of `gpt-5.4-mini`.

**Backends:** OpenAI, Anthropic Claude, Ollama (local), vLLM (self-hosted)

**FastAPI endpoints:** `POST /generate` · `GET /health` · `GET /usage/summary` · `GET /metrics`

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

**Model:** `unsloth/Meta-Llama-3.1-8B-Instruct` · **GPU:** NVIDIA A10G · **GPU cost:** $1.10/hr · **Total run time:** ~36.7 min · **Run:** 2025-04-30 · **Raw data:** [results/modal_quant_a10g.json](results/modal_quant_a10g.json)

### Core metrics

| Mode | Mean Latency (ms) | P95 Latency (ms) | Output tok/s | VRAM (MB) | Perplexity | MMLU (50q) | Cost / 1k out tok (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **vllm** | 8,477 | 8,638 | 30.2 | — ¹ | n/a ² | **94%** | $0.0101 |
| **spec-dec** | 8,576 | 10,210 | 30.9 | 17,694 | 5.099 | — | $0.0099 |
| **fp16** | 9,558 | 9,559 | 26.7 | 15,318 | 5.099 | — | $0.0114 |
| **nf4** | 10,291 | 10,483 | 24.7 | 5,910 | 5.274 | — | $0.0124 |
| **nf4-dq** | 16,385 | 16,453 | 15.7 | 5,574 | 5.277 | — | $0.0195 |
| **int8** | 30,850 | 31,132 | 8.3 | 12,556 | 5.100 | — | $0.0368 |
| **awq** | 37,273 | 37,618 | 3.8 | 5,464 | 5.347 | — | $0.0804 |

> ¹ vLLM pre-allocates a managed memory pool; `torch.cuda.memory_reserved()` reports 0 — actual GPU usage is ~16 GB.  
> ² vLLM does not expose per-token NLL, so perplexity cannot be computed.

> **Note on MMLU (archived results):** The results file was produced with a bug where the accuracy denominator used 100 (full list) but only 50 questions were evaluated, halving the reported score for the second batch of modes. This is now fixed — new runs will correctly report accuracy over 50 questions. The perplexity values in the table are unaffected and remain the authoritative quality signal.

### Batch throughput (output tok/s)

| Mode | Batch 1 | Batch 4 | Batch 8 |
|---|---:|---:|---:|
| **vllm** | **30.2** | **114.4** | **225.1** |
| fp16 | 26.8 | 103.5 | 202.5 |
| nf4 | 24.6 | 27.7 | 54.9 |
| nf4-dq | 15.6 | 28.4 | 56.4 |
| int8 | 8.3 | 30.1 | 60.1 |
| awq | 6.8 | 8.1 | 16.5 |
| spec-dec | 30.9 | — | — |

### Time to first token (mean)

| Mode | TTFT mean (ms) |
|---|---:|
| fp16 | 40.3 |
| spec-dec | 42.2 |
| awq | 40.5 |
| int8 | 163.7 |
| nf4 | 145.6 |
| nf4-dq | 142.7 |

### Analysis

**Best overall: `spec-dec`** — fastest latency (8.6s), highest throughput (30.9 tok/s), lowest cost ($0.0099/1k), same perplexity as fp16 (5.099). Requires 17.7 GB VRAM (main model + 1B draft). Use this when the A10G has headroom.

**Best balance of speed and VRAM: `nf4`** — nearly matches fp16 throughput (24.7 vs 26.7 tok/s) at 62% less VRAM (5.9 GB vs 15.3 GB), only 7% slower, 9% more expensive. Clear winner when VRAM is constrained.

**Avoid `int8` and `awq` for latency-sensitive workloads:**
- `int8` is 3.2× slower than fp16 with no perplexity gain (5.100 vs 5.099) — quantization overhead outweighs VRAM savings on A10G
- `awq` has the worst latency (37.3s, ~3.9× slower than fp16), poor throughput (3.8 tok/s single, 16.5 batch-8), and highest cost — the pre-quantized checkpoint bottlenecks on dequantization

**vLLM vs fp16 HuggingFace (same model, same GPU):** vLLM achieves 30.2 tok/s batch-1 vs fp16's 26.7 tok/s (+13%), and 225.1 vs 202.5 tok/s at batch-8 (+11%). Latency is similar (8.5s vs 9.6s mean). The throughput gain comes from PagedAttention's efficient KV cache management — more meaningful at higher concurrency than single-request benchmarks. MMLU accuracy is 94% (47/50), the highest of any mode.

**fp16 `batch-8` throughput (202.5 tok/s) vs batch-1 (26.8 tok/s):** 7.6× improvement — batching is critical for throughput-optimized serving. `nf4` and `nf4-dq` scale more modestly (batch-8 ~2.2× batch-1), likely memory-bandwidth limited at 4-bit precision. vLLM's batch-8 (225.1 tok/s) further edges out fp16 HF.

**Perplexity is stable across all HF modes:** 5.099–5.347 range confirms 4-bit and 8-bit quantization is safe for this model at this scale. The ~0.05 delta between fp16 (5.099) and nf4 (5.274) is negligible in practice.

**Decision guide:**

| Constraint | Recommended mode |
|---|---|
| Production serving (high concurrency) | vllm |
| Lowest single-request latency | spec-dec |
| VRAM ≤ 8 GB, latency matters | nf4 |
| VRAM ≤ 6 GB | nf4-dq |
| Baseline / reproducibility reference | fp16 |

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

See [.env.example](.env.example) for the full reference including model overrides, vLLM config, custom pricing, and Modal benchmark options.

---

## Running Benchmarks

### Gateway benchmark (tier/cost/latency)

Requires provider credentials in `.env` (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`). The benchmark calls providers directly via LangChain — the gateway server does **not** need to be running.

```bash
uv run llm-gateway-bench --iterations 3 --output results/gateway_benchmark_snapshot.json
```

### Quantization benchmark (Modal GPU)

Runs six quantization modes on a Modal GPU. Requires a Modal account (`modal setup` once per machine).

Each GPU automatically gets its own output file — running on a different GPU never overwrites another GPU's results.

```bash
# Run all modes on A10G → writes results/modal_quant_a10g.json
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py

# Run specific modes only
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --modes fp16,nf4,awq

# Run on a different GPU → writes results/modal_quant_a100_40gb.json
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --gpu A100-40GB

# Re-run specific modes and merge into the existing GPU results file
# Replaces those modes in-place, preserves modes not re-run, other GPU files untouched
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --modes fp16,nf4,awq \
  --merge
```

**Supported GPUs:** `T4` ($0.59/hr) · `A10G` ($1.10/hr) · `A100-40GB` ($3.70/hr) · `A100-80GB` ($4.00/hr) · `H100` ($6.45/hr)

**Supported modes:** `fp16` · `int8` · `nf4` · `nf4-dq` · `awq` · `spec-dec` · `vllm`

The `vllm` mode runs the same model through vLLM's engine (PagedAttention, continuous batching) and reports the same latency/throughput schema as the HuggingFace modes for direct comparison. Perplexity is not available in vLLM mode.

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

**Results (cheap tier, 50 req/level, gateway routing to OpenAI gpt-5.4-mini):** · **Run:** 2026-04-30 · **Raw data:** [results/load_test_cheap.json](results/load_test_cheap.json)

| Concurrency | Req/s | P50 (ms) | P95 (ms) | P99 (ms) | Error rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.58 | 1,661 | 2,662 | 4,450 | 0% |
| 5 | 2.15 | 1,566 | 2,824 | 9,826 | 0% |
| 10 | 5.15 | 1,567 | 2,648 | 3,424 | 0% |
| 20 | 8.42 | 1,690 | 2,858 | 3,079 | 0% |

Gateway scales linearly from 0.58 req/s (c=1) to 8.42 req/s (c=20) with no errors. P50 latency stays flat at ~1.6s across all concurrency levels — the bottleneck is provider response time, not the gateway itself. P99 spike at c=5 (9.8s) is a cold-start outlier; P99 stabilises under higher concurrency.

---

## Dev

```bash
uv run ruff check .
uv run pytest
```
