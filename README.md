# llm-inference-benchmarking

Cost-aware LLM routing gateway and benchmarking toolkit. Measures latency, throughput, cost, and quality across routing tiers and quantization formats — including zero-shot MMLU evaluation, task-specific LLM-as-judge eval with a deterministic release gate, FLOPs roofline analysis, latency-cost autoscaler signal, and cross-provider A/B testing.

---

## Architecture

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
        │                          if metadata.live_metrics present → autoscaler_signal()
        │                          biases tier (up→premium, down→cheap, hold→heuristics)
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

### GPU Quantization Benchmark

```
GPU containers per mode, run in parallel
        │
        ├─ Load model  (HuggingFace / vLLM engine)
        ├─ Latency     (mean / P95 / TTFT over 5 bench prompts × 3 iterations)
        ├─ Throughput  (batch 1 / 4 / 8 output tok/s)
        ├─ Perplexity  (WikiText-2, HF modes only)
        ├─ MMLU        (50-question log-prob scoring, zero-shot)
        └─ FLOPs funnel  ←─ arithmetic intensity, roofline bound, achieved MFU %,
                            compute vs memory bound, peak theoretical tok/s
        │
        ▼
Results merged → results/modal_quant_<gpu>.json
        Each mode entry gains a "flops_funnel" sub-dict with roofline analysis.
```

### Eval Harness + A/B Testing

```
50 prompts × tier (cheap / balanced / premium)
        │
        ├─ Model response    (parallel, up to 5 concurrent)
        ├─ LLM-as-judge      (independent model scores 0–10)
        └─ Regression check  (vs prior run, flags Δ > 0.5)
        │
        ▼
results/eval_<timestamp>.json   ←─ consumed by quality router

A/B: same prompts → two variants in parallel → independent judge → win rate + cost delta
POST /ab endpoint exposes this via the gateway API
```

---

## Quantization Results

**Model:** `unsloth/Meta-Llama-3.1-8B-Instruct` · **GPU:** NVIDIA A10G ($1.10/hr) · **Raw data:** [results/modal_quant_a10g.json](results/modal_quant_a10g.json)

TTFT ≈ prefill duration. GPTQ has the fastest prefill (Marlin INT4 kernels); NF4/int8 are slower due to dequantization overhead on attention projections.

| Mode | Engine | Latency (ms) | TTFT (ms) | Tok/s | Batch 8 tok/s | VRAM (MB) | MMLU | Cost/1k out (USD) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **tensor-parallel** | vLLM (2× A100-80GB) | **1,762** | — | **146.7** | **1,106.0** | 2× 80 GB | **94%** | $0.0042 ¹ |
| **fp8** | vLLM | 4,665 | — | 54.9 | 420.4 | ~16 GB | ⚠ 6% | $0.0056 |
| **gptq** | HuggingFace | **7,375** | **37.1** | **28.9** | 278.1 | 5,495 | 92% | **$0.0088** |
| **vllm** | vLLM | 8,776 | 37.8 | 29.2 | 222.3 | 19,154 | **94%** | $0.0101 |
| **fp16** | HuggingFace | 9,522 | 41.5 | 26.5 | 202.9 | 17,321 | **94%** | $0.0110 |
| **nf4** | HuggingFace | 10,403 | 145.6 | 25.3 | 54.9 | 7,787 | 74% | $0.0121 |
| **nf4-dq** | HuggingFace | 16,400 | 142.7 | 15.7 | 56.4 | 5,541 | 74% | $0.0195 |
| **int8** | HuggingFace | 30,846 | 163.7 | 8.3 | 60.1 | 12,296 | 74% | $0.0368 |
| **cpu-q4km** | llama.cpp | ~95,000 | — | ~0.7 | — | — | ~70% ² | CPU only |

> ¹ tensor-parallel runs on 2× A100-80GB ($8.00/hr combined); cost reflects the 2-GPU pair.
> ² CPU modes run a 20-question MMLU subset. Treat as directional only.
> ⚠ **fp8**: SW-emulated on A10G — quality degrades to 6% MMLU. Hardware-native FP8 requires H100/H200.

### Model Evaluation

#### MMLU accuracy by subject (zero-shot log-probability, 50-question CS/ML subset)

| Mode | CS & Programming | ML & Deep Learning | Systems & Networking | Statistics & Math | Overall |
|---|---:|---:|---:|---:|---:|
| **fp16** | 85.7% | **95.0%** | **100%** | **100%** | **94%** |
| **vllm** | 85.7% | **95.0%** | **100%** | **100%** | **94%** |
| gptq | 85.7% | 95.0% | 100% | 85.7% | 92% |

> All three modes score within 2pp overall. GPTQ misses one statistics question (6/7 vs 7/7) — the only difference from fp16. Confirms INT4 quantization does not meaningfully degrade accuracy on this subset.

### Decision guide

| Constraint | Recommended mode |
|---|---|
| Multi-GPU batch serving (2× A100-80GB) | tensor-parallel |
| H100 single-GPU production | fp8 |
| Single GPU, lowest latency | gptq |
| Single GPU, best throughput + accuracy | vllm |
| VRAM ≤ 8 GB | nf4 |
| VRAM ≤ 6 GB | nf4-dq |
| Baseline / reproducibility reference | fp16 |
| TCO comparison at ≤1 req/min | cpu-q4km (or cpu-q8_0 for quality) |

---

## Gateway Results

**Backend:** OpenAI · **Raw data:** [results/gateway_benchmark_snapshot.json](results/gateway_benchmark_snapshot.json)

| Tier | Model | Mean (ms) | P50 (ms) | P95 (ms) | Cost/req (USD) |
|---|---|---:|---:|---:|---:|
| cheap | gpt-5.4-mini | 4,182 | 2,204 | 9,897 | $0.000701 |
| balanced | gpt-5.4 | 8,972 | 5,640 | 18,775 | $0.004024 |
| premium | gpt-5.5 | 11,129 | 4,945 | 28,148 | $0.004911 |

- `cheap` is **5.7× cheaper** than `balanced` and **2.1× faster** on mean latency — strongly preferred for simple/short tasks
- P50 is the reliable signal for `balanced`/`premium` — both have a long tail; P50 stays 4.9–5.6s while mean runs 9–11s
- `balanced` and `premium` cost delta is small (~22%) — `premium` is better value for complex tasks

### Concurrent load (50 req/level)

Raw: [cheap](results/load_test_cheap.json) · [balanced](results/load_test_balanced.json) · [premium](results/load_test_premium.json)

| Tier | Concurrency | Req/s | P50 (ms) | P95 (ms) | Error rate |
|---|---:|---:|---:|---:|---:|
| cheap | 1 | 0.58 | 1,661 | 2,662 | 0% |
|  | 5 | 2.15 | 1,566 | 2,824 | 0% |
|  | 10 | 5.15 | 1,567 | 2,648 | 0% |
|  | 20 | 8.42 | 1,690 | 2,858 | 0% |
| balanced | 1 | 0.21 | 3,196 | 9,255 | 0% |
|  | 5 | 1.04 | 3,337 | 8,192 | 0% |
|  | 10 | 1.94 | 3,332 | 7,659 | 0% |
|  | 20 | 1.87 | 3,460 | 16,780 | **22%** |
| premium | 1 | 0.20 | 3,471 | 6,345 | 0% |
|  | 5 | 1.21 | 3,369 | 7,216 | 0% |
|  | 10 | 2.27 | 3,291 | 6,980 | 0% |
|  | 20 | 6.80 | 3,655 | 6,129 | **50%** |

- **Cheap tier scales cleanly to c=20** (0% errors, P50 flat ~1.6s) — bottleneck is provider response time, not the gateway
- **Balanced and premium hit rate limits at c=20** (22% / 50% errors) — OpenAI per-tier RPM caps; P50 stays stable even under load

---

## Eval & A/B Results

### LLM Eval Harness

**Results** (n=50, judge=`gpt-5.4-mini`; raw data: [run 1](results/eval_2026-05-27T01-00-17.json) · [run 2](results/eval_2026-05-27T01-01-58.json) · [run 3](results/eval_2026-05-27T01-02-56.json)):

| Tier | Model | Avg score | Latency (ms) | Cost/run | Gate |
|---|---|---:|---:|---:|---:|
| cheap | gpt-5.4-mini | 9.06/10 | 1,506 | $0.021 | ✓ passed |

Score by task type (avg across 3 runs):

| Task type | Score | Notes |
|---|---:|---|
| qa | 9.8/10 | Consistently high |
| reasoning | 9.7/10 | Consistently high |
| summarization | 9.2/10 | Stable |
| code | 8.2/10 | Lowest — judge may favor terse responses |
| instruction_following | 8.4/10 | Stable |

> All three gate runs passed (default threshold 6.0). Code and instruction_following are the soft spots but well above the gate floor. Cost/run increased vs earlier results because `total_cost_usd` now includes the judge call cost.

> Earlier cross-provider comparison (cheap vs claude-opus-4-6 premium) is in the A/B results below — cheap scores higher on average at 132× lower cost.

### A/B Testing

**Results** (n=50 each run; raw data: [cheap vs balanced](results/ab_2026-05-18T01-20-22.json)):

| Run | Variant A | Variant B | Judge | A score | B score | A win rate | Cost A | Cost B |
|---|---|---|---:|---:|---:|---:|---:|---:|
| same-provider | gpt-5.4-mini | gpt-5.4 | gpt-5.5 | 8.86 | 8.98 | 10% | $0.013 | $0.070 |
| cross-provider | gpt-5.4-mini | claude-opus-4-6 | gpt-5.4 | 9.34 | 9.12 | 26% | $0.012 | $1.580 |

- **cheap vs balanced:** balanced adds 0.12 quality points at **5.4× higher cost** — worth it only for code and complex reasoning
- **cheap vs premium:** cheap scores higher on average; claude-opus-4-6 wins more individual matchups but is **132× more expensive** and **6× slower**

---

## Quickstart

```bash
uv sync --group dev                  # install
cp .env.example .env                 # add API keys
uv run uvicorn llm_inference_benchmarking.gateway:app --host 0.0.0.0 --port 8010

curl http://localhost:8010/health
```

---

## Configuration

Minimum required keys in `.env`:

```bash
GATEWAY_API_KEY=your-secret        # auth header value
OPENAI_API_KEY=sk-...              # or ANTHROPIC_API_KEY for Claude backend
AGENT_LLM=openai                   # openai | claude | vllm
```

See [.env.example](.env.example) for the full reference including model overrides, vLLM config, custom pricing, benchmark options, rate limiting, SLA caps, and quality routing.

---

## Running Benchmarks

### Gateway benchmark (tier/cost/latency)

Requires provider credentials in `.env`. The benchmark calls providers directly — the gateway server does **not** need to be running.

```bash
uv run llm-gateway-bench --iterations 3 --output results/gateway_benchmark_snapshot.json
```

**Prompt caching benchmark** — measures cold vs warm latency (Claude automatic for prompts >1024 tokens, OpenAI same threshold):

```bash
uv run llm-gateway-bench --cache
# → writes results/cache_benchmark_snapshot.json
```

### Quantization benchmark (GPU)

Runs modes in parallel on a cloud GPU. Requires a Modal account (`modal setup` once per machine).

**Supported GPUs:** `T4` ($0.59/hr) · `A10G` ($1.10/hr) · `A100-40GB` ($3.70/hr) · `A100-80GB` ($4.00/hr) · `H100` ($6.45/hr)

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

### Concurrent load test

Requires the gateway to be running.

```bash
uv run llm-load-test --concurrency 1,5,10,20 --total 50 --tier cheap
uv run llm-load-test --concurrency 10 --total 100 --tier balanced \
  --output results/load_test_balanced.json
```

### Analysis charts

```bash
uv run llm-pareto --results results/
uv run llm-charts --results results/ --output-dir charts/
```

### LLM Eval Harness

Task-specific evaluation with LLM-as-judge scoring across 50 prompts (summarization, reasoning, code, Q&A, instruction-following). Writes `results/eval_<timestamp>.json` in the same schema as quantization benchmarks so the quality router can consume them. Auto-detects a prior run for regression comparison.

```bash
uv run python -m llm_inference_benchmarking.eval --tier cheap
uv run python -m llm_inference_benchmarking.eval --tier cheap --dry-run
```

Add `--gate` to block on quality regression (exit 1 if any task type scores below `EVAL_GATE_THRESHOLD`, default `6.0`):

```bash
uv run python -m llm_inference_benchmarking.eval --tier cheap --gate
uv run python -m llm_inference_benchmarking.eval --tier cheap --gate --gate-threshold 7.0
```

### A/B Testing

Routes the same 50 prompts through two variants in parallel, scores both with an independent LLM judge, and reports win rate + cost delta. Also available as `POST /ab` via the gateway API.

```bash
uv run python -m llm_inference_benchmarking.ab_router \
  --variant-a '{"tier":"cheap"}' --variant-b '{"tier":"balanced"}' \
  --output results/ab_out.json
```

### FLOPs Roofline Analysis

Runs automatically alongside every quantization benchmark — no separate command. Each mode entry in the results JSON gains a `flops_funnel` sub-dict with arithmetic intensity, roofline bound, achieved MFU %, and compute vs memory bound classification.

### Latency-Cost Autoscaler Signal

Pass `live_metrics` in a request's `metadata` to bias `auto` tier routing (score < 0.4 → premium, > 0.8 → cheap). Env vars: `AUTOSCALER_UP_THRESHOLD` (0.4), `AUTOSCALER_DOWN_THRESHOLD` (0.8), `AUTOSCALER_LATENCY_WEIGHT` (0.6), `AUTOSCALER_COST_WEIGHT` (0.4).

---

## Dev

```bash
uv sync --group dev        # install dev deps + pre-commit
uv run pre-commit install  # wire hooks into .git
make ci-test               # lint (ruff check --fix + format) + pytest
```
