# llm-inference-benchmarking

Cost-aware LLM routing gateway and benchmarking toolkit. Measures latency, throughput, cost, and quality across routing tiers and quantization formats — including zero-shot MMLU evaluation, task-specific LLM-as-judge eval, and cross-provider A/B testing.

---

## Architecture

Three independent components: a **cost-aware routing gateway**, a **GPU quantization benchmark**, and an **LLM eval + A/B testing harness**.

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

### GPU Quantization Benchmark

```
GPU containers per mode, run in parallel
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
| **gptq** | HuggingFace | **7,375** | **31.6** | **34.8** | 278.1 | 5,495 | 76% | **$0.0088** |
| **vllm** | vLLM | 8,776 | — | 29.1 | 222.3 | ~16 GB | **94%** | $0.0101 |
| **fp16** | HuggingFace | 9,533 | 41.4 | 26.7 | 203.2 | 17,321 | 74% | $0.0110 |
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
| **vllm** | 85.7% | **95.0%** | **100%** | **100%** | **94%** |
| fp16 | 78.6% | 70.0% | 66.7% | 85.7% | 74% |
| gptq | 78.6% | 70.0% | 77.8% | 85.7% | 76% |

> vLLM scores highest across all subjects. GPTQ INT4 compression hurts ML/DL questions most — quantization degrades nuanced reasoning more than factual recall.

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

**Results** (n=50, judge=`gpt-5.4-mini`; raw data: [eval cheap](results/eval_2026-05-18T01-10-11.json) · [eval premium](results/eval_2026-05-18T01-17-38.json)):

| Tier | Model | Avg score | Latency (ms) | Cost/run |
|---|---|---:|---:|---:|
| cheap | gpt-5.4-mini | 9.10/10 | 1,408 | $0.003 |
| premium | claude-opus-4-6 | 8.72/10 | 9,423 | $1.64 |

Score by task type:

| Task type | cheap | premium | Δ |
|---|---:|---:|---:|
| code | 8.4 | 7.2 | −1.2 ⚠ |
| instruction_following | 7.6 | 8.1 | +0.5 |
| qa | 9.8 | 9.8 | 0.0 |
| reasoning | 9.9 | 9.2 | −0.7 ⚠ |
| summarization | 9.1 | 9.3 | +0.2 |

> Regressions on code/reasoning reflect judge bias (gpt-5.4-mini scores OpenAI-style responses higher). The A/B run below uses an independent judge for a fairer cross-provider comparison.

### A/B Testing

Routes the same 50 prompts through two variants in parallel, scores both with an independent LLM judge, and reports win rate + cost delta. Also available as `POST /ab` via the gateway API.

```bash
uv run python -m llm_inference_benchmarking.ab_router \
  --variant-a '{"tier":"cheap"}' --variant-b '{"tier":"balanced"}' \
  --output results/ab_out.json
uv run python -m llm_inference_benchmarking.ab_router \
  --variant-a '{"tier":"cheap"}' --variant-b '{"tier":"balanced"}' --dry-run
```

**Results** (n=50 each run; raw data: [cheap vs balanced](results/ab_2026-05-18T01-20-22.json)):

| Run | Variant A | Variant B | Judge | A score | B score | A win rate | Cost A | Cost B |
|---|---|---|---|---:|---:|---:|---:|---:|
| same-provider | gpt-5.4-mini | gpt-5.4 | gpt-5.5 | 8.86 | 8.98 | 10% | $0.013 | $0.070 |
| cross-provider | gpt-5.4-mini | claude-opus-4-6 | gpt-5.4 | 9.34 | 9.12 | 26% | $0.012 | $1.580 |

- **cheap vs balanced:** balanced adds 0.12 quality points at **5.4× higher cost** — worth it only for code and complex reasoning
- **cheap vs premium:** cheap scores higher on average; claude-opus-4-6 wins more individual matchups but is **132× more expensive** and **6× slower**
- Independent judge used in both runs to avoid self-grading bias

---

## Dev

```bash
uv sync --group dev        # install dev deps + pre-commit
uv run pre-commit install  # wire hooks into .git
make ci-test               # lint (ruff check --fix + format) + pytest
```
