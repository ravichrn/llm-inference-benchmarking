# llm-inference-benchmarking

Cost-aware LLM routing gateway and benchmarking toolkit. Measures latency, throughput, cost, and quality across routing tiers and quantization formats — including zero-shot MMLU evaluation, task-specific LLM-as-judge eval with a deterministic release gate, FLOPs roofline analysis, latency-cost autoscaler signal, and cross-provider A/B testing. Deployable as a live HTTPS endpoint on Modal.

---

## Architecture

### Gateway

Each request goes through: rate limiter → routing policy (prompt length + role + keyword heuristics, or `auto` biased by autoscaler signal) → budget cap check → SLA p99 check → quality-aware tier selection → LangChain backend call → usage logged to SQLite + Prometheus.

**Routing tiers:**

| Tier | Default model | Use when |
|---|---|---|
| `cheap` | gpt-5.4-mini | Fast, simple tasks — rewrites, classification, short Q&A |
| `balanced` | gpt-5.4 | General-purpose agent workloads |
| `premium` | gpt-5.5 | Complex reasoning, long-form synthesis |
| `auto` | heuristic | Routes based on prompt length, role, and keyword signals |

> Defaults above assume no local Ollama. If Ollama is running, `cheap` routes to the configured local model instead of `gpt-5.4-mini`.

**Supported backends:** OpenAI · Anthropic Claude · Ollama (local) · vLLM (self-hosted)

**FastAPI endpoints:** `POST /generate` · `POST /ab` · `GET /health` · `GET /usage/summary` · `GET /metrics` · `GET /sla/status`

**Live deployment:** `modal_gateway.py` wraps the FastAPI app as a Modal HTTPS endpoint with a persistent SQLite ledger and `keep_warm=1`. See [Running the live gateway](#running-the-live-gateway).

### GPU Quantization Benchmark

Runs modes in parallel on Modal GPU containers. Each mode measures latency (mean/P95/TTFT), throughput (batch 1/4/8), VRAM, perplexity (HF modes), zero-shot MMLU accuracy, and FLOPs roofline (arithmetic intensity, MFU %, compute vs memory bound). Results merge into `results/modal_quant_<gpu>.json`.

**Benchmark modes:**

| Mode | Engine | GPU | Notes |
|---|---|---|---|
| `fp16` | HuggingFace | A10G | Full precision baseline |
| `int8` | bitsandbytes | A10G | 8-bit; compute-bound on Ampere |
| `nf4` | bitsandbytes | A10G | 4-bit NormalFloat; best VRAM/quality tradeoff |
| `spec-dec` | HuggingFace | A10G | Speculative decoding with 1B draft model |
| `vllm` | vLLM | A10G | PagedAttention; best throughput |
| `sglang` | SGLang | A10G | RadixAttention + chunked prefill; comparable to vLLM on diverse prompts |
| `gptq` | gptqmodel | A10G | INT4 with Marlin/exllama2 CUDA kernels |
| `gptq-triton` | gptqmodel | A10G | INT4 with Triton kernel backend; benchmarks compiler vs hand-tuned kernels |
| `awq` | autoawq | A10G | INT4 AWQ with fused attention+MLP kernels; best quality at 4-bit |
| `flash-attn` | HuggingFace (SDPA) | A10G | Tiled attention; O(n) memory |
| `torch-compile` | HuggingFace | A10G | `torch.compile()` graph optimization |
| `continuous-batching` | vLLM async | A10G | Dynamic batching for multi-tenant serving |
| `fp8` | vLLM | **H100** | Native FP8 tensor cores (SW emulation on A10G) |
| `tensor-parallel` | vLLM | **2× A100-80GB** | Sharded across 2 GPUs |
| `cpu-q4km` | llama.cpp | CPU | GGUF Q4_K_M; edge/CPU deployment baseline |


### Eval Harness + A/B Testing

Scores 50 prompts with an independent LLM judge (0–10). Regression gate exits 1 if any task type drops below threshold. Results feed the quality router. A/B runs the same prompts through two variants in parallel and reports win rate + cost delta — also available as `POST /ab`.

---

## Quantization Results

**Model:** `unsloth/Meta-Llama-3.1-8B-Instruct` · **GPU:** NVIDIA A10G ($1.10/hr) · **Raw data:** [results/modal_quant_a10g.json](results/modal_quant_a10g.json) · [H100](results/modal_quant_h100.json) · [2× A100-80GB](results/modal_quant_a100_80gb.json)

TTFT ≈ prefill duration. GPTQ (Marlin CUDA kernels) has the fastest prefill and decode; gptq-triton is 2.6× slower showing hand-tuned CUDA beats the Triton compiler on Ampere. AWQ via vLLM scores the highest MMLU of any INT4 mode (92%) — activation-aware calibration preserves accuracy — but throughput is lower than GPTQ due to vLLM's AWQ dequantization overhead on A10G. SGLang matches vLLM throughput on diverse prompts (no shared prefix); RadixAttention advantage appears on workloads with a shared system prompt or retrieval context.

| Mode | Engine | Latency (ms) | TTFT (ms) | Tok/s | Batch 8 tok/s | VRAM (MB) | MMLU | Cost/1k out (USD) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **fp8** | vLLM (H100) | **1,116** | — | **232.0** | **1,785.2** | HBM3 ³ | **94%** | **$0.0077** ² |
| **tensor-parallel** | vLLM (2× A100-80GB) | **1,745** | — | **147.5** | **1,108.0** | 2× 80 GB | **94%** | $0.0075 ¹ |
| **gptq** | gptqmodel | **7,558** | **32.8** | **33.3** | **267.7** | 5,488 | 76% | **$0.0092** |
| **spec-dec** | HuggingFace | 8,187 | 41.5 | 31.3 | — | 17,758 | 74% | $0.0098 |
| **vllm** | vLLM | 8,780 | — | 29.1 | 222.4 | PagedAttn ³ | **94%** | $0.0105 |
| **sglang** | SGLang | 9,967 | — | 28.5 | 220.3 | PagedAttn ³ | **94%** | $0.0107 |
| **torch-compile** | HuggingFace | 9,218 | 39.2 | 27.1 | 197.9 | 17,610 | 74% | $0.0113 |
| **flash-attn** | HuggingFace (SDPA) | 9,558 | 41.7 | 26.7 | 202.4 | 17,610 | 74% | $0.0114 |
| **fp16** | HuggingFace | 9,563 | 41.8 | 26.5 | 201.4 | 17,610 | 74% | $0.0115 |
| **nf4** | HuggingFace | 10,436 | 145.0 | 24.8 | 55.4 | 7,914 | 74% | $0.0123 |
| **gptq-triton** | gptqmodel | 19,919 | 78.6 | 12.8 | 102.3 | 5,470 | 76% | $0.0239 |
| **int8** | bitsandbytes | 25,313 | 130.0 | 10.0 | 74.3 | 11,174 | 74% | $0.0306 |
| **awq** | vLLM (AWQ) | 23,606 | — | 10.7 | 84.0 | PagedAttn ³ | **92%** | $0.0286 |
| **cpu-q4km** | llama.cpp | ~95,000 | — | ~0.7 | — | — | ~70% ⁴ | CPU only |

> ¹ tensor-parallel runs on 2× A100-80GB ($8.00/hr combined); cost reflects the 2-GPU pair.
> ² fp8 runs on H100 ($6.45/hr); 8.6× faster than A10G fp16 at lower cost-per-token despite higher hourly rate.
> ³ vLLM, SGLang, and AWQ use PagedAttention-style KV management; VRAM is dynamically allocated rather than reserved upfront.
> ⁴ CPU MMLU uses a 20-question subset.

### Model Evaluation

#### MMLU accuracy by subject (zero-shot log-probability, 50-question CS/ML subset)

| Mode | CS & Programming | ML & Deep Learning | Systems & Networking | Statistics & Math | Overall |
|---|---:|---:|---:|---:|---:|
| **vllm** | **85.7%** | **95.0%** | **100%** | **100%** | **94%** |
| **sglang** | **85.7%** | **95.0%** | **100%** | **100%** | **94%** |
| gptq | 78.6% | 70.0% | 77.8% | 85.7% | 76% |
| fp16 / int8 / nf4 * | 78.6% | 70.0% | 66.7% | 85.7% | 74% |

> \* All HuggingFace-backed modes (fp16, int8, nf4, spec-dec, flash-attn, torch-compile) produce identical predictions at temperature=0 — same model, deterministic decoding. vLLM and SGLang both score 94% using greedy decoding with their respective engines. GPTQ gains one Systems question (7/9 vs 6/9) over HF modes.

### Decision guide

| Constraint | Recommended mode |
|---|---|
| Multi-GPU batch serving (2× A100-80GB) | tensor-parallel |
| H100 single-GPU production | fp8 (232 tok/s, 94% MMLU, $0.0077/1k) |
| Single GPU, lowest latency | gptq (Marlin kernels) |
| Single GPU, best INT4 quality (accuracy > speed) | awq (92% MMLU vs gptq 76%) |
| Triton vs CUDA kernel comparison | gptq-triton vs gptq |
| Single GPU, best throughput + accuracy | vllm or sglang (equivalent on diverse prompts; sglang wins with shared-prefix workloads) |
| VRAM ≤ 8 GB | nf4 |
| Baseline / reproducibility reference | fp16 |
| CPU / edge deployment | cpu-q4km |

---

## Gateway Results

**Backend:** OpenAI · **Live endpoint:** Modal HTTPS · **Raw data:** [results/gateway_benchmark_snapshot.json](results/gateway_benchmark_snapshot.json)

### Routing decisions (live demo)

| Request | Tier | Model | Latency | Cost |
|---|---|---|---:|---:|
| Simple fact | cheap | gpt-5.4-mini | 1,644ms | $0.00002 |
| Classification | cheap | gpt-5.4-mini | 578ms | $0.00001 |
| Deep analysis (keyword) | premium | gpt-5.5 | 20,037ms | $0.00775 |
| Explicit cheap override | cheap | gpt-5.4-mini | 2,013ms | $0.00011 |
| Explicit premium override | premium | gpt-5.5 | 35,358ms | $0.01717 |
| Long prompt (length heuristic) | balanced | gpt-5.4 | 19,893ms | $0.00953 |

- cheap is **166× cheaper** than premium and **14× faster** — 3 of 6 demo requests routed there automatically
- Long prompt (100 repetitions) correctly escalates to `balanced` via length heuristic
- Keyword signals (`trade-offs`, `compare`) correctly escalate to `premium`

### Usage summary (6 requests)

| Tier | Requests | Avg latency | Total cost |
|---|---|---:|---:|
| cheap | 3 | 1,412ms | $0.00015 |
| balanced | 1 | 19,893ms | $0.00953 |
| premium | 2 | 27,698ms | $0.02491 |

### Concurrent load

Raw: [cheap](results/load_test_cheap.json) · [balanced](results/load_test_balanced.json) · [premium](results/load_test_premium.json)

Cheap scales to c=20 with 0% errors (P50 flat ~1.6s) — bottleneck is provider latency, not the gateway. Balanced and premium hit OpenAI RPM caps at c=20 (22% / 50% errors); P50 stays stable, only the tail degrades.

---

## Eval & A/B Results

### LLM Eval Harness

**Results** (n=50, judge=`gpt-5.4-mini`):

| Tier | Model | Avg score | Latency (ms) | Cost/run | Gate |
|---|---|---:|---:|---:|---:|
| cheap | gpt-5.4-mini | 9.06/10 | 1,506 | $0.021 | ✓ passed |

All three gate runs passed (threshold 6.0). QA/reasoning score highest (9.7–9.8); code lowest (8.2) — judge may favour terse responses. Raw data: [run 1](results/eval_2026-05-27T01-00-17.json) · [run 2](results/eval_2026-05-27T01-01-58.json) · [run 3](results/eval_2026-05-27T01-02-56.json)

### A/B Testing

**Results** (n=50 each run; raw data: [cheap vs balanced](results/ab_2026-05-18T01-20-22.json)):

| Run | Variant A | Variant B | Judge | A score | B score | A win rate | Cost A | Cost B |
|---|---|---|---:|---:|---:|---:|---:|---:|
| same-provider | gpt-5.4-mini | gpt-5.4 | gpt-5.5 | 8.86 | 8.98 | 10% | $0.013 | $0.070 |
| cross-provider | gpt-5.4-mini | claude-opus-4-6 | gpt-5.4 | 9.34 | 9.12 | 26% | $0.012 | $1.580 |

- **cheap vs balanced:** balanced adds 0.12 quality points at **5.4× higher cost** — worth it only for code and complex reasoning
- **cheap vs premium:** cheap scores higher on average; claude-opus-4-6 wins more individual matchups but is **132× more expensive** and **6× slower**

---

## Running the live gateway

`modal_gateway.py` deploys the full FastAPI gateway as a persistent HTTPS endpoint on Modal. The ledger is stored in a Modal volume so usage history survives container restarts.

**Setup Modal secrets (one-time):**
```bash
modal secret create openai-secret OPENAI_API_KEY=sk-...
modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
modal secret create gateway-secret GATEWAY_API_KEY=your-secret
```

**Deploy:**
```bash
# Live reload (dev)
modal serve src/llm_inference_benchmarking/modal_gateway.py

# Permanent deployment
modal deploy src/llm_inference_benchmarking/modal_gateway.py
```

**Demo — routing decisions + autoscaler signal:**
```bash
python demo_gateway.py --dry-run                                             # routing logic only, no API key
python demo_gateway.py --url https://<your-app>.modal.run --api-key $KEY    # live endpoint
```

---

## Quickstart

```bash
uv sync --group dev                  # install
cp .env.example .env                 # set GATEWAY_API_KEY, OPENAI_API_KEY, AGENT_LLM=openai
uv run uvicorn llm_inference_benchmarking.gateway:app --host 0.0.0.0 --port 8010
curl http://localhost:8010/health
```

See [.env.example](.env.example) for the full reference (model overrides, vLLM config, rate limiting, SLA caps, budget caps, quality routing).

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
# Default A10G sweep (fp16, int8, nf4, spec-dec, vllm, gptq, gptq-triton,
# awq, flash-attn, torch-compile, continuous-batching)
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py

# Merge new results into existing file (keeps untouched modes)
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py --merge

# Specialist modes (each requires separate run)
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py --modes fp8        # needs H100
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py --modes cpu-q4km  # CPU container
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --modes tensor-parallel --gpu A100-80GB                                             # 2× A100-80GB

# Cross-model or cross-GPU
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 --gpu A100-40GB
```

NVTX range markers are active around warmup, TTFT, and throughput loops — attach Nsight Systems with `nsys profile --trace=cuda,nvtx` to capture a GPU timeline.

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

```bash
uv run python -m llm_inference_benchmarking.eval --tier cheap
uv run python -m llm_inference_benchmarking.eval --tier cheap --gate            # exit 1 on regression
uv run python -m llm_inference_benchmarking.eval --tier cheap --gate --gate-threshold 7.0
```

### A/B Testing

```bash
uv run python -m llm_inference_benchmarking.ab_router \
  --variant-a '{"tier":"cheap"}' --variant-b '{"tier":"balanced"}' \
  --output results/ab_out.json
```

---

## Dev

```bash
uv sync --group dev        # install dev deps + pre-commit
uv run pre-commit install  # wire hooks into .git
make ci-test               # lint (ruff check --fix + format) + pytest
```
