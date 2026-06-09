# llm-inference-benchmarking

**[📊 Interactive Benchmark Report](https://ravichrn.github.io/llm-inference-benchmarking/report.html)**

A benchmarking and evaluation harness for LLM inference that measures cost, throughput, and quality tradeoffs for Llama-3.1-8B across 15 configurations spanning quantization formats, serving engines, and inference optimizations on A10G, H100, and dual A100-80GB GPUs. Includes GPU kernel profiling, an LLM-as-judge evaluation layer with regression detection, and a cost-aware routing gateway that selects the cheapest configuration meeting a target quality threshold.

---

## Key Insights

**Gateway routing**
- `cheap` is **5.7× cheaper** and **2.1× faster** (mean) than `balanced` — strongly preferred for simple/short tasks
- `balanced` → `premium` cost gap is only ~22%; choose `premium` freely for complex reasoning
- `cheap` scales cleanly to c=20 (0% errors); `balanced`/`premium` hit OpenAI RPM caps at c=20 (22% / 50% error rates)
- `auto` tier uses a **learned logistic-regression classifier** trained on ledger history (falls back to keyword heuristics until ≥50 routing samples)

**GPU quantization — Llama-3.1-8B on A10G**
- **vLLM** is the single-GPU production default: 94% MMLU, 222 tok/s batch-8
- **GPTQ** is the fastest HF mode: 267 tok/s batch-8, lowest TTFT (31 ms), 68% less VRAM than fp16, and most energy-efficient HF mode at **0.27 tok/joule**
- **NF4** is the VRAM-constrained pick: 7.8 GB vs 17.3 GB for fp16, comparable quality
- MMLU scores are measured on a **50-question CS/ML-domain subset** (not the full 57-subject benchmark); published 5-shot general MMLU for Llama-3.1-8B is ~66–68%

**Size vs. quantization — A10G (equal VRAM budget ~6–8 GB)**
- **3B-fp16** (6.2 GB) vs **8B-nf4** (7.8 GB): 3B is **94% faster** (46 vs 24 tok/s) and **78% more energy-efficient** (0.32 vs 0.18 tok/joule); 8B-nf4 has **15% higher quality** (74% vs 64% MMLU)
- **Decision rule**: prefer 3B-fp16 for latency/energy-constrained tasks; 8B-nf4 when quality matters and VRAM allows

**Energy efficiency — tokens per joule (A10G)**
| Mode | tok/s | VRAM | tok/joule |
|---|---|---|---|
| 3B-fp16 | 46 | 6.2 GB | **0.32** |
| gptq | 33 | 5.5 GB | 0.27 |
| nf4 | 26 | 7.8 GB | 0.19 |
| fp16 | 27 | 17.3 GB | 0.18 |
| 8B-nf4 | 24 | 7.8 GB | 0.18 |

**Scaling up — A100 & H100**
- **H100 fp8**: **8× faster than A10G vllm** (1,785 vs 222 tok/s batch-8) with **identical 94% MMLU (CS/ML subset)** at $6.45/hr
- **H100 fp8 beats 2× A100-80GB tensor-parallel** (1,785 vs 1,108 tok/s) at lower cost ($6.45 vs $8.00/hr combined)
- **kv-fp8**: halves KV cache VRAM for long-context workloads; same quality, lower raw throughput than fp8

**Poisson load**
- Fixed-concurrency tests hide queuing — saturation knee only appears under open-loop Poisson arrivals
- Cheap tier saturates at λ≈2 rps; beyond that p99 grows 5–20× while p50 stays flat

---

## Architecture

The project has two components: a **cost-aware routing gateway** and a **GPU quantization benchmark** running on Modal cloud. They are connected via the quality router: benchmark results (`results/modal_quant_*.json`) feed `quality_router.py`, which uses measured MMLU scores to select the cheapest tier meeting a configurable quality threshold at request time.

### Gateway

```
Client request
        │
        ▼
Rate limiter          ←─ token bucket / sliding window, 429 on breach
        ▼
RoutingPolicyEngine   ←─ forced → ML classifier → keyword heuristics
        ▼
Budget policy         ←─ daily hard/soft USD caps via ledger
        ▼
SLA check             ←─ p99 sliding window; breach → downgrade tier
        ▼
Quality router        ←─ cheapest model meeting MMLU threshold
        ▼
GatewayClient         ←─ OpenAI / Claude / Ollama / vLLM (LangChain)
        ├─→ SQLite ledger     (usage history, cost tracking)
        └─→ Prometheus        GET /metrics
```

**Routing tiers:**

| Tier | Default model | Use when |
|---|---|---|
| `cheap` | gpt-5.4-mini | Fast, simple tasks |
| `balanced` | gpt-5.4 | General-purpose |
| `premium` | gpt-5.5 | Complex reasoning |
| `auto` | adaptive | ML classifier → heuristic fallback |

**Endpoints:** `POST /generate` · `POST /ab` · `GET /health` · `GET /usage/summary` · `GET /metrics` · `GET /sla/status`

### GPU Benchmark (Modal)

Runs all quantization modes in parallel on a Modal GPU, writes `results/modal_quant_<gpu>.json`.

**Supported modes:**

| Mode | Engine | Notes |
|---|---|---|
| `fp16` | HuggingFace | Baseline reference |
| `int8` | HuggingFace | bitsandbytes 8-bit |
| `nf4` / `nf4-dq` | HuggingFace | Best speed/VRAM balance on A10G |
| `gptq` | HuggingFace | Fastest HF mode; Marlin INT4 kernels; best tok/joule among 8B modes |
| `vllm` | vLLM | Production default; 94% MMLU (CS/ML subset) |
| `kv-fp8` | vLLM | fp16 weights + fp8 KV cache; halves KV VRAM |
| `fp8` | vLLM | HW-native on H100; SW-emulated on A10G (broken) |
| `flash-attn` | HuggingFace | Gains at long sequences |
| `torch-compile` | HuggingFace | JIT fusion; slow first call |
| `tensor-parallel` | vLLM | 2× A100-80GB; best multi-GPU throughput |
| `continuous-batching` | vLLM async | Queue depth sweep |
| `kv-analysis` | vLLM | Context-length VRAM sweep (512→8k) |
| `cpu-q4km` | llama.cpp | Best CPU speed/quality balance |
| `size-sweep` | HuggingFace | Cross-size comparison: 3B-fp16, 8B-nf4, 8B-fp16 at equal VRAM |

**Decision guide:**

| Constraint | Mode | GPU |
|---|---|---|
| Max throughput | **fp8** | H100 |
| Max throughput + long context | **kv-fp8** | H100 |
| Price-performance | **vllm** | A10G |
| Lowest latency on A10G | **gptq** | A10G |
| Best energy efficiency (8B) | **gptq** | A10G |
| Best energy efficiency (any size) | **3B-fp16** via size-sweep | A10G |
| Multi-GPU scale-out | tensor-parallel | 2× A100-80GB |
| VRAM ≤ 8 GB, quality priority | nf4 | A10G |
| VRAM ≤ 8 GB, speed/energy priority | 3B-fp16 via size-sweep | A10G |
| CPU-only | cpu-q4km | CPU |

---

## Quickstart

```bash
uv sync --group dev
cp .env.example .env   # add API keys

uv run uvicorn llm_inference_benchmarking.gateway:app --host 0.0.0.0 --port 8010

curl http://localhost:8010/health
curl -X POST http://localhost:8010/generate \
  -H "Content-Type: application/json" -H "x-api-key: $GATEWAY_API_KEY" \
  -d '{"prompt": "Summarize RAG benefits", "tier": "auto"}'
```

---

## Configuration

Minimum `.env`:

```bash
GATEWAY_API_KEY=your-secret
OPENAI_API_KEY=sk-...
AGENT_LLM=openai          # openai | claude | vllm
```

See [.env.example](.env.example) for model overrides, vLLM config, rate limiting, SLA caps, and quality routing.

---

## Running Benchmarks

### Gateway

```bash
uv run llm-gateway-bench --iterations 3
uv run llm-gateway-bench --cache    # cold vs warm prefix caching latency
```

### Modal GPU benchmark

```bash
# All modes on A10G (default)
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py

# Specific modes; --merge adds to existing file without overwriting other modes
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --modes fp16,gptq,vllm --merge

# H100 (hardware fp8)
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --modes fp8,kv-fp8 --gpu H100

# KV cache context-length sweep
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --modes kv-analysis

# Size-vs-quant sweep (3B-fp16 vs 8B-nf4 vs 8B-fp16 at equal VRAM; reports tok/joule)
uv run modal run src/llm_inference_benchmarking/modal_benchmark.py \
  --modes size-sweep --gpu A10G
```

### Load tests (gateway must be running)

```bash
# Fixed concurrency
uv run llm-load-test --concurrency 1,5,10,20 --total 50 --tier cheap

# Poisson arrivals (reveals saturation knee)
uv run llm-poisson-test \
  --lambda-values 0.5,1.0,2.0,3.0,5.0 --duration 30 --tier cheap \
  --prompt-mix "short=0.6,medium=0.3,long=0.1"
```

### Report

```bash
uv run python report.py   # → docs/report.html
```

---

## Dev

```bash
uv sync --group dev
uv run pre-commit install
make ci-test               # lint + pytest

uv run pytest tests/ -v    # 137 tests, ~1s
uv run pytest tests/test_poisson_load.py tests/test_classifier_router.py -v
```
