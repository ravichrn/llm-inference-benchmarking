# llm-inference-benchmarking

**[📊 Interactive Benchmark Report](https://ravichrn.github.io/llm-inference-benchmarking/report.html)**

Cost-aware LLM routing gateway and benchmarking toolkit. Measures latency, cost, and quality tradeoffs across routing tiers (gateway benchmark) and quantization formats (Modal GPU benchmark). Includes production-realistic Poisson arrival load simulation, KV cache quantization analysis, MFU roofline visualization, and an adaptive ML routing classifier.

---

## Key Insights

**Gateway routing**
- `cheap` is **5.7× cheaper** and **2.1× faster** (mean) than `balanced` — strongly preferred for simple/short tasks
- `balanced` → `premium` cost gap is only ~22%; choose `premium` freely for complex reasoning
- `cheap` scales cleanly to c=20 (0% errors); `balanced`/`premium` hit OpenAI RPM caps at c=20 (22% / 50% error rates)
- `auto` tier uses a **learned logistic-regression classifier** trained on ledger history (falls back to keyword heuristics until ≥50 routing samples)

**GPU quantization — Llama-3.1-8B on A10G**
- **vLLM** is the single-GPU production default: 94% MMLU, 222 tok/s batch-8
- **GPTQ** is the fastest HF mode: 268 tok/s, lowest TTFT (31 ms), 68% less VRAM than fp16
- **NF4** is the VRAM-constrained pick: 7.8 GB vs 17.3 GB for fp16, comparable quality

**Scaling up — A100 & H100**
- **H100 fp8**: **8.3× faster than A10G vllm** (1,859 vs 222 tok/s batch-8) with **identical 94% MMLU** at $6.45/hr
- **H100 fp8 beats 2× A100-80GB tensor-parallel** (1,859 vs 1,108 tok/s) at lower cost ($6.45 vs $8.00/hr combined)
- **kv-fp8**: halves KV cache VRAM for long-context workloads; same quality, lower raw throughput than fp8

**Poisson load**
- Fixed-concurrency tests hide queuing — saturation knee only appears under open-loop Poisson arrivals
- Cheap tier saturates at λ≈2 rps; beyond that p99 grows 5–20× while p50 stays flat

---

## Architecture

The project has two independent components: a **cost-aware routing gateway** and a **GPU quantization benchmark** running on Modal cloud.

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
| `gptq` | HuggingFace | Fastest HF mode; Marlin INT4 kernels |
| `vllm` | vLLM | Production default; 94% MMLU |
| `kv-fp8` | vLLM | fp16 weights + fp8 KV cache; halves KV VRAM |
| `fp8` | vLLM | HW-native on H100; SW-emulated on A10G (broken) |
| `flash-attn` | HuggingFace | Gains at long sequences |
| `torch-compile` | HuggingFace | JIT fusion; slow first call |
| `tensor-parallel` | vLLM | 2× A100-80GB; best multi-GPU throughput |
| `continuous-batching` | vLLM async | Queue depth sweep |
| `kv-analysis` | vLLM | Context-length VRAM sweep (512→8k) |
| `cpu-q4km` | llama.cpp | Best CPU speed/quality balance |

**Decision guide:**

| Constraint | Mode | GPU |
|---|---|---|
| Max throughput | **fp8** | H100 |
| Max throughput + long context | **kv-fp8** | H100 |
| Price-performance | **vllm** | A10G |
| Lowest latency on A10G | **gptq** | A10G |
| Multi-GPU scale-out | tensor-parallel | 2× A100-80GB |
| VRAM ≤ 8 GB | nf4 | A10G |
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
