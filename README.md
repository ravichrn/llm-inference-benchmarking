# llm-inference-benchmarking

Standalone **cost-aware LLM routing gateway**, **benchmark harness**, and optional **FastAPI inference service**.  
This repository is intentionally **separate** from application codebases (for example an arXiv research tracker): it does not need any project-specific data to run benchmarks.

## What it does

- Tiered routing (`cheap` / `balanced` / `premium`) with deterministic policy defaults.
- Per-call telemetry: latency, token usage (when providers expose it), estimated USD cost.
- SQLite usage ledger and Prometheus metrics.
- CLI benchmark snapshots with optional quantization metadata (Ollama / vLLM env hints).

## Install

```bash
cd llm-inference-benchmarking
uv sync
```

Copy `.env` from your provider keys (OpenAI, Anthropic, etc.) as needed.

## Benchmark

```bash
uv run llm-gateway-bench --iterations 1 --output results/snapshot.json
```

Or:

```bash
uv run python -m llm_inference_benchmarking.benchmark --iterations 1
```

## Standalone API

```bash
uv run uvicorn llm_inference_benchmarking.service_app:app --host 0.0.0.0 --port 8010
```

Docker:

```bash
docker compose up --build
```

## Ledger location

By default the SQLite ledger is:

`~/.llm_inference_benchmarking/gateway_usage.db`

Override with:

`GATEWAY_LEDGER_DB=/path/to/gateway_usage.db`

## Policy template

See `policy.example.yaml` for a future config-driven policy shape.
