"""Poisson arrival load tester for the LLM gateway.

Unlike the fixed-concurrency load test (load_test.py), this module simulates
production-realistic traffic where requests arrive as a Poisson process with
exponential inter-arrival times. Under a Poisson arrival model:

  - Requests fire at rate λ (arrivals/sec) regardless of in-flight state.
  - When λ exceeds the server's throughput capacity, requests queue up and
    tail latency (p99) blows up — the "saturation knee."
  - Little's Law: mean queue depth ≈ λ x mean_latency_s in steady state.

This directly models what production serving teams measure: the
throughput-vs-tail-latency tradeoff curve that determines the operating
point of a serving system.

Requires the gateway to be running:
  uv run uvicorn llm_inference_benchmarking.gateway:app --host 0.0.0.0 --port 8010

Usage:
  uv run llm-poisson-test --lambda-values 0.5,1.0,2.0,3.0,5.0 --duration 30 --tier cheap
  uv run llm-poisson-test --lambda-values 0.5,1,2,5 --duration 20 \\
      --prompt-mix "short=0.6,medium=0.3,long=0.1" --plot charts/poisson_curve.png
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_BASE_URL = "http://localhost:8010"

# ---------------------------------------------------------------------------
# Prompt corpus — varied lengths to stress different parts of the stack
# ---------------------------------------------------------------------------

_SHORT_PROMPTS: list[str] = [
    "What is gradient descent?",
    "Define overfitting in machine learning.",
    "What does RLHF stand for?",
    "Explain what a transformer is in one sentence.",
    "What is the difference between precision and recall?",
]

_MEDIUM_PROMPTS: list[str] = [
    (
        "You are a helpful assistant. A user is building a real-time recommendation system "
        "and wants to understand the trade-offs between collaborative filtering and content-based "
        "filtering. Summarize the key differences and when you would choose one over the other. "
        "Keep your answer to three paragraphs."
    ),
    (
        "Explain the architecture of a decoder-only transformer model. Include how attention "
        "heads work, what role the feed-forward network plays, and how autoregressive generation "
        "proceeds token by token. Aim for a thorough but concise explanation suitable for a "
        "software engineer new to machine learning."
    ),
    (
        "A software team is deciding between PostgreSQL and MongoDB for a new application that "
        "stores user profiles and activity logs. The team expects high read volume with complex "
        "queries but infrequent writes. Analyze the trade-offs and recommend one. Justify your "
        "recommendation in two to three paragraphs."
    ),
    (
        "What is retrieval-augmented generation (RAG)? Describe how it improves on standard "
        "language model completions, what the retrieval component does, and what failure modes "
        "you should watch for when deploying a RAG system in production."
    ),
    (
        "Explain the concept of model quantization in the context of large language model "
        "inference. Cover what quantization does to model weights, the accuracy vs speed "
        "trade-off, and the practical differences between INT8, NF4, and GPTQ quantization "
        "in two to three paragraphs."
    ),
]

_LONG_PROMPTS: list[str] = [
    (
        "Context document 1: The attention mechanism in transformer models computes queries, "
        "keys, and values from the input sequence. The attention score between position i and "
        "position j is computed as the dot product of the query at i and the key at j, scaled "
        "by the square root of the head dimension, then normalized via softmax. Multi-head "
        "attention runs this process in parallel across H heads with different learned projections.\n\n"
        "Context document 2: The KV cache in autoregressive generation stores the key and value "
        "tensors from all prior positions so that at each new token, only the new position needs "
        "to compute its query and attend over the cached keys and values. Without the KV cache, "
        "generation cost would scale quadratically with sequence length. With it, the cost of "
        "each decode step is linear in context length.\n\n"
        "Context document 3: FP8 KV cache quantization reduces the memory footprint of the KV "
        "cache by representing each key and value element in 8-bit floating point instead of 16-bit. "
        "This halves KV cache VRAM consumption, enabling longer context windows or larger batch "
        "sizes without additional memory. Quality impact is typically below 0.5 perplexity points "
        "for models above 7B parameters.\n\n"
        "Question: Based on the above documents, explain how FP8 KV cache quantization interacts "
        "with multi-head attention during autoregressive decoding. What operations touch the KV "
        "cache at each step, and how does quantization affect the numerical precision of those "
        "operations? What is the expected trade-off in terms of memory savings vs output quality?"
    ),
    (
        "System context: You are an expert in distributed systems and machine learning infrastructure. "
        "A company is scaling their LLM inference service from 100 to 10,000 requests per day. "
        "Their current setup uses a single GPU server with a vLLM instance serving Llama-3.1-8B. "
        "They are hitting rate limits and p99 latency is growing.\n\n"
        "Background data:\n"
        "- Current GPU: 1x A10G (24 GB VRAM, 600 GB/s memory bandwidth)\n"
        "- Current throughput: ~90 tokens/sec at batch size 1\n"
        "- Current p99 latency: 8 seconds at 50 req/min\n"
        "- Target p99 latency: under 3 seconds at 500 req/min\n"
        "- Budget constraint: under $5,000/month\n\n"
        "Available options to evaluate:\n"
        "1. Horizontal scaling: add more A10G instances behind a load balancer\n"
        "2. Upgrade to A100-80GB: 3x throughput, $3.20/hr vs $1.10/hr for A10G\n"
        "3. Use tensor parallelism across 2x A100s\n"
        "4. Switch to INT8 quantization to halve memory and increase throughput\n"
        "5. Use continuous batching (vLLM async engine) to improve GPU utilization\n\n"
        "Question: Analyze each option against the latency target and budget constraint. "
        "Which combination of options would you recommend, and why? Show your reasoning "
        "step by step, including rough cost and throughput estimates for your recommended approach."
    ),
    (
        "You are reviewing the following Python code for a production LLM inference service. "
        "Identify any correctness bugs, performance issues, security vulnerabilities, and "
        "architectural problems. For each issue found, explain the root cause and provide "
        "a corrected version of the relevant code.\n\n"
        "```python\n"
        "import os\n"
        "import sqlite3\n"
        "from fastapi import FastAPI, Request\n"
        "from langchain_openai import ChatOpenAI\n\n"
        "app = FastAPI()\n"
        "DB = sqlite3.connect('usage.db')\n"
        "DB.execute('CREATE TABLE IF NOT EXISTS usage (prompt TEXT, cost REAL)')\n\n"
        "@app.post('/generate')\n"
        "async def generate(request: Request):\n"
        "    body = await request.json()\n"
        "    prompt = body['prompt']\n"
        "    model = ChatOpenAI(model='gpt-4o', api_key=os.environ['OPENAI_KEY'])\n"
        "    result = model.invoke(prompt)\n"
        "    cost = len(result.content) * 0.00003\n"
        "    DB.execute(f\"INSERT INTO usage VALUES ('{prompt}', {cost})\")\n"
        "    DB.commit()\n"
        "    return {'content': result.content}\n"
        "```\n\n"
        "Please be thorough. List each issue with a severity rating (critical/high/medium/low), "
        "the specific line(s) involved, and the corrected code."
    ),
    (
        "You are an inference engineer at a large AI lab. Your team is comparing three serving "
        "strategies for a 70B parameter model:\n\n"
        "Strategy A — Tensor Parallelism across 4x H100s:\n"
        "  Peak throughput: 1,200 tokens/sec at batch 8\n"
        "  P99 latency (batch 1): 450ms\n"
        "  Cost: $12.80/hr (4x H100 at $3.20/hr)\n"
        "  MFU: 68%\n\n"
        "Strategy B — Pipeline Parallelism across 8x A100-80GB:\n"
        "  Peak throughput: 980 tokens/sec at batch 8\n"
        "  P99 latency (batch 1): 820ms (pipeline bubble overhead)\n"
        "  Cost: $14.40/hr (8x A100 at $1.80/hr)\n"
        "  MFU: 52%\n\n"
        "Strategy C — Speculative Decoding with draft model on 2x H100 + 4x A100:\n"
        "  Peak throughput: 1,500 tokens/sec at batch 8 (3x speedup from spec-dec)\n"
        "  P99 latency (batch 1): 290ms\n"
        "  Cost: $13.60/hr\n"
        "  MFU: 71% (target model), 45% (draft model)\n"
        "  Draft acceptance rate: 82%\n\n"
        "Your serving requirements:\n"
        "  - 95th percentile: p99 < 500ms for interactive users\n"
        "  - Peak load: 800 concurrent requests\n"
        "  - SLA: 99.9% of requests complete within 2 seconds\n"
        "  - Budget: $50,000/month\n\n"
        "Question: Which strategy would you recommend and why? Include analysis of: "
        "(1) whether each strategy meets the SLA at peak load, (2) cost at full utilization, "
        "(3) operational complexity trade-offs, and (4) how you would handle the draft model's "
        "variable acceptance rate in Strategy C to avoid tail latency spikes."
    ),
    (
        "Paper summary task: You are reading a research paper on continuous batching for LLM "
        "inference (similar to the Orca paper). The key idea is that instead of processing "
        "requests in fixed batches where all requests must complete before the next batch starts, "
        "the scheduler can insert new requests into the batch mid-generation as soon as a slot "
        "opens up (i.e., when a prior request finishes generating its last token).\n\n"
        "The paper reports the following results on A100 GPU with OPT-30B:\n"
        "  - Traditional batching (batch=16): 47.3 req/min, p99=12.4s\n"
        "  - Continuous batching (max_batch=16): 89.1 req/min, p99=4.2s\n"
        "  - Continuous batching (max_batch=32): 121.4 req/min, p99=6.8s\n\n"
        "The paper also shows that continuous batching improves GPU utilization from 58% to 87% "
        "because the GPU is never idle waiting for the slowest request in a batch to finish.\n\n"
        "Question 1: Explain in precise technical terms why continuous batching improves both "
        "throughput and tail latency simultaneously. Draw the connection to queuing theory.\n\n"
        "Question 2: Under what conditions would continuous batching NOT improve tail latency? "
        "Consider cases where request lengths are highly variable vs uniform.\n\n"
        "Question 3: How does the KV cache interact with continuous batching? What is the "
        "memory management challenge, and how do systems like PagedAttention (vLLM) solve it?"
    ),
]

_PROMPT_POOL: dict[str, list[str]] = {
    "short": _SHORT_PROMPTS,
    "medium": _MEDIUM_PROMPTS,
    "long": _LONG_PROMPTS,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ArrivalRecord:
    request_id: int
    prompt_class: str
    scheduled_at: float  # monotonic wall time the request should have fired
    fired_at: float = 0.0  # when asyncio actually dispatched
    completed_at: float = 0.0
    latency_ms: float = 0.0
    status_code: int = 0
    ok: bool = False
    error: str = ""


@dataclass
class LevelResult:
    lambda_rps: float
    duration_s: float
    prompt_mix: dict[str, float]
    total_fired: int
    successful: int
    failed: int
    achieved_rps: float  # successful / duration_s
    latency_ms: dict[str, float]  # mean, p50, p95, p99, min, max
    queue_depth_mean: float  # mean in-flight count (from event-timeline reconstruction)
    queue_depth_max: int
    scheduler_drift_ms: float  # mean(fired_at - scheduled_at) * 1000
    saturated: bool  # p99 > 3xp50 OR achieved_rps < 0.85xlambda_rps


# ---------------------------------------------------------------------------
# Core async simulation
# ---------------------------------------------------------------------------


async def _fire_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    prompt: str,
    prompt_class: str,
    tier: str,
    record: ArrivalRecord,
    in_flight: list[int],
) -> ArrivalRecord:
    """Issue one HTTP request, fill the ArrivalRecord, and return it."""
    record.fired_at = asyncio.get_event_loop().time()
    in_flight[0] += 1
    try:
        r = await client.post(
            url,
            json={"prompt": prompt, "tier": tier, "role": "agent"},
            headers=headers,
        )
        record.status_code = r.status_code
        record.ok = r.status_code == 200
    except Exception as exc:
        record.status_code = 0
        record.ok = False
        record.error = str(exc)
    finally:
        record.completed_at = asyncio.get_event_loop().time()
        record.latency_ms = (record.completed_at - record.fired_at) * 1000
        in_flight[0] -= 1
    return record


async def _run_poisson_level(
    lambda_rps: float,
    duration_s: float,
    warmup_s: float,
    tier: str,
    base_url: str,
    api_key: str,
    prompt_mix: dict[str, float],
    rng: random.SystemRandom,
) -> list[ArrivalRecord]:
    """
    Simulate a Poisson arrival process at rate λ for (warmup_s + duration_s) seconds.

    Requests are scheduled using exponential inter-arrival times and dispatched as
    independent asyncio Tasks. This creates real open-loop queuing: if the server
    is slower than λ, tasks pile up and tail latency grows.

    Returns only the measurement-window records (warmup excluded).
    """
    url = f"{base_url.rstrip('/')}/generate"
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    # Build prompt pool respecting the mix ratios
    prompt_pool: list[tuple[str, str]] = []
    for cls, weight in prompt_mix.items():
        prompts = _PROMPT_POOL.get(cls, [])
        if not prompts or weight <= 0:
            continue
        # Approximate ratio by repeating: use relative weights x 100 as integer counts
        count = max(1, round(weight * 100))
        prompt_pool.extend((p, cls) for p in prompts * ((count // len(prompts)) + 1))
    if not prompt_pool:
        prompt_pool = [(_SHORT_PROMPTS[0], "short")]

    total_duration = warmup_s + duration_s
    t_wall_start = asyncio.get_event_loop().time()
    in_flight: list[int] = [0]
    all_records: list[ArrivalRecord] = []
    tasks: list[asyncio.Task] = []
    request_id = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        t = 0.0
        while t < total_duration:
            inter_arrival = rng.expovariate(lambda_rps)
            t += inter_arrival
            if t >= total_duration:
                break

            fire_at_wall = t_wall_start + t
            prompt, cls = prompt_pool[request_id % len(prompt_pool)]
            record = ArrivalRecord(
                request_id=request_id,
                prompt_class=cls,
                scheduled_at=fire_at_wall,
            )
            all_records.append(record)
            request_id += 1

            async def _delayed(rec: ArrivalRecord, target_wall: float, p: str, c: str) -> ArrivalRecord:
                now = asyncio.get_event_loop().time()
                delay = max(0.0, target_wall - now)
                if delay > 0:
                    await asyncio.sleep(delay)
                return await _fire_request(client, url, headers, p, c, tier, rec, in_flight)

            tasks.append(asyncio.create_task(_delayed(record, fire_at_wall, prompt, cls)))

        # Wait for all requests to complete (or timeout)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # Return only measurement-window records (after warmup)
    warmup_end_wall = t_wall_start + warmup_s
    return [r for r in all_records if r.scheduled_at >= warmup_end_wall and r.completed_at > 0]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _pct(lst: list[float], p: float) -> float:
    if not lst:
        return 0.0
    idx = min(int(len(lst) * p / 100), len(lst) - 1)
    return round(lst[idx], 1)


def _reconstruct_queue_depth(records: list[ArrivalRecord]) -> tuple[float, int]:
    """
    Reconstruct mean and max in-flight count from completed ArrivalRecords.

    Algorithm: build a timeline of (time, +1 or -1) events representing
    request start and end. Walk the timeline, maintaining a running count.
    Weight each interval by its duration to get time-averaged queue depth.
    """
    if not records:
        return 0.0, 0

    events: list[tuple[float, int]] = []
    for r in records:
        if r.completed_at > 0 and r.completed_at > r.fired_at:
            events.append((r.fired_at, +1))
            events.append((r.completed_at, -1))

    if not events:
        return 0.0, 0

    events.sort(key=lambda e: e[0])
    t_min = events[0][0]
    t_max = events[-1][0]
    total_duration = t_max - t_min
    if total_duration <= 0:
        return 0.0, 0

    running = 0
    prev_t = t_min
    weighted_sum = 0.0
    peak = 0
    for t, delta in events:
        dt = t - prev_t
        weighted_sum += running * dt
        running += delta
        peak = max(peak, running)
        prev_t = t

    return round(weighted_sum / total_duration, 2), peak


def _compute_level_stats(
    records: list[ArrivalRecord],
    lambda_rps: float,
    duration_s: float,
    prompt_mix: dict[str, float],
) -> LevelResult:
    ok = [r for r in records if r.ok]
    failed = [r for r in records if not r.ok]
    latencies = sorted(r.latency_ms for r in ok)

    lat_dict: dict[str, float] = {
        "mean": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "p50": _pct(latencies, 50),
        "p95": _pct(latencies, 95),
        "p99": _pct(latencies, 99),
        "min": round(min(latencies), 1) if latencies else 0.0,
        "max": round(max(latencies), 1) if latencies else 0.0,
    }

    achieved_rps = round(len(ok) / duration_s, 3) if duration_s > 0 else 0.0

    drifts = [(r.fired_at - r.scheduled_at) * 1000 for r in records if r.fired_at > 0]
    scheduler_drift_ms = round(statistics.mean(drifts), 2) if drifts else 0.0

    queue_mean, queue_max = _reconstruct_queue_depth(records)

    p50 = lat_dict["p50"]
    p99 = lat_dict["p99"]
    saturated = (p99 > 3.0 * p50 and p50 > 0) or (achieved_rps < 0.85 * lambda_rps and len(ok) > 3)

    return LevelResult(
        lambda_rps=lambda_rps,
        duration_s=duration_s,
        prompt_mix=prompt_mix,
        total_fired=len(records),
        successful=len(ok),
        failed=len(failed),
        achieved_rps=achieved_rps,
        latency_ms=lat_dict,
        queue_depth_mean=queue_mean,
        queue_depth_max=queue_max,
        scheduler_drift_ms=scheduler_drift_ms,
        saturated=saturated,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_poisson_sweep(
    lambda_values: list[float],
    duration_s: float,
    warmup_s: float,
    tier: str,
    base_url: str,
    api_key: str,
    prompt_mix: dict[str, float],
) -> list[LevelResult]:
    """Run one Poisson level per λ value sequentially, return LevelResult list."""
    rng = random.SystemRandom()
    results: list[LevelResult] = []
    for lam in lambda_values:
        records = await _run_poisson_level(lam, duration_s, warmup_s, tier, base_url, api_key, prompt_mix, rng)
        result = _compute_level_stats(records, lam, duration_s, prompt_mix)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_sweep_table(levels: list[LevelResult]) -> None:
    header = (
        f"  {'λ (rps)':>8}  {'Achieved':>9}  {'P50 (ms)':>10}  {'P95 (ms)':>10}  "
        f"{'P99 (ms)':>10}  {'Queue':>7}  {'Saturated':>10}"
    )
    print("\n" + header)
    print("  " + "─" * (len(header) - 2))
    for lvl in levels:
        lat = lvl.latency_ms
        sat_flag = "YES ←" if lvl.saturated else "no"
        print(
            f"  {lvl.lambda_rps:>8.2f}  {lvl.achieved_rps:>9.2f}  "
            f"{lat['p50']:>10.0f}  {lat['p95']:>10.0f}  {lat['p99']:>10.0f}  "
            f"{lvl.queue_depth_mean:>7.1f}  {sat_flag:>10}"
        )
    print()


def _find_knee(levels: list[LevelResult]) -> LevelResult | None:
    """Return the first saturated level (the knee of the curve)."""
    for lvl in levels:
        if lvl.saturated:
            return lvl
    return None


def _plot_saturation_curve(levels: list[LevelResult], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting: pip install matplotlib", flush=True)
        return

    achieved = [lvl.achieved_rps for lvl in levels]
    p50s = [lvl.latency_ms["p50"] for lvl in levels]
    p99s = [lvl.latency_ms["p99"] for lvl in levels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(achieved, p50s, marker="o", color="tab:blue", label="P50 latency", linewidth=1.8)
    ax.plot(achieved, p99s, marker="s", color="tab:orange", label="P99 latency", linewidth=1.8)

    knee = _find_knee(levels)
    if knee is not None:
        ax.axvline(
            knee.achieved_rps,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Saturation knee (λ≈{knee.lambda_rps:.1f} rps)",
        )
        ax.annotate(
            f"Saturation\nλ≈{knee.lambda_rps:.1f} rps",
            (knee.achieved_rps, knee.latency_ms["p99"]),
            xytext=(15, -30),
            textcoords="offset points",
            fontsize=8,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )

    ax.set_xlabel("Achieved throughput (req/sec)", fontsize=11)
    ax.set_ylabel("Latency (ms) — log scale", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("Throughput vs Tail Latency: Saturation Curve (Poisson arrivals)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    note = (
        "Poisson arrival model: requests fire at rate λ regardless of in-flight state. "
        "P99 blows up at the saturation knee — the server's throughput ceiling."
    )
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=7, va="bottom", color="grey")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved saturation curve → {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_prompt_mix(raw: str) -> dict[str, float]:
    """Parse 'short=0.6,medium=0.3,long=0.1' into {short: 0.6, ...}."""
    mix: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if "=" in part:
            cls, val = part.split("=", 1)
            mix[cls.strip()] = float(val.strip())
        else:
            raise ValueError(f"Invalid prompt-mix format: {part!r}. Expected 'class=weight'.")
    # Normalize
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("prompt-mix weights must sum to a positive value")
    return {k: v / total for k, v in mix.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poisson arrival load test for the LLM gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run llm-poisson-test --lambda-values 0.5,1.0,2.0,5.0 --duration 30\n"
            "  uv run llm-poisson-test --lambda-values 0.5,1,2 --duration 20 \\\n"
            "      --prompt-mix short=0.5,medium=0.4,long=0.1 --plot charts/curve.png\n"
        ),
    )
    parser.add_argument(
        "--lambda-values",
        default="0.5,1.0,2.0,3.0,5.0",
        help="Comma-separated arrival rates in req/sec (default: 0.5,1.0,2.0,3.0,5.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Measurement window per λ level in seconds (default: 30)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=5.0,
        help="Warmup period per level to discard (default: 5)",
    )
    parser.add_argument(
        "--tier",
        default="cheap",
        choices=["cheap", "balanced", "premium", "auto"],
        help="Gateway tier to use (default: cheap)",
    )
    parser.add_argument(
        "--prompt-mix",
        default="short=0.6,medium=0.3,long=0.1",
        help="Comma-separated class=weight pairs (default: short=0.6,medium=0.3,long=0.1)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("GATEWAY_BASE_URL", _DEFAULT_BASE_URL),
        help="Gateway base URL (default: http://localhost:8010)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON results",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional path to write the saturation curve PNG",
    )
    args = parser.parse_args()

    api_key = os.getenv("GATEWAY_API_KEY", "")
    if not api_key:
        parser.error("GATEWAY_API_KEY env var is not set — required for authentication")

    lambda_values = [float(v.strip()) for v in args.lambda_values.split(",") if v.strip()]
    if not lambda_values:
        parser.error("--lambda-values must contain at least one value")

    try:
        prompt_mix = _parse_prompt_mix(args.prompt_mix)
    except ValueError as exc:
        parser.error(str(exc))

    mix_desc = ", ".join(f"{k}={v:.0%}" for k, v in sorted(prompt_mix.items()))
    print(f"Poisson load test → {args.base_url}  tier={args.tier}")
    print(f"Prompt mix: {mix_desc}")
    print(
        f"Lambda sweep: {lambda_values} rps  ({args.duration:.0f}s measurement + {args.warmup:.0f}s warmup per level)\n"
    )

    rng = random.SystemRandom()
    all_levels: list[LevelResult] = []
    for lam in lambda_values:
        print(f"  λ={lam:.2f} rps  running …", end="", flush=True)
        t0 = time.perf_counter()
        records = asyncio.run(
            _run_poisson_level(lam, args.duration, args.warmup, args.tier, args.base_url, api_key, prompt_mix, rng)
        )
        result = _compute_level_stats(records, lam, args.duration, prompt_mix)
        all_levels.append(result)
        elapsed = time.perf_counter() - t0
        lat = result.latency_ms
        sat_flag = "  SATURATED" if result.saturated else ""
        print(
            f"  done ({elapsed:.0f}s)  achieved={result.achieved_rps:.2f}  "
            f"p50={lat['p50']:.0f}ms  p99={lat['p99']:.0f}ms  "
            f"queue={result.queue_depth_mean:.1f}{sat_flag}"
        )

    knee = _find_knee(all_levels)
    if knee:
        print(f"\nSaturation knee at λ≈{knee.lambda_rps:.2f} rps (achieved throughput ≈ {knee.achieved_rps:.2f} rps)")
    else:
        print("\nNo saturation detected — server handled all tested arrival rates.")

    _print_sweep_table(all_levels)

    if args.plot:
        _plot_saturation_curve(all_levels, Path(args.plot))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_url": args.base_url,
            "tier": args.tier,
            "prompt_mix": prompt_mix,
            "duration_s": args.duration,
            "warmup_s": args.warmup,
            "levels": [
                {
                    "lambda_rps": lvl.lambda_rps,
                    "achieved_rps": lvl.achieved_rps,
                    "total_fired": lvl.total_fired,
                    "successful": lvl.successful,
                    "failed": lvl.failed,
                    "latency_ms": lvl.latency_ms,
                    "queue_depth_mean": lvl.queue_depth_mean,
                    "queue_depth_max": lvl.queue_depth_max,
                    "scheduler_drift_ms": lvl.scheduler_drift_ms,
                    "saturated": lvl.saturated,
                }
                for lvl in all_levels
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
