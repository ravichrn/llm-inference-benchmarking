"""Concurrent load tester for the LLM gateway.

Fires N parallel requests against the gateway /generate endpoint and reports
latency percentiles, throughput (req/s), and error rate at each concurrency level.

Requires the gateway to be running:
  uv run uvicorn llm_inference_benchmarking.gateway:app --host 0.0.0.0 --port 8010

Usage:
  uv run llm-load-test --concurrency 1,5,10,20 --total 50 --tier cheap
  uv run llm-load-test --concurrency 10 --total 100 --output results/load_test.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_BASE_URL = "http://localhost:8010"
_DEFAULT_PROMPTS = [
    "Summarize the key benefits of retrieval-augmented generation in two sentences.",
    "What is the difference between precision and recall?",
    "Explain gradient descent in one paragraph.",
]


async def _worker(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json=payload, headers=headers)
            latency_ms = (time.perf_counter() - t0) * 1000
            return {"latency_ms": latency_ms, "status": r.status_code, "ok": r.status_code == 200}
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            return {"latency_ms": latency_ms, "status": 0, "ok": False, "error": str(exc)}


async def _run_level(
    concurrency: int,
    total: int,
    tier: str,
    base_url: str,
    api_key: str,
) -> dict:
    url = f"{base_url.rstrip('/')}/generate"
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [
            _worker(
                client,
                url,
                headers,
                {
                    "prompt": _DEFAULT_PROMPTS[i % len(_DEFAULT_PROMPTS)],
                    "tier": tier,
                    "role": "agent",
                },
                semaphore,
            )
            for i in range(total)
        ]
        wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_elapsed = time.perf_counter() - wall_start

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    latencies = sorted(r["latency_ms"] for r in ok)

    def pct(lst: list[float], p: float) -> float:
        if not lst:
            return 0.0
        idx = min(int(len(lst) * p / 100), len(lst) - 1)
        return round(lst[idx], 1)

    return {
        "concurrency": concurrency,
        "total_requests": total,
        "successful": len(ok),
        "failed": len(failed),
        "error_rate_pct": round(len(failed) / total * 100, 1),
        "wall_time_s": round(wall_elapsed, 2),
        "req_per_sec": round(total / wall_elapsed, 2),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1) if latencies else 0,
            "p50": pct(latencies, 50),
            "p95": pct(latencies, 95),
            "p99": pct(latencies, 99),
            "min": round(min(latencies), 1) if latencies else 0,
            "max": round(max(latencies), 1) if latencies else 0,
        },
    }


def _print_table(levels: list[dict]) -> None:
    header = (
        f"{'Concurrency':>12}  {'Req/s':>8}  {'P50 (ms)':>10}"
        f"  {'P95 (ms)':>10}  {'P99 (ms)':>10}  {'Errors':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for lvl in levels:
        lat = lvl["latency_ms"]
        print(
            f"{lvl['concurrency']:>12}  {lvl['req_per_sec']:>8.2f}  "
            f"{lat['p50']:>10.1f}  {lat['p95']:>10.1f}  {lat['p99']:>10.1f}  "
            f"{lvl['error_rate_pct']:>6.1f}%"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent load test for the LLM gateway")
    parser.add_argument(
        "--concurrency",
        default="1,5,10,20",
        help="Comma-separated concurrency levels to test (default: 1,5,10,20)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=50,
        help="Total requests per concurrency level (default: 50)",
    )
    parser.add_argument(
        "--tier",
        default="cheap",
        choices=["cheap", "balanced", "premium", "auto"],
        help="Gateway tier to use (default: cheap)",
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
    args = parser.parse_args()

    api_key = os.getenv("GATEWAY_API_KEY", "")
    if not api_key:
        parser.error("GATEWAY_API_KEY env var is not set — required for authentication")

    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",") if c.strip()]

    print(f"Load test → {args.base_url}  tier={args.tier}  total={args.total} req/level")
    print(f"Concurrency levels: {concurrency_levels}\n")

    all_levels: list[dict] = []
    for c in concurrency_levels:
        print(f"  concurrency={c} …", end="", flush=True)
        result = asyncio.run(_run_level(c, args.total, args.tier, args.base_url, api_key))
        all_levels.append(result)
        lat = result["latency_ms"]
        print(
            f"  {result['req_per_sec']:.1f} req/s  "
            f"p50={lat['p50']:.0f}ms  p95={lat['p95']:.0f}ms  "
            f"errors={result['error_rate_pct']:.0f}%"
        )

    _print_table(all_levels)

    output = {
        "base_url": args.base_url,
        "tier": args.tier,
        "total_per_level": args.total,
        "levels": all_levels,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
