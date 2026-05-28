"""Live gateway demo — sends varied requests and prints routing decisions.

Shows how the policy engine selects tier + backend based on prompt content,
role, and explicit tier override. Also demonstrates the autoscaler signal.

Usage:
    # Against local dev server (uvicorn):
    uvicorn src.llm_inference_benchmarking.gateway:app --port 8000
    python demo_gateway.py

    # Against deployed Modal endpoint:
    python demo_gateway.py --url https://<your-modal-app>.modal.run

    # Skip live LLM calls (routing logic only, no API key needed):
    python demo_gateway.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Demo requests — varied prompts that exercise different routing paths
# ---------------------------------------------------------------------------

_REQUESTS = [
    {
        "label": "simple fact (cheap)",
        "prompt": "What is the capital of France?",
        "tier": "auto",
        "role": "fast",
    },
    {
        "label": "classification (cheap keyword)",
        "prompt": "Classify this as positive or negative: 'The model latency was terrible.'",
        "tier": "auto",
        "role": "agent",
    },
    {
        "label": "deep analysis (premium keyword)",
        "prompt": "Summarize the trade-offs between GPTQ INT4 and AWQ INT4 quantization for LLM serving.",
        "tier": "auto",
        "role": "agent",
    },
    {
        "label": "explicit cheap tier",
        "prompt": "List three benefits of quantization in one sentence each.",
        "tier": "cheap",
        "role": "agent",
    },
    {
        "label": "explicit premium tier",
        "prompt": "Compare vLLM PagedAttention and continuous batching for multi-tenant LLM serving.",
        "tier": "premium",
        "role": "agent",
    },
    {
        "label": "long prompt (balanced by length)",
        "prompt": "Explain transformer attention. " * 100,
        "tier": "auto",
        "role": "agent",
    },
]

# ---------------------------------------------------------------------------
# Autoscaler demo inputs
# ---------------------------------------------------------------------------

_AUTOSCALER_SCENARIOS = [
    {
        "label": "underloaded (scale up)",
        "p99_latency_ms": 50,
        "output_tps": 200,
        "batch_size": 1,
        "utilization": 0.05,
        "gpu_cost_per_hr": 1.10,
    },
    {
        "label": "nominal (hold)",
        "p99_latency_ms": 3000,
        "output_tps": 40,
        "batch_size": 4,
        "utilization": 0.55,
        "gpu_cost_per_hr": 1.10,
    },
    {
        "label": "overloaded (scale down)",
        "p99_latency_ms": 4500,
        "output_tps": 12,
        "batch_size": 8,
        "utilization": 0.20,
        "gpu_cost_per_hr": 1.10,
    },
]

# ---------------------------------------------------------------------------
# Routing-only dry run (no API key, no LLM calls)
# ---------------------------------------------------------------------------


def _dry_run() -> None:
    from llm_inference_benchmarking.autoscaler import autoscaler_signal
    from llm_inference_benchmarking.policy import RoutingPolicyEngine
    from llm_inference_benchmarking.types import GatewayRequest

    policy = RoutingPolicyEngine()
    print("\n=== Routing decisions (dry run — no LLM calls) ===\n")
    print(f"{'Request':<35} {'Tier':<10} {'Backend':<10} {'Model':<30} {'Reason'}")
    print("-" * 105)
    for r in _REQUESTS:
        req = GatewayRequest(prompt=r["prompt"], tier=r["tier"], role=r["role"])
        decision = policy.decide(req)
        model_short = decision.model.split("/")[-1][:28]
        print(f"{r['label']:<35} {decision.tier:<10} {decision.backend:<10} {model_short:<30} {decision.reason}")

    print("\n=== Autoscaler signal ===\n")
    print(f"{'Scenario':<30} {'Direction':<10} {'Score':<8} {'Batch':<8} Reason")
    print("-" * 100)
    for s in _AUTOSCALER_SCENARIOS:
        sig = autoscaler_signal(s)
        reason_short = sig["reason"].split("—")[0].strip()
        batch = sig["recommended_batch_size"]
        print(f"{s['label']:<30} {sig['scale_direction']:<10} {sig['score']:<8.3f} {batch:<8} {reason_short}")

    print()


# ---------------------------------------------------------------------------
# Live run against a real gateway endpoint
# ---------------------------------------------------------------------------


def _live_run(base_url: str, api_key: str) -> None:
    try:
        import httpx
    except ImportError:
        print("httpx required for live run: pip install httpx")
        sys.exit(1)

    headers = {"X-API-Key": api_key} if api_key else {}
    client = httpx.Client(base_url=base_url, headers=headers, timeout=60)

    # Health check
    try:
        r = client.get("/health")
        r.raise_for_status()
        print(f"\nGateway healthy at {base_url}\n")
    except Exception as exc:
        print(f"Gateway not reachable at {base_url}: {exc}")
        sys.exit(1)

    print("=== Live routing decisions ===\n")
    print(f"{'Request':<35} {'Tier':<10} {'Backend':<10} {'Model':<28} {'Latency':>8}  {'Cost':>10}  Preview")
    print("-" * 120)

    for r in _REQUESTS:
        t0 = time.perf_counter()
        try:
            resp = client.post("/generate", json={"prompt": r["prompt"], "tier": r["tier"], "role": r["role"]})
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            if resp.status_code != 200:
                print(f"{r['label']:<35} ERROR {resp.status_code}: {resp.text[:60]}")
                continue
            data = resp.json()
            usage = data.get("usage", {})
            model_short = data.get("model", "")[-28:]
            cost_str = f"${usage.get('estimated_cost_usd', 0):.5f}"
            preview = str(data.get("content", ""))[:40].replace("\n", " ")
            lat_str = f"{usage.get('latency_ms', elapsed_ms)}ms"
            tier = data.get("tier", "")
            backend = data.get("backend", "")
            row = f"{r['label']:<35} {tier:<10} {backend:<10} {model_short:<28}"
            print(f"{row} {lat_str:>8}  {cost_str:>10}  {preview}…")
        except Exception as exc:
            print(f"{r['label']:<35} FAILED: {exc}")

    # Usage summary
    print("\n=== Usage summary ===\n")
    try:
        r = client.get("/usage/summary")
        rows = r.json().get("summary", [])
        if rows:
            print(f"{'Date':<12} {'Tier':<10} {'Model':<30} {'Requests':>9} {'Tokens':>8} {'Cost':>10} {'Avg lat':>9}")
            print("-" * 92)
            for row in rows:
                print(
                    f"{row.get('date', ''):<12} {row.get('tier', ''):<10} {str(row.get('model', ''))[-28:]:<30} "
                    f"{row.get('requests', 0):>9} {row.get('output_tokens', 0):>8} "
                    f"${row.get('total_cost_usd', 0):.5f}  {row.get('avg_latency_ms', 0):>7.0f}ms"
                )
        else:
            print("No usage data yet.")
    except Exception as exc:
        print(f"Could not fetch usage summary: {exc}")

    # SLA status
    print("\n=== SLA status ===\n")
    try:
        r = client.get("/sla/status")
        for tier, info in r.json().get("sla", {}).items():
            p99 = info.get("p99_ms")
            cap = info.get("cap_ms")
            breached = info.get("breached", False)
            status = "BREACHED" if breached else "ok"
            p99_str = f"{p99:.0f}ms" if p99 is not None else "n/a"
            cap_str = f"{cap:.0f}ms" if cap is not None else "no cap"
            print(f"  {tier:<10} p99={p99_str:<10} cap={cap_str:<10} {status}")
    except Exception as exc:
        print(f"Could not fetch SLA status: {exc}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Gateway routing demo")
    parser.add_argument("--url", default="http://localhost:8000", help="Gateway base URL")
    parser.add_argument("--api-key", default=os.getenv("GATEWAY_API_KEY", ""), help="Gateway API key")
    parser.add_argument("--dry-run", action="store_true", help="Routing logic only, no LLM calls")
    args = parser.parse_args()

    if args.dry_run:
        _dry_run()
    else:
        _live_run(args.url, args.api_key)


if __name__ == "__main__":
    main()
