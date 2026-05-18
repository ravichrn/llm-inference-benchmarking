"""LLM evaluation harness with LLM-as-judge scoring and regression detection.

Usage:
    uv run python -m llm_inference_benchmarking.eval --tier cheap
    uv run python -m llm_inference_benchmarking.eval --tier balanced --compare results/eval_prev.json
    uv run python -m llm_inference_benchmarking.eval --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llm_inference_benchmarking.client import GatewayClient
from llm_inference_benchmarking.types import GatewayRequest

load_dotenv()

_log = logging.getLogger(__name__)

_JUDGE_TMPL = """\
You are an impartial evaluator. Score the response 0-10 on correctness, completeness, and conciseness.
Reply with JSON only, no markdown fences: {{"score": <int 0-10>, "reason": "<one sentence>"}}

Task type: {task_type}
Reference answer: {reference}
Model response: {response}"""

_REGRESSION_THRESHOLD = 0.5


def build_judge_prompt(task_type: str, reference: str, response: str) -> str:
    return _JUDGE_TMPL.format(task_type=task_type, reference=reference, response=response)


def parse_judge_response(raw: str) -> tuple[int, str]:
    """Extract score and reason from judge JSON. Returns (0, 'parse error') on failure."""
    try:
        text = raw.strip()
        # Strip markdown fences if the model ignored the instruction
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        data = json.loads(text)
        score = max(0, min(10, int(data["score"])))
        reason = str(data.get("reason", ""))
        return score, reason
    except Exception:
        return 0, "parse error"


def _load_prompts(prompts_path: Path) -> list[dict]:
    with prompts_path.open() as f:
        return json.load(f)


def _find_latest_eval(results_dir: Path) -> Path | None:
    candidates = sorted(results_dir.glob("eval_*.json"), reverse=True)
    return candidates[0] if candidates else None


def run_eval(
    tier: str,
    prompts_path: Path,
    results_dir: Path,
    judge_tier: str = "cheap",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the eval harness. Returns the full result dict."""
    prompts = _load_prompts(prompts_path)
    client = GatewayClient()
    run_id = f"eval_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"

    if dry_run:
        print(f"[dry-run] Would evaluate {len(prompts)} prompts on tier={tier!r} with judge tier={judge_tier!r}")
        for p in prompts[:3]:
            print(f"  {p['id']} ({p['task_type']}): {p['prompt'][:60]}...")
        print("  ...")
        return {}

    results: list[dict] = [{} for _ in prompts]
    model_used = ""
    total = len(prompts)

    def _eval_one(idx: int, p: dict) -> None:
        nonlocal model_used
        prompt_id = p["id"]
        task_type = p["task_type"]
        prompt_text = p["prompt"]
        reference = p.get("reference", "")

        try:
            res = client.invoke(GatewayRequest(prompt=prompt_text, tier=tier, role="agent"))
            response_text = str(res.content)
            latency_ms = res.usage.latency_ms
            cost = res.usage.estimated_cost_usd
            if not model_used:
                model_used = res.model
        except Exception as exc:
            print(f"[{idx}/{total}] {prompt_id} ({task_type})... error", flush=True)
            results[idx] = {
                "id": prompt_id,
                "task_type": task_type,
                "score": 0,
                "latency_ms": 0,
                "estimated_cost_usd": 0.0,
                "judge_reason": f"model error: {exc}",
            }
            return

        judge_prompt = build_judge_prompt(task_type, reference, response_text)
        try:
            judge_res = client.invoke(GatewayRequest(prompt=judge_prompt, tier=judge_tier, role="agent"))
            score, reason = parse_judge_response(str(judge_res.content))
            cost += judge_res.usage.estimated_cost_usd
        except Exception as exc:
            score, reason = 0, f"judge error: {exc}"

        print(f"[{idx}/{total}] {prompt_id} ({task_type})... score:{score}", flush=True)
        results[idx] = {
            "id": prompt_id,
            "task_type": task_type,
            "score": score,
            "latency_ms": latency_ms,
            "estimated_cost_usd": round(cost, 8),
            "judge_reason": reason,
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_eval_one, i, p): i for i, p in enumerate(prompts)}
        for fut in as_completed(futures):
            fut.result()

    # Compute summary
    by_type: dict[str, list[int]] = {}
    results = [r for r in results if r]  # drop any unfilled slots
    for r in results:
        by_type.setdefault(r["task_type"], []).append(r["score"])

    avg_score = round(sum(r["score"] for r in results) / len(results), 2) if results else 0.0
    avg_latency = round(sum(r["latency_ms"] for r in results) / len(results), 0) if results else 0.0
    total_cost = round(sum(r["estimated_cost_usd"] for r in results), 6)

    summary = {
        "avg_score": avg_score,
        "avg_latency_ms": avg_latency,
        "total_cost_usd": total_cost,
        "by_task_type": {tt: round(sum(sc) / len(sc), 2) for tt, sc in by_type.items()},
    }

    output = {
        "run_id": run_id,
        "tier": tier,
        "model": model_used,
        "results": results,
        "summary": summary,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Results written to {out_path}")

    return output


def print_summary(result: dict[str, Any]) -> None:
    s = result.get("summary", {})
    print(f"\nEval run: {result.get('run_id')}  tier={result.get('tier')}  model={result.get('model')}")
    print(f"  avg_score:    {s.get('avg_score')}/10")
    print(f"  avg_latency:  {s.get('avg_latency_ms')} ms")
    print(f"  total_cost:   ${s.get('total_cost_usd'):.6f}")
    print("\n  By task type:")
    for tt, sc in sorted((s.get("by_task_type") or {}).items()):
        print(f"    {tt:<25} {sc}/10")


def print_regression_report(current: dict[str, Any], prior: dict[str, Any]) -> None:
    cur = current.get("summary", {}).get("by_task_type", {})
    prv = prior.get("summary", {}).get("by_task_type", {})
    print(f"\n  Regression vs {prior.get('run_id')} (tier={prior.get('tier')}):")
    any_regression = False
    for tt in sorted(set(cur) | set(prv)):
        c = cur.get(tt)
        p = prv.get(tt)
        if c is None or p is None:
            continue
        delta = c - p
        flag = ""
        if delta < -_REGRESSION_THRESHOLD:
            flag = "  ⚠ REGRESSION"
            any_regression = True
        print(f"    {tt:<25} {p} → {c}  ({delta:+.2f}){flag}")
    if not any_regression:
        print("    No regressions detected.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM eval harness")
    parser.add_argument("--tier", default="cheap", help="Gateway tier to evaluate (cheap/balanced/premium)")
    parser.add_argument("--judge-tier", default="cheap", help="Tier used for the LLM judge")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data" / "eval_prompts.json",
        help="Path to eval_prompts.json",
    )
    parser.add_argument("--output", type=Path, default=Path("results"), help="Directory to write result JSON")
    parser.add_argument("--compare", type=Path, default=None, help="Path to prior eval JSON for regression comparison")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without making API calls")
    args = parser.parse_args()

    result = run_eval(
        tier=args.tier,
        prompts_path=args.prompts,
        results_dir=args.output,
        judge_tier=args.judge_tier,
        dry_run=args.dry_run,
    )

    if args.dry_run or not result:
        return

    print_summary(result)

    compare_path = args.compare
    if compare_path is None:
        latest = _find_latest_eval(args.output)
        if latest and latest.stem != result.get("run_id"):
            compare_path = latest

    if compare_path and compare_path.exists():
        prior = json.loads(compare_path.read_text())
        print_regression_report(result, prior)


if __name__ == "__main__":
    main()
