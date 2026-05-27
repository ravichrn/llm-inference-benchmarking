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
import os
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


def _regression_threshold() -> float:
    """Read regression threshold from env, defaulting to 0.5."""
    return float(os.getenv("EVAL_REGRESSION_THRESHOLD", "0.5"))


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


# ---------------------------------------------------------------------------
# Deterministic scoring helpers
# ---------------------------------------------------------------------------

_CHOICE_LETTERS = {"A", "B", "C", "D"}


def score_exact_match(response: str, choices: list[str], correct: str) -> float:
    """Return 1.0 if the response selects the correct MMLU choice letter, else 0.0.

    Checks for a bare letter (A/B/C/D) in the first non-whitespace characters
    of the response, then falls back to checking the full text.
    """
    text = response.strip().upper()
    first_token = text.split()[0] if text.split() else ""
    # Strip trailing punctuation from first token
    first_token = first_token.rstrip(".),:;")
    if first_token in _CHOICE_LETTERS:
        return 1.0 if first_token == correct.upper() else 0.0
    # Fallback: check if correct letter appears as a standalone word
    import re as _re

    if _re.search(rf"\b{correct.upper()}\b", text):
        return 1.0
    return 0.0


def score_bertscore(response: str, reference: str) -> float | None:
    """BERTScore F1 against reference. Returns None if bert-score is not installed."""
    try:
        from bert_score import score as _bs_score  # type: ignore[import]
    except ImportError:
        _log.warning("bert-score not installed; skipping BERTScore gate signal")
        return None
    try:
        model = os.getenv("EVAL_BERTSCORE_MODEL", "distilbert-base-uncased")
        _P, _R, F = _bs_score([response], [reference], model_type=model, verbose=False)
        return float(F[0])
    except Exception as exc:
        _log.warning("BERTScore failed: %s", exc)
        return None


# Default gate thresholds: judge score >= 6.0 per task type.
# Override via EVAL_GATE_THRESHOLD env var (applies to all task types).
def _gate_thresholds() -> dict[str, float]:
    default = float(os.getenv("EVAL_GATE_THRESHOLD", "6.0"))
    return {"__default__": default}


def _check_gate(summary: dict[str, Any], thresholds: dict[str, float]) -> tuple[bool, list[str]]:
    """Return (passed, failure_reasons) by comparing task-type scores against thresholds."""
    by_task = summary.get("by_task_type", {})
    default_thresh = thresholds.get("__default__", 6.0)
    failures: list[str] = []
    for task_type, score in by_task.items():
        thresh = thresholds.get(task_type, default_thresh)
        if score < thresh:
            failures.append(f"{task_type}: {score:.2f} < {thresh:.2f}")
    return (len(failures) == 0, failures)


def _print_gate_verdict(passed: bool, failures: list[str]) -> None:
    if passed:
        print("\n  ✓ Gate PASSED — all task-type scores meet threshold.")
    else:
        print("\n  ✗ Gate FAILED — the following task types are below threshold:")
        for f in failures:
            print(f"      {f}")


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
    gate: bool = False,
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

    gate_passed = True
    gate_failures: list[str] = []
    if gate:
        gate_passed, gate_failures = _check_gate(summary, _gate_thresholds())

    output = {
        "run_id": run_id,
        "tier": tier,
        "model": model_used,
        "results": results,
        "summary": summary,
        "gate_passed": gate_passed,
        "gate_failures": gate_failures,
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
        if delta < -_regression_threshold():
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
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Enable release gate: exit 1 if any task type is below threshold",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=None,
        help="Minimum judge score per task type for gate (default: from EVAL_GATE_THRESHOLD env, else 6.0)",
    )
    args = parser.parse_args()

    if args.gate_threshold is not None:
        os.environ["EVAL_GATE_THRESHOLD"] = str(args.gate_threshold)

    result = run_eval(
        tier=args.tier,
        prompts_path=args.prompts,
        results_dir=args.output,
        judge_tier=args.judge_tier,
        dry_run=args.dry_run,
        gate=args.gate,
    )

    if args.dry_run or not result:
        return

    print_summary(result)

    if args.gate:
        _print_gate_verdict(result.get("gate_passed", True), result.get("gate_failures", []))
        if not result.get("gate_passed", True):
            raise SystemExit(1)

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
