"""A/B testing framework for comparing two LLM routing variants.

Routes the same prompt set through two model/tier configurations, scores
both with an LLM judge, and reports win rate, latency delta, and cost delta.

Usage:
    uv run python -m llm_inference_benchmarking.ab_router \\
        --prompts data/eval_prompts.json \\
        --variant-a '{"tier":"cheap"}' \\
        --variant-b '{"tier":"balanced"}' \\
        --output results/ab_2026-05-18.json

    uv run python -m llm_inference_benchmarking.ab_router \\
        --prompts data/eval_prompts.json \\
        --variant-a '{"tier":"cheap"}' \\
        --variant-b '{"tier":"balanced"}' \\
        --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from llm_inference_benchmarking.eval import build_judge_prompt, parse_judge_response
from llm_inference_benchmarking.types import ABResult, ABVariantResult

load_dotenv()

_log = logging.getLogger(__name__)


class ABRouter:
    """Invoke two model variants on a prompt set and score with an LLM judge."""

    def __init__(self, judge_tier: str = "cheap"):
        from llm_inference_benchmarking.client import GatewayClient

        self._client = GatewayClient()
        self._judge_tier = judge_tier
        self._judge_model: str = ""

    def run(
        self,
        prompts: list[dict],
        variant_a: dict,
        variant_b: dict,
    ) -> ABResult:
        """Run A/B evaluation.

        Args:
            prompts: list of prompt dicts with keys id, task_type, prompt, reference.
            variant_a: dict with 'tier' and optional 'model' override for variant A.
            variant_b: same for variant B.

        Returns:
            ABResult with per-variant scores, win rate, latency, and cost.
        """

        tier_a = variant_a.get("tier", "cheap")
        tier_b = variant_b.get("tier", "balanced")
        model_a = variant_a.get("model", "")
        model_b = variant_b.get("model", "")

        scores_a: list[int] = [0] * len(prompts)
        scores_b: list[int] = [0] * len(prompts)
        latencies_a: list[float] = [0.0] * len(prompts)
        latencies_b: list[float] = [0.0] * len(prompts)
        costs_a: list[float] = [0.0] * len(prompts)
        costs_b: list[float] = [0.0] * len(prompts)

        total = len(prompts)

        def _run_one(idx: int, p: dict) -> None:
            nonlocal model_a, model_b
            task_type = p.get("task_type", "qa")
            prompt_text = p.get("prompt", "")
            reference = p.get("reference", "")
            prompt_id = p.get("id", "?")

            # Run A and B in parallel
            with ThreadPoolExecutor(max_workers=2) as inner:
                fut_a = inner.submit(self._invoke_variant, prompt_text, tier_a, model_a, prompt_id, "A")
                fut_b = inner.submit(self._invoke_variant, prompt_text, tier_b, model_b, prompt_id, "B")
                resp_a, lat_a, cost_a, ma = fut_a.result()
                resp_b, lat_b, cost_b, mb = fut_b.result()

            if ma:
                model_a = ma
            if mb:
                model_b = mb

            # Run judges in parallel too
            with ThreadPoolExecutor(max_workers=2) as inner:
                fut_ja = inner.submit(self._judge, task_type, reference, resp_a)
                fut_jb = inner.submit(self._judge, task_type, reference, resp_b)
                score_a = fut_ja.result()
                score_b = fut_jb.result()

            print(f"[{idx + 1}/{total}] {prompt_id} ({task_type})... A:{score_a} B:{score_b}", flush=True)

            scores_a[idx] = score_a
            scores_b[idx] = score_b
            latencies_a[idx] = lat_a
            latencies_b[idx] = lat_b
            costs_a[idx] = cost_a
            costs_b[idx] = cost_b

        # Run prompts concurrently (up to 5 at a time to avoid rate limits)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_run_one, i, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futures):
                fut.result()  # re-raise any exception

        n = len(prompts)
        wins_a = sum(1 for sa, sb in zip(scores_a, scores_b, strict=True) if sa > sb)

        return ABResult(
            variant_a=ABVariantResult(
                tier=tier_a,
                model=model_a,
                avg_score=round(sum(scores_a) / n, 2) if n else 0.0,
                avg_latency_ms=round(sum(latencies_a) / n, 1) if n else 0.0,
                total_cost_usd=round(sum(costs_a), 6),
            ),
            variant_b=ABVariantResult(
                tier=tier_b,
                model=model_b,
                avg_score=round(sum(scores_b) / n, 2) if n else 0.0,
                avg_latency_ms=round(sum(latencies_b) / n, 1) if n else 0.0,
                total_cost_usd=round(sum(costs_b), 6),
            ),
            win_rate_a=round(wins_a / n, 3) if n else 0.0,
            n_prompts=n,
            judge_model=self._judge_model,
        )

    def _invoke_variant(
        self, prompt_text: str, tier: str, model_override: str, prompt_id: str, label: str
    ) -> tuple[str, float, float, str]:
        from llm_inference_benchmarking.types import GatewayRequest

        if model_override:
            # model override is not yet supported; tier controls model selection via routing policy
            print(f"[WARN variant {label}] model override {model_override!r} ignored; tier={tier!r} controls selection")
        try:
            req = GatewayRequest(prompt=prompt_text, tier=tier, role="agent")
            res = self._client.invoke(req)
            return str(res.content), float(res.usage.latency_ms), res.usage.estimated_cost_usd, res.model
        except Exception as exc:
            print(f"[ERR variant {label}] {exc}")
            return "", 0.0, 0.0, model_override

    def _judge(self, task_type: str, reference: str, response: str) -> int:
        from llm_inference_benchmarking.types import GatewayRequest

        if not response:
            return 0
        judge_prompt = build_judge_prompt(task_type, reference, response)
        try:
            res = self._client.invoke(GatewayRequest(prompt=judge_prompt, tier=self._judge_tier, role="agent"))
            if not self._judge_model:
                self._judge_model = res.model
            score, _ = parse_judge_response(str(res.content))
            return score
        except Exception as exc:
            print(f"[ERR judge] {exc}")
            return 0


def print_ab_result(result: ABResult) -> None:
    a = result.variant_a
    b = result.variant_b
    winner = "A" if result.win_rate_a > 0.5 else ("B" if result.win_rate_a < 0.5 else "tie")
    print(f"\nA/B Result  n={result.n_prompts}  judge={result.judge_model}")
    print(f"  {'':25} {'Variant A':>12} {'Variant B':>12}")
    print(f"  {'tier':25} {a.tier:>12} {b.tier:>12}")
    print(f"  {'model':25} {(a.model or '-'):>12} {(b.model or '-'):>12}")
    print(f"  {'avg_score':25} {a.avg_score:>12.2f} {b.avg_score:>12.2f}")
    print(f"  {'avg_latency_ms':25} {a.avg_latency_ms:>12.1f} {b.avg_latency_ms:>12.1f}")
    print(f"  {'total_cost_usd':25} {a.total_cost_usd:>12.6f} {b.total_cost_usd:>12.6f}")
    print(f"  {'win_rate_A':25} {result.win_rate_a:>12.1%}")
    print(f"\n  Winner: Variant {winner}")


def run_ab(
    prompts_path: Path,
    variant_a: dict,
    variant_b: dict,
    output_path: Path | None,
    judge_tier: str = "cheap",
    dry_run: bool = False,
) -> ABResult | None:
    with prompts_path.open() as f:
        prompts: list[dict] = json.load(f)

    if dry_run:
        print(f"[dry-run] Would run A/B on {len(prompts)} prompts")
        print(f"  Variant A: {variant_a}")
        print(f"  Variant B: {variant_b}")
        print(f"  Judge tier: {judge_tier}")
        return None

    router = ABRouter(judge_tier=judge_tier)
    result = router.run(prompts, variant_a, variant_b)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(dataclasses.asdict(result), indent=2))
        print(f"Results written to {output_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test two LLM routing variants")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data" / "eval_prompts.json",
    )
    parser.add_argument(
        "--variant-a", type=json.loads, default={"tier": "cheap"}, help='JSON dict, e.g. {"tier":"cheap"}'
    )
    parser.add_argument(
        "--variant-b", type=json.loads, default={"tier": "balanced"}, help='JSON dict, e.g. {"tier":"balanced"}'
    )
    parser.add_argument("--judge-tier", default="cheap")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: results/ab_<timestamp>.json)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_path = args.output
    if output_path is None and not args.dry_run:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_path = Path("results") / f"ab_{ts}.json"

    result = run_ab(
        prompts_path=args.prompts,
        variant_a=args.variant_a,
        variant_b=args.variant_b,
        output_path=output_path,
        judge_tier=args.judge_tier,
        dry_run=args.dry_run,
    )

    if result:
        print_ab_result(result)


if __name__ == "__main__":
    main()
