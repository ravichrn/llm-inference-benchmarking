"""Cross-model cost/quality Pareto chart generator.

Reads modal_benchmark JSON result files and plots:
  X-axis: estimated cost per 1k output tokens (USD)
  Y-axis: MMLU accuracy (quality proxy)

Usage:
  uv run python -m llm_inference_benchmarking.pareto --results results/ --output pareto.png

Requires: matplotlib (optional dep — only needed for this script)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_points(results_dir: Path) -> list[dict]:
    """Return list of {label, cost_per_1k, accuracy, mode, gpu} dicts."""
    points = []
    for fpath in sorted(results_dir.glob("modal_quant*.json")):
        try:
            data = json.loads(fpath.read_text())
        except Exception as exc:
            print(f"Skipping {fpath.name}: {exc}", file=sys.stderr)
            continue
        # Top-level is a dict with a "results" list and "gpu_cost_per_hr_usd"
        if isinstance(data, dict):
            gpu_cost_hr = data.get("gpu_cost_per_hr_usd", 1.10)
            entries = data.get("results", [])
        elif isinstance(data, list):
            gpu_cost_hr = 1.10  # fallback
            entries = data
        else:
            continue
        for entry in entries:
            mode = entry.get("quant_mode") or entry.get("mode", "")
            gpu = entry.get("gpu", "")
            q = entry.get("quality", {})
            if not isinstance(q, dict):
                continue
            accuracy = q.get("mmlu_accuracy")
            # Derive cost per 1k output tokens from GPU hourly rate and throughput
            tput = entry.get("throughput", {})
            output_tps = tput.get("output_tokens_per_sec") if isinstance(tput, dict) else None
            if output_tps and output_tps > 0:
                cost_per_1k = gpu_cost_hr / (output_tps * 3600) * 1000
            else:
                cost_per_1k = entry.get("cost", {}).get("self_hosted_per_1k_output_usd")
            if accuracy is None or cost_per_1k is None:
                continue
            model = entry.get("model_id", "unknown")
            short_model = model.split("/")[-1] if "/" in model else model
            label = f"{short_model}\n{mode}"
            points.append(
                {
                    "label": label,
                    "cost_per_1k": float(cost_per_1k),
                    "accuracy": float(accuracy),
                    "mode": mode,
                    "gpu": gpu,
                    "model": model,
                }
            )
    return points


def _pareto_frontier(points: list[dict]) -> list[dict]:
    """Return points on the cost/quality Pareto frontier (lower cost, higher accuracy)."""
    sorted_pts = sorted(points, key=lambda p: p["cost_per_1k"])
    frontier = []
    best_acc = -1.0
    for p in sorted_pts:
        if p["accuracy"] > best_acc:
            best_acc = p["accuracy"]
            frontier.append(p)
    return frontier


def plot(results_dir: Path, output: Path) -> None:
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    points = _load_points(results_dir)
    if not points:
        print(f"No benchmark data found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    frontier = _pareto_frontier(points)
    frontier_labels = {p["label"] for p in frontier}

    fig, ax = plt.subplots(figsize=(10, 7))

    # All points
    for p in points:
        on_frontier = p["label"] in frontier_labels
        color = "tab:orange" if on_frontier else "tab:blue"
        marker = "*" if on_frontier else "o"
        size = 120 if on_frontier else 60
        ax.scatter(p["cost_per_1k"], p["accuracy"], color=color, marker=marker, s=size, zorder=3)
        ax.annotate(
            p["label"],
            (p["cost_per_1k"], p["accuracy"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    # Pareto frontier line
    if len(frontier) > 1:
        xs = [p["cost_per_1k"] for p in frontier]
        ys = [p["accuracy"] for p in frontier]
        ax.plot(xs, ys, "k--", linewidth=1, label="Pareto frontier", zorder=2)

    ax.set_xlabel("Cost per 1k output tokens (USD)", fontsize=11)
    ax.set_ylabel("MMLU accuracy (quality proxy)", fontsize=11)
    ax.set_title("Cross-Model Cost / Quality Pareto Chart", fontsize=13)

    patches = [
        mpatches.Patch(color="tab:orange", label="Pareto-optimal"),
        mpatches.Patch(color="tab:blue", label="Dominated"),
    ]
    ax.legend(handles=patches, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved Pareto chart → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cost/quality Pareto chart")
    parser.add_argument("--results", default="results/", help="Directory with benchmark JSON files")
    parser.add_argument("--output", default="charts/pareto.png", help="Output PNG path")
    args = parser.parse_args()
    plot(Path(args.results), Path(args.output))


if __name__ == "__main__":
    main()
