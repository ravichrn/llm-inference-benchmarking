"""Visualisation tools for modal benchmark results.

Two entry points:
  llm-pareto  — cross-model cost/quality Pareto chart
  llm-charts  — analysis charts (TTFT vs throughput, batch latency, quant quality)

Usage:
  uv run llm-pareto  --results results/ --output charts/pareto.png
  uv run llm-charts  --results results/ --output-dir charts/
  uv run python -m llm_inference_benchmarking.viz pareto  --results results/
  uv run python -m llm_inference_benchmarking.viz charts   --results results/

Requires: matplotlib (optional dep)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Canonical ordering for quantization quality axis (coarsest → finest precision)
_QUANT_ORDER = ["cpu-q2k", "cpu-q4km", "cpu-q5km", "cpu-q8_0", "int8", "nf4-dq", "nf4", "gptq", "fp8", "fp16"]


# ---------------------------------------------------------------------------
# Shared loader
# ---------------------------------------------------------------------------


def _load_results(results_dir: Path) -> list[dict]:
    """Load all modal_quant*.json result entries into a flat list."""
    entries: list[dict] = []
    for fpath in sorted(results_dir.glob("modal_quant*.json")):
        try:
            data = json.loads(fpath.read_text())
        except Exception as exc:
            print(f"Skipping {fpath.name}: {exc}", file=sys.stderr)
            continue
        results = data.get("results", data) if isinstance(data, dict) else data
        if isinstance(results, list):
            entries.extend(results)
    return entries


# ---------------------------------------------------------------------------
# Pareto chart
# ---------------------------------------------------------------------------


def _load_points(results_dir: Path) -> list[dict]:
    """Return list of {label, cost_per_1k, accuracy, mode, gpu} dicts."""
    points = []
    for fpath in sorted(results_dir.glob("modal_quant*.json")):
        try:
            data = json.loads(fpath.read_text())
        except Exception as exc:
            print(f"Skipping {fpath.name}: {exc}", file=sys.stderr)
            continue
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


def plot_pareto(results_dir: Path, output: Path) -> None:
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


# ---------------------------------------------------------------------------
# Analysis charts
# ---------------------------------------------------------------------------


def _ttft_vs_throughput(entries: list[dict], output: Path) -> None:
    """Chart 1: TTFT (ms) vs output throughput (tok/s) per mode."""
    import matplotlib.pyplot as plt

    points = []
    for e in entries:
        mode = e.get("quant_mode", "")
        lat = e.get("latency") or {}
        thr = e.get("throughput") or {}
        ttft = lat.get("ttft_mean_ms")
        tps = thr.get("output_tokens_per_sec")
        if ttft is None or tps is None:
            continue
        points.append({"mode": mode, "ttft_ms": float(ttft), "tps": float(tps)})

    if not points:
        print("ttft_vs_throughput: no data with both ttft_mean_ms and output_tokens_per_sec", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for i, p in enumerate(points):
        color = colors[i % len(colors)]
        ax.scatter(p["tps"], p["ttft_ms"], color=color, s=100, zorder=3)
        ax.annotate(p["mode"], (p["tps"], p["ttft_ms"]), fontsize=8, xytext=(5, 3), textcoords="offset points")

    ax.set_xlabel("Output throughput (tokens/sec)", fontsize=11)
    ax.set_ylabel("Time to first token — TTFT (ms)", fontsize=11)
    ax.set_title("TTFT vs Throughput: Latency/Throughput Tradeoff by Mode", fontsize=13)
    ax.grid(True, alpha=0.3)
    note = "Lower-left = fast TTFT + high throughput (ideal). Batching pushes right and up."
    ax.text(0.01, 0.99, note, transform=ax.transAxes, fontsize=7, va="top", color="grey")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved → {output}")
    plt.close(fig)


def _batch_size_vs_latency(entries: list[dict], output: Path) -> None:
    """Chart 2: Per-request latency (ms) at batch sizes 1 / 4 / 8, one line per mode."""
    import matplotlib.pyplot as plt

    series: dict[str, dict[int, float]] = {}
    for e in entries:
        mode = e.get("quant_mode", "")
        bt = e.get("batch_throughput") or {}
        if not isinstance(bt, dict):
            continue
        row: dict[int, float] = {}
        for key, val in bt.items():
            if isinstance(val, dict):
                bs = val.get("batch_size") or int(key.split("_")[-1])
                tps = val.get("output_tokens_per_sec")
            elif isinstance(val, int | float):
                import re

                m = re.search(r"batch(\d+)", key)
                if not m:
                    continue
                bs = int(m.group(1))
                tps = float(val)
            else:
                continue
            if tps and tps > 0:
                row[int(bs)] = tps
        if len(row) >= 2:
            series[mode] = row

    if not series:
        print("batch_size_vs_latency: no batch_throughput data found", file=sys.stderr)
        return

    batch_sizes = sorted({bs for row in series.values() for bs in row})
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, (mode, row) in enumerate(sorted(series.items())):
        xs = [bs for bs in batch_sizes if bs in row]
        base_tps = row.get(1) or row.get(min(row))
        ys = [base_tps / row[bs] for bs in xs]
        ax.plot(xs, ys, marker="o", color=colors[i % len(colors)], label=mode, linewidth=1.5)

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="batch=1 baseline")
    ax.set_xlabel("Batch size", fontsize=11)
    ax.set_ylabel("Per-request latency relative to batch=1", fontsize=11)
    ax.set_title("Batching Pressure: Relative Latency Increase vs Batch Size", fontsize=13)
    ax.set_xticks(batch_sizes)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    note = "Values > 1.0 mean higher per-request latency under batching. Flat lines = good scheduling."
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=7, va="bottom", color="grey")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved → {output}")
    plt.close(fig)


def _quant_quality(entries: list[dict], output: Path) -> None:
    """Chart 3: MMLU accuracy and perplexity vs quantization method (coarsest → finest)."""
    import matplotlib.pyplot as plt

    data: dict[str, dict] = {}
    for e in entries:
        mode = e.get("quant_mode", "")
        qual = e.get("quality") or {}
        ppl_info = e.get("perplexity") or {}
        acc = qual.get("mmlu_accuracy")
        ppl = ppl_info.get("perplexity") if isinstance(ppl_info, dict) else None
        if acc is not None:
            data[mode] = {"acc": float(acc), "ppl": float(ppl) if ppl else None}

    if not data:
        print("quant_quality: no quality data found", file=sys.stderr)
        return

    ordered_modes = [m for m in _QUANT_ORDER if m in data]
    remaining = [m for m in data if m not in ordered_modes]
    ordered_modes = ordered_modes + sorted(remaining)

    accs = [data[m]["acc"] for m in ordered_modes]
    ppls = [data[m]["ppl"] for m in ordered_modes]
    has_ppl = any(p is not None for p in ppls)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = range(len(ordered_modes))

    bars = ax1.bar(x, accs, color="tab:blue", alpha=0.7, label="MMLU accuracy (left)")
    ax1.set_ylabel("MMLU accuracy", fontsize=11, color="tab:blue")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(ordered_modes, rotation=30, ha="right", fontsize=9)
    ax1.set_title("Quality vs Quantization Method (MMLU accuracy + perplexity)", fontsize=13)

    for bar, acc in zip(bars, accs, strict=False):
        ax1.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.0%}", ha="center", fontsize=8)

    if has_ppl:
        ax2 = ax1.twinx()
        ppl_x = [i for i, p in zip(x, ppls, strict=False) if p is not None]
        ppl_y = [p for p in ppls if p is not None]
        ax2.plot(ppl_x, ppl_y, color="tab:orange", marker="o", linewidth=1.5, label="Perplexity (right)")
        ax2.set_ylabel("Perplexity (lower = better)", fontsize=11, color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")

    ax1.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved → {output}")
    plt.close(fig)


def generate_all(results_dir: Path, output_dir: Path) -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    entries = _load_results(results_dir)
    if not entries:
        print(f"No benchmark data found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(entries)} result entries from {results_dir}\n")

    _ttft_vs_throughput(entries, output_dir / "ttft_vs_throughput.png")
    _batch_size_vs_latency(entries, output_dir / "batch_latency.png")
    _quant_quality(entries, output_dir / "quant_quality.png")


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def main_pareto() -> None:
    parser = argparse.ArgumentParser(description="Generate cost/quality Pareto chart")
    parser.add_argument("--results", default="results/", help="Directory with benchmark JSON files")
    parser.add_argument("--output", default="charts/pareto.png", help="Output PNG path")
    args = parser.parse_args()
    plot_pareto(Path(args.results), Path(args.output))


def main_charts() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis charts from benchmark results")
    parser.add_argument("--results", default="results/", help="Directory with benchmark JSON files")
    parser.add_argument("--output-dir", default="charts/", help="Output directory for PNG charts")
    args = parser.parse_args()
    generate_all(Path(args.results), Path(args.output_dir))


def main() -> None:
    """Dispatch sub-command: `python -m llm_inference_benchmarking.viz pareto|charts`."""
    parser = argparse.ArgumentParser(description="Visualisation tools for benchmark results")
    parser.add_argument("command", choices=["pareto", "charts"])
    parser.add_argument("--results", default="results/")
    parser.add_argument("--output", default="charts/pareto.png", help="(pareto only)")
    parser.add_argument("--output-dir", default="charts/", help="(charts only)")
    args = parser.parse_args()
    if args.command == "pareto":
        plot_pareto(Path(args.results), Path(args.output))
    else:
        generate_all(Path(args.results), Path(args.output_dir))


if __name__ == "__main__":
    main()
