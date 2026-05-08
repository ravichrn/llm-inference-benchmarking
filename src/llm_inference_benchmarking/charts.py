"""Analysis charts for modal benchmark results.

Generates three charts from modal_benchmark JSON result files:
  1. TTFT vs Throughput scatter  — fundamental latency/throughput tradeoff
  2. Batch size vs latency       — how batching pressure increases per-request latency
  3. Quant level vs quality      — MMLU accuracy and perplexity across quantization methods

Usage:
  uv run llm-charts --results results/ --output-dir charts/
  uv run python -m llm_inference_benchmarking.charts --results results/ --output-dir charts/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Canonical ordering for quantization quality axis (coarsest → finest precision)
_QUANT_ORDER = ["cpu-q2k", "cpu-q4km", "cpu-q5km", "cpu-q8_0", "int8", "nf4-dq", "nf4", "gptq", "fp8", "fp16"]


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

    # batch_throughput schema: {"batch_1": {"output_tokens_per_sec": X, "batch_size": 1}, ...}
    # Derive per-request latency from throughput: latency_ms ≈ (max_new_tokens / tps) * 1000
    # We use relative latency (normalised to batch=1) since absolute depends on max_new_tokens.
    series: dict[str, dict[int, float]] = {}
    for e in entries:
        mode = e.get("quant_mode", "")
        bt = e.get("batch_throughput") or {}
        if not isinstance(bt, dict):
            continue
        row: dict[int, float] = {}
        for key, val in bt.items():
            if isinstance(val, dict):
                # nested schema: {"batch_1": {"output_tokens_per_sec": X, "batch_size": 1}}
                bs = val.get("batch_size") or int(key.split("_")[-1])
                tps = val.get("output_tokens_per_sec")
            elif isinstance(val, int | float):
                # flat schema: {"batch1_output_tokens_per_sec": X}
                # key format: "batch{N}_output_tokens_per_sec"
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
        # Convert tps → relative slowdown vs batch=1 (higher = worse per-request latency)
        base_tps = row.get(1) or row.get(min(row))
        ys = [base_tps / row[bs] for bs in xs]  # ratio: 1.0 = same as batch-1, 2.0 = 2x slower
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis charts from benchmark results")
    parser.add_argument("--results", default="results/", help="Directory with benchmark JSON files")
    parser.add_argument("--output-dir", default="charts/", help="Output directory for PNG charts")
    args = parser.parse_args()
    generate_all(Path(args.results), Path(args.output_dir))


if __name__ == "__main__":
    main()
