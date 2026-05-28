"""Parse nsys stats CSV output into a kernel breakdown JSON.

run_profile.sh calls `nsys stats` to produce CSV files, then calls this script
to aggregate and format them. No subprocess or sqlite dependency here.

Usage (called by run_profile.sh):
    python profiling/parse_nsys.py \\
        --kernsum profiling/profiles/fp16_kernsum.csv \\
        --memsum  profiling/profiles/fp16_memsum.csv \\
        --mode fp16 \\
        --output results/profile_kernels_fp16.json
"""

from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path


def _read_csv(path: str) -> list[dict]:
    text = Path(path).read_text()
    lines = text.splitlines()
    # Skip preamble lines (Generating..., blanks) until we reach the CSV header
    csv_start = next(
        (i for i, line in enumerate(lines) if "," in line and not line.startswith("Generating")),
        None,
    )
    if csv_start is None:
        return []
    reader = csv.DictReader(StringIO("\n".join(lines[csv_start:])))
    return list(reader)


def _get(row: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in row:
            try:
                return float(row[k])
            except (ValueError, TypeError):
                pass
    return default


def parse(kernsum_csv: str, memsum_csv: str, mode: str) -> dict:
    kern_rows = _read_csv(kernsum_csv) if Path(kernsum_csv).is_file() else []
    total_gpu_ns = sum(_get(r, "Total Time (ns)", "Total(ns)") for r in kern_rows)

    top_kernels = []
    for row in kern_rows[:20]:
        duration_ns = _get(row, "Total Time (ns)", "Total(ns)")
        count = int(_get(row, "Count", "Instances", default=1))
        name = row.get("Name", row.get("Kernel Name", "unknown"))
        pct = round(duration_ns / total_gpu_ns * 100, 2) if total_gpu_ns else 0.0
        top_kernels.append(
            {
                "name": name[:80],
                "total_ms": round(duration_ns / 1e6, 3),
                "count": count,
                "pct_gpu_time": pct,
                "avg_us": round(duration_ns / max(count, 1) / 1e3, 2),
            }
        )

    mem_rows = _read_csv(memsum_csv) if Path(memsum_csv).is_file() else []
    total_memcpy_ms = sum(_get(r, "Total Time (ns)", "Total(ns)") for r in mem_rows) / 1e6

    categories: dict[str, float] = {
        "attention": 0.0,
        "matmul": 0.0,
        "dequantize": 0.0,
        "layernorm": 0.0,
        "other": 0.0,
    }
    for k in top_kernels:
        name_lower = k["name"].lower()
        ms = k["total_ms"]
        if any(x in name_lower for x in ["attention", "flash", "sdpa", "fmha", "mhsa"]):
            categories["attention"] += ms
        elif any(x in name_lower for x in ["gemm", "cublas", "matmul", "volta_h884", "ampere_h1688", "sgemm", "hgemm"]):
            categories["matmul"] += ms
        elif any(x in name_lower for x in ["dequant", "weight_only", "marlin", "awq", "gptq"]):
            categories["dequantize"] += ms
        elif any(x in name_lower for x in ["layernorm", "rms_norm", "layer_norm", "rmsnorm"]):
            categories["layernorm"] += ms
        else:
            categories["other"] += ms

    return {
        "mode": mode,
        "total_gpu_time_ms": round(total_gpu_ns / 1e6, 1),
        "total_memcpy_ms": round(total_memcpy_ms, 1),
        "compute_memcpy_ratio": round(total_gpu_ns / 1e6 / max(total_memcpy_ms, 0.01), 2),
        "top_kernels": top_kernels,
        "categories_ms": {k: round(v, 1) for k, v in categories.items()},
        "categories_pct": {k: round(v / max(total_gpu_ns / 1e6, 0.01) * 100, 1) for k, v in categories.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernsum", required=True, help="CSV from: nsys stats --report gpukernsum")
    parser.add_argument("--memsum", required=True, help="CSV from: nsys stats --report gpumemtimesum")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = parse(args.kernsum, args.memsum, args.mode)

    print(f"\n=== {args.mode} kernel breakdown ===")
    print(f"Total GPU time: {result['total_gpu_time_ms']:.1f}ms  Memcpy: {result['total_memcpy_ms']:.1f}ms")
    print("Category breakdown:")
    for cat, pct in result["categories_pct"].items():
        ms = result["categories_ms"][cat]
        print(f"  {cat:15s} {pct:5.1f}%  ({ms:.1f}ms)")
    print("\nTop kernels:")
    for k in result["top_kernels"][:10]:
        print(f"  {k['pct_gpu_time']:5.1f}%  {k['avg_us']:8.1f}µs  {k['name'][:60]}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"\nWritten to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
