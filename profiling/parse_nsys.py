"""Parse Nsight Systems sqlite output into a kernel breakdown JSON.

Reads the .sqlite file that nsys produces directly via Python's stdlib sqlite3 —
no nsys binary required for parsing.

Usage:
    python profiling/parse_nsys.py profiling/profiles/fp16.sqlite --mode fp16
    python profiling/parse_nsys.py profiling/profiles/fp16.sqlite --mode fp16 --output results/profile_kernels_fp16.json

Generate the sqlite file with:
    nsys profile --output profiling/profiles/fp16 --export sqlite --trace cuda,nvtx \\
        python profiling/profile_benchmark.py --mode fp16
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def _open_readonly(sqlite_path: str) -> sqlite3.Connection:
    path = Path(sqlite_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {sqlite_path!r}")
    if path.suffix.lower() != ".sqlite":
        raise ValueError(f"Expected a .sqlite file, got: {sqlite_path!r}")
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def _table_names(con: sqlite3.Connection) -> frozenset[str]:
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return frozenset(r[0] for r in rows)


def _read_kernels(sqlite_path: str) -> list[dict]:
    """Return per-kernel aggregate rows sorted by total GPU time descending."""
    con = _open_readonly(sqlite_path)
    try:
        tables = _table_names(con)
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            print("Warning: CUPTI_ACTIVITY_KIND_KERNEL table not found; no kernel data.", file=sys.stderr)
            return []

        if "StringIds" in tables:
            sql = """
                SELECT COALESCE(s.value, CAST(k.shortName AS TEXT)) AS name,
                       COUNT(*)                                       AS cnt,
                       SUM(k.end - k.start)                          AS total_ns
                FROM   CUPTI_ACTIVITY_KIND_KERNEL k
                LEFT JOIN StringIds s ON k.shortName = s.id
                GROUP  BY k.shortName
                ORDER  BY total_ns DESC
            """
        else:
            sql = """
                SELECT CAST(shortName AS TEXT) AS name,
                       COUNT(*)                AS cnt,
                       SUM(end - start)        AS total_ns
                FROM   CUPTI_ACTIVITY_KIND_KERNEL
                GROUP  BY shortName
                ORDER  BY total_ns DESC
            """

        rows = []
        for name, cnt, total_ns in con.execute(sql):
            rows.append(
                {
                    "name": str(name or "unknown"),
                    "count": int(cnt or 1),
                    "total_ns": float(total_ns or 0.0),
                }
            )
        return rows

    except sqlite3.OperationalError as exc:
        print(f"Warning: kernel query failed: {exc}", file=sys.stderr)
        return []
    finally:
        con.close()


def _read_memcpy_ns(sqlite_path: str) -> float:
    """Return total memory-copy time in nanoseconds."""
    con = _open_readonly(sqlite_path)
    try:
        if "CUPTI_ACTIVITY_KIND_MEMCPY" not in _table_names(con):
            return 0.0
        result = con.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()
        return float(result[0] or 0.0)
    except sqlite3.OperationalError:
        return 0.0
    finally:
        con.close()


def _get(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (ValueError, TypeError):
        return default


def parse(sqlite_path: str, mode: str) -> dict:
    kern_rows = _read_kernels(sqlite_path)
    total_gpu_ns = sum(r["total_ns"] for r in kern_rows)

    top_kernels = []
    for row in kern_rows[:20]:
        duration_ns = row["total_ns"]
        count = row["count"]
        name = row["name"]
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

    total_memcpy_ms = _read_memcpy_ns(sqlite_path) / 1e6

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
    parser.add_argument("sqlite", help=".sqlite file from nsys profile --export sqlite")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = parse(args.sqlite, args.mode)

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
