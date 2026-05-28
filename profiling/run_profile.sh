#!/usr/bin/env bash
# Profile benchmark modes with Nsight Systems on a Lambda Labs GPU instance.
#
# Usage:
#   bash profiling/run_profile.sh                        # all default modes
#   bash profiling/run_profile.sh fp16,vllm,gptq         # specific modes
#
# Output:
#   profiling/profiles/<mode>.nsys-rep         raw Nsight Systems trace
#   results/profile_bench_<mode>.json          latency/throughput numbers
#   results/profile_kernels_<mode>.json        kernel breakdown
#   results/profiling_summary.json             all modes merged

set -euo pipefail

MODES="${1:-fp16,vllm,gptq,sglang,int8}"
CACHE_DIR="${MODEL_CACHE_DIR:-${HOME}/model-cache}"
PROFILE_DIR="profiling/profiles"
RESULTS_DIR="results"

mkdir -p "$PROFILE_DIR" "$RESULTS_DIR"

if ! command -v nsys &>/dev/null; then
  echo "nsys not found. Install with: sudo apt-get install -y nsight-systems"
  exit 1
fi

IFS=',' read -ra MODE_LIST <<< "$MODES"

for mode in "${MODE_LIST[@]}"; do
  echo ""
  echo "=== Profiling: $mode ==="

  nsys profile \
    --output "${PROFILE_DIR}/${mode}" \
    --trace cuda,nvtx \
    --force-overwrite true \
    python profiling/profile_benchmark.py \
      --mode "$mode" \
      --cache-dir "$CACHE_DIR" \
      --output "${RESULTS_DIR}/profile_bench_${mode}.json"

  REP="${PROFILE_DIR}/${mode}.nsys-rep"

  nsys stats --report gpukernsum --format csv "$REP" \
    > "${PROFILE_DIR}/${mode}_kernsum.csv" 2>/dev/null || true

  nsys stats --report gpumemtimesum --format csv "$REP" \
    > "${PROFILE_DIR}/${mode}_memsum.csv" 2>/dev/null || true

  python profiling/parse_nsys.py \
    --kernsum "${PROFILE_DIR}/${mode}_kernsum.csv" \
    --memsum  "${PROFILE_DIR}/${mode}_memsum.csv" \
    --mode "$mode" \
    --output "${RESULTS_DIR}/profile_kernels_${mode}.json"

  echo "Done: $mode"
done

# Merge all kernel results into one summary file
python3 - <<'EOF'
import json, glob, pathlib

kernel_files = sorted(glob.glob("results/profile_kernels_*.json"))
bench_files  = sorted(glob.glob("results/profile_bench_*.json"))

kernel_by_mode = {json.loads(pathlib.Path(f).read_text())["mode"]: json.loads(pathlib.Path(f).read_text()) for f in kernel_files}
bench_by_mode  = {json.loads(pathlib.Path(f).read_text())["mode"]: json.loads(pathlib.Path(f).read_text()) for f in bench_files}

summary = []
for mode in sorted(set(list(kernel_by_mode) + list(bench_by_mode))):
    entry = {"mode": mode}
    if mode in bench_by_mode:
        b = bench_by_mode[mode]
        entry["latency_mean_ms"] = b.get("latency_mean_ms")
        entry["output_tps"] = b.get("output_tps")
        entry["vram_mb"] = b.get("vram_mb")
    if mode in kernel_by_mode:
        k = kernel_by_mode[mode]
        entry["total_gpu_time_ms"] = k.get("total_gpu_time_ms")
        entry["categories_pct"] = k.get("categories_pct")
        entry["top_kernel"] = k["top_kernels"][0]["name"][:60] if k.get("top_kernels") else None
    summary.append(entry)

pathlib.Path("results/profiling_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nSummary written to results/profiling_summary.json ({len(summary)} modes)")

print(f"\n{'Mode':<10} {'Lat(ms)':>8} {'TPS':>6} {'Attn%':>7} {'Matmul%':>8} {'Dequant%':>9} {'Top kernel'}")
print("-" * 90)
for e in summary:
    cats = e.get("categories_pct") or {}
    print(
        f"{e['mode']:<10} {e.get('latency_mean_ms') or 0:>8.0f} {e.get('output_tps') or 0:>6.1f}"
        f" {cats.get('attention', 0):>7.1f} {cats.get('matmul', 0):>8.1f}"
        f" {cats.get('dequantize', 0):>9.1f}  {(e.get('top_kernel') or '')[:40]}"
    )
EOF
