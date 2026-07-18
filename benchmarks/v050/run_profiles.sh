#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/benchmarks/v050/profiles}"
PHP_BIN="${PHP_BIN:-php}"
EXTENSION="${ZMATRIX_EXTENSION:-$ROOT/modules/zmatrix.so}"
mkdir -p "$OUT"

nsys --version > "$OUT/tool_versions.txt" 2>&1
ncu --version >> "$OUT/tool_versions.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >> "$OUT/tool_versions.txt"
nvcc --version >> "$OUT/tool_versions.txt" 2>&1

set +e
timeout "${NSYS_TIMEOUT:-45}s" nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --output="$OUT/zmatrix_systems" \
  "$PHP_BIN" -n -d "extension=$EXTENSION" "$ROOT/benchmarks/v050/profile_target.php" 1024 10 \
  > "$OUT/nsys.log" 2>&1
echo "$?" > "$OUT/nsys.exitcode"

ncu \
  --target-processes all \
  --set full \
  --kernel-name regex:'kernel_|void gemm' \
  --launch-skip 3 \
  --launch-count 12 \
  --export "$OUT/zmatrix_compute" \
  "$PHP_BIN" -n -d "extension=$EXTENSION" "$ROOT/benchmarks/v050/profile_target.php" 1024 10 \
  > "$OUT/ncu.log" 2>&1
echo "$?" > "$OUT/ncu.exitcode"
set -e
