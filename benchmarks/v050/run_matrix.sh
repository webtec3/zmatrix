#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/benchmarks/v050/results}"
SUITE="${SUITE:-full}"
REPETITIONS="${REPETITIONS:-9}"
WARMUPS="${WARMUPS:-3}"
PHP_BIN="${PHP_BIN:-php}"
EXTENSION="${ZMATRIX_EXTENSION:-$ROOT/modules/zmatrix.so}"
BENCHMARK="$ROOT/benchmarks/v050/benchmark_matrix.php"
mkdir -p "$OUT"

"$PHP_BIN" -n -d "extension=$EXTENSION" "$BENCHMARK" \
  --suite="$SUITE" --categories=sgemm,chain,reduction \
  --repetitions="$REPETITIONS" --warmups="$WARMUPS" --output="$OUT"

for implementation in serial hierarchical cub; do
  ZMATRIX_REDUCTION_IMPL="$implementation" \
  "$PHP_BIN" -n -d "extension=$EXTENSION" "$BENCHMARK" \
    --suite="$SUITE" --categories=reduction \
    --repetitions="$REPETITIONS" --warmups="$WARMUPS" --output="$OUT"
done

