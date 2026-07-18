#!/usr/bin/env bash

set -euo pipefail

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$(mktemp -d /tmp/zmatrix-cuda-invariants.XXXXXX)"

rsync -r \
    --exclude=.git \
    --exclude=.libs \
    --exclude=modules \
    --exclude=gpu_kernels.o \
    --exclude=.venv \
    "$source_dir/" "$build_dir/"

cd "$build_dir"
phpize >/dev/null
CXXFLAGS="-O2 -g -DZMATRIX_ENABLE_DEBUG_INVARIANTS" \
    ./configure --enable-zmatrix >/tmp/zmatrix-cuda-invariants-configure.log
make clean >/dev/null
make -j"$(nproc)" >/tmp/zmatrix-cuda-invariants-build.log

php -n -d "extension=$build_dir/modules/zmatrix.so" "$build_dir/tests/core_validation.php"
php -n -d "extension=$build_dir/modules/zmatrix.so" "$build_dir/tests/cuda/new_kernels_correctness.php"
php -n -d "extension=$build_dir/modules/zmatrix.so" "$build_dir/tests/cuda/residency_coherence.php"
php -n -d "extension=$build_dir/modules/zmatrix.so" "$build_dir/tests/cuda/extended_ops_correctness.php"
php -n -d "extension=$build_dir/modules/zmatrix.so" "$build_dir/tests/cuda/extended_ops_residency.php"
ZMATRIX_CUDA_ALLOCATOR=auto php -n -d "extension=$build_dir/modules/zmatrix.so" "$build_dir/tests/cuda/allocator_lifecycle_stress.php"
ZMATRIX_CUDA_ALLOCATOR=legacy php -n -d "extension=$build_dir/modules/zmatrix.so" "$build_dir/tests/cuda/allocator_lifecycle_stress.php"
ZMATRIX_CUDA_ALLOCATOR=auto php -n -d "extension=$build_dir/modules/zmatrix.so" "$build_dir/tests/cuda/shutdown_live_tensors.php"

echo "CUDA debug-invariant build passed: $build_dir"
