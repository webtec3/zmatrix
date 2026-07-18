#!/usr/bin/env bash

set -euo pipefail

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$(mktemp -d /tmp/zmatrix-asan-build.XXXXXX)"

rsync -r \
    --exclude=.git \
    --exclude=.libs \
    --exclude=modules \
    --exclude=gpu_kernels.o \
    --exclude=.venv \
    "$source_dir/" "$build_dir/"

cd "$build_dir"
phpize >/dev/null
ac_cv_path_NVCC=no \
    CXXFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" \
    LDFLAGS="-fsanitize=address,undefined" \
    ./configure --enable-zmatrix >/tmp/zmatrix-asan-configure.log
make clean >/dev/null
make -j"$(nproc)" >/tmp/zmatrix-asan-build.log

asan_library="$(gcc -print-file-name=libasan.so)"
LD_PRELOAD="$asan_library" \
ASAN_OPTIONS="detect_leaks=1:halt_on_error=1" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
php -n -d "extension=$build_dir/modules/zmatrix.so" "$build_dir/tests/cpu_lifecycle.php"
LD_PRELOAD="$asan_library" \
ASAN_OPTIONS="detect_leaks=1:halt_on_error=1" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
php -n -d "extension=$build_dir/modules/zmatrix.so" "$build_dir/tests/core_extended_ops.php"

echo "Sanitized CPU build passed: $build_dir"
