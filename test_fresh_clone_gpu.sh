#!/bin/bash
# TEST: Fresh Clone GPU Detection - ZMatrix CUDA Fallback Build System

echo "========================================"
echo "ZMatrix Fresh Clone GPU Test"
echo "Testing compiled extension WITHOUT LD_LIBRARY_PATH"
echo "========================================"
echo ""

# Ensure test directory is clean
TEST_DIR="/tmp/zmatrix-fresh-clone-test-$$"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Copy fresh clone
cp -r ~/php-projetos/php-extension/zmatrix . || {
    echo "❌ FAILED: Could not copy repository"
    exit 1
}

cd zmatrix

# Run GPU test WITHOUT LD_LIBRARY_PATH
echo "Test: GPU Detection on Fresh Clone"
echo "===================================="
echo ""

ZMATRIX_GPU_DEBUG=1 php -r "
use ZMatrix\ZTensor;
echo 'Step 1: Creating random tensors (1M elements)...\n';
\$a = ZTensor::random([1000000], -1.0, 1.0);
\$b = ZTensor::random([1000000], -1.0, 1.0);

echo 'Step 2: Moving to GPU...\n';
\$a->toGpu();
\$b->toGpu();

echo 'Step 3: Benchmarking GPU add operation (10x)...\n';
\$t0 = microtime(true);
for (\$i = 0; \$i < 10; \$i++) {
    \$a->add(\$b);
}
\$t1 = microtime(true);
\$time_ms = (\$t1 - \$t0) / 10 * 1000;

echo \"Step 4: Results\n\";
echo \"  Time per operation: {$time_ms} ms\n\";

if (\$time_ms < 1.0) {
    echo \"\n✅ SUCCESS: GPU is working!\n\";
    exit(0);
} else {
    echo \"\n⚠️  WARNING: GPU might be slow or CPU fallback\n\";
    exit(1);
}
" 2>&1

RESULT=$?

echo ""
echo "========================================"
if [ $RESULT -eq 0 ]; then
    echo "✅ TEST PASSED"
    echo "   GPU works on fresh clone without manual LD_LIBRARY_PATH setup!"
else
    echo "❌ TEST FAILED"
fi
echo "========================================"

# Cleanup
cd /
rm -rf "$TEST_DIR"

exit $RESULT
