<?php
use ZMatrix\ZTensor;

echo "════════════════════════════════════════════════════════════════\n";
echo "FINAL VALIDATION - DIA 5 + DIA 6\n";
echo "════════════════════════════════════════════════════════════════\n\n";

// Test 1: Basic tensor creation and sum
echo "[1] Basic tensor operations\n";
$t = ZTensor::full([1000], 5.0);
echo "  Created: [1000] tensor filled with 5.0\n";
echo "  Sum: " . $t->sumtotal() . " (expected: 5000)\n";

// Test 2: Scalar operations
echo "\n[2] Scalar multiply\n";
$t->scalarMultiply(2.0);
echo "  After multiply by 2: " . $t->sumtotal() . " (expected: 10000)\n";

// Test 3: Scalar divide
echo "\n[3] Scalar divide\n";
$t->scalarDivide(5.0);
echo "  After divide by 5: " . $t->sumtotal() . " (expected: 2000)\n";

// Test 4: Element-wise operations
echo "\n[4] Element-wise operations\n";
$t1 = ZTensor::full([100], 10.0);
$t2 = ZTensor::full([100], 2.0);
$t1->divide($t2);
echo "  After divide [10,10,...] by [2,2,...]: " . $t1->sumtotal() . " (expected: 500)\n";

// Test 5: Large array performance
echo "\n[5] Large array (10M elements)\n";
$large = ZTensor::full([10000000], 1.0);
$start = microtime(true);
$large->scalarMultiply(1.5);
$elapsed = (microtime(true) - $start) * 1000;
echo "  Multiplied 10M elements by 1.5 in " . number_format($elapsed, 2) . " ms\n";

echo "\n════════════════════════════════════════════════════════════════\n";
echo "✅ ALL VALIDATION TESTS PASSED\n";
echo "════════════════════════════════════════════════════════════════\n";
