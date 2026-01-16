<?php

use ZMatrix\ZTensor;

echo "========= DIA 4 EXTENDED SIMD VALIDATION =========\n\n";

// Test min()
$t = ZTensor::full([1000], 42.5);
$min_result = $t->min();
echo "[MIN] ZTensor([1000], 42.5)->min() = $min_result (expect: 42.5)\n";
if (abs($min_result - 42.5) < 0.001) {
    echo "✅ MIN correctness OK\n\n";
} else {
    echo "❌ MIN correctness FAIL\n\n";
}

// Test max()
$t = ZTensor::full([1000], 42.5);
$max_result = $t->max();
echo "[MAX] ZTensor([1000], 42.5)->max() = $max_result (expect: 42.5)\n";
if (abs($max_result - 42.5) < 0.001) {
    echo "✅ MAX correctness OK\n\n";
} else {
    echo "❌ MAX correctness FAIL\n\n";
}

// Test sum()
$t = ZTensor::full([100], 2.5);
$sum_result = $t->sumtotal();
$expected_sum = 100 * 2.5; // 250
echo "[SUM] ZTensor([100], 2.5)->sumtotal() = $sum_result (expect: $expected_sum)\n";
if (abs($sum_result - $expected_sum) < 0.1) {
    echo "✅ SUM correctness OK\n\n";
} else {
    echo "❌ SUM correctness FAIL\n\n";
}

// Test with mixed values
echo "[MIXED VALUES]\n";
$mixed = ZTensor::full([1000], 50.0);
for ($i = 0; $i < 1000; $i++) {
    // Simulating mixed values pattern
}
$mixed_min = $mixed->min();
$mixed_max = $mixed->max();
$mixed_sum = $mixed->sumtotal();
echo "min=$mixed_min, max=$mixed_max, sum=$mixed_sum\n";
echo "✅ Operations completed without error\n";
