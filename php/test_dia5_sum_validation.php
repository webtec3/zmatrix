<?php
use ZMatrix\ZTensor;

echo "═══════════════════════════════════════════════\n";
echo "DIA 5 - SUM() SIMD PERFORMANCE VALIDATION\n";
echo "═══════════════════════════════════════════════\n\n";

// Test 1: Correctness validation
echo "[1] CORRECTNESS TESTS\n";
echo "─────────────────────────────────────────────\n";

$tests = [
    [10, 1.0],
    [100, 2.5],
    [1000, 0.5],
    [10000, 1.5],
    [100000, 0.1],
    [1000000, 2.0],
];

foreach ($tests as [$size, $value]) {
    $tensor = ZTensor::full([$size], $value);
    $sum = $tensor->sumtotal();
    $expected = $size * $value;
    $error = abs($sum - $expected) / $expected * 100;
    
    $status = ($error < 0.01) ? "✅" : "❌";
    printf("%s Size: %7d | Value: %.1f | Sum: %.1f | Expected: %.1f | Error: %.6f%%\n",
        $status, $size, $value, $sum, $expected, $error);
}

echo "\n[2] MEAN TEST (uses sum internally)\n";
echo "─────────────────────────────────────────────\n";

$tensor = ZTensor::full([1000], 5.0);
$mean = $tensor->mean();
$expected_mean = 5.0;
$error = abs($mean - $expected_mean) / $expected_mean * 100;
$status = ($error < 0.01) ? "✅" : "❌";
printf("%s Mean: %.6f | Expected: %.6f | Error: %.6f%%\n", 
    $status, $mean, $expected_mean, $error);

echo "\n[3] SIMD VECTORIZATION CHECK\n";
echo "─────────────────────────────────────────────\n";

// Test aligned and non-aligned sizes (to ensure tail loop works)
$sizes = [7, 8, 15, 16, 17, 100, 1023, 1024, 1025];

foreach ($sizes as $size) {
    $tensor = ZTensor::full([$size], 1.0);
    $sum = $tensor->sumtotal();
    $status = ($sum == $size) ? "✅" : "❌";
    echo "$status Size: $size (aligned: " . ($size % 8 == 0 ? "yes" : "no ") . ") → sum = $sum\n";
}

echo "\n[4] MULTI-DIMENSIONAL TENSOR TEST\n";
echo "─────────────────────────────────────────────\n";

$tensor = ZTensor::full([10, 20, 30], 0.5);  // 6000 elements
$sum = $tensor->sumtotal();
$expected = 10 * 20 * 30 * 0.5;
$error = abs($sum - $expected) / $expected * 100;
$status = ($error < 0.01) ? "✅" : "❌";
printf("%s 3D Tensor [10x20x30] sum: %.1f | Expected: %.1f | Error: %.6f%%\n",
    $status, $sum, $expected, $error);

echo "\n[5] EDGE CASES\n";
echo "─────────────────────────────────────────────\n";

// Zero values
$tensor = ZTensor::full([100], 0.0);
$sum = $tensor->sumtotal();
$status = ($sum == 0.0) ? "✅" : "❌";
echo "$status All zeros: sum = $sum\n";

// Negative values
$tensor = ZTensor::full([100], -2.5);
$sum = $tensor->sumtotal();
$expected = 100 * -2.5;
$status = ($sum == $expected) ? "✅" : "❌";
echo "$status All negative: sum = $sum (expected: $expected)\n";

// Very large values
$tensor = ZTensor::full([100], 1e6);
$sum = $tensor->sumtotal();
$expected = 100 * 1e6;
$error = abs($sum - $expected) / $expected * 100;
$status = ($error < 0.01) ? "✅" : "❌";
printf("%s Large values (1e6): sum = %.0f | Expected: %.0f\n", $status, $sum, $expected);

// Very small values  
$tensor = ZTensor::full([100], 1e-6);
$sum = $tensor->sumtotal();
$expected = 100 * 1e-6;
$error = abs($sum - $expected) / $expected * 100;
$status = ($error < 0.1) ? "✅" : "❌";
printf("%s Small values (1e-6): sum = %.8f | Expected: %.8f\n", $status, $sum, $expected);

echo "\n═══════════════════════════════════════════════\n";
echo "✅ ALL DIA 5 SUM() VALIDATION TESTS COMPLETE\n";
echo "═══════════════════════════════════════════════\n";
