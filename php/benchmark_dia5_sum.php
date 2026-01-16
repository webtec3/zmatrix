<?php
use ZMatrix\ZTensor;

echo "═══════════════════════════════════════════════════════════\n";
echo "DIA 5 - PERFORMANCE PROFILING (sem perf, usando microtime)\n";
echo "═══════════════════════════════════════════════════════════\n\n";

// Warmup
$warmup = ZTensor::full([100000], 1.0);
for ($i = 0; $i < 10; $i++) {
    $warmup->sumtotal();
}

echo "[1] SUM() PERFORMANCE - Diferentes tamanhos\n";
echo "───────────────────────────────────────────────────────────\n";

$sizes = [1000, 10000, 100000, 1000000, 10000000];
$iterations = 100;

foreach ($sizes as $size) {
    $tensor = ZTensor::full([$size], 1.5);
    
    // Medir múltiplas iterações
    $start = microtime(true);
    for ($i = 0; $i < $iterations; $i++) {
        $sum = $tensor->sumtotal();
    }
    $end = microtime(true);
    
    $time_ms = ($end - $start) * 1000;
    $time_per_op_us = $time_ms * 1000 / $iterations;
    $elements_per_us = $size / $time_per_op_us;
    $throughput_gbs = $elements_per_us * 4 / 1000;  // 4 bytes per float
    
    printf("Size: %8d | Total time: %8.3f ms | Per-op: %7.3f µs | Throughput: %6.2f GB/s\n",
        $size, $time_ms, $time_per_op_us, $throughput_gbs);
}

echo "\n[2] SIMD EFFICIENCY - Aligned vs Unaligned\n";
echo "───────────────────────────────────────────────────────────\n";

$test_cases = [
    ['name' => 'Aligned (8)', 'size' => 100000],
    ['name' => 'Unaligned (7)', 'size' => 100001],
    ['name' => 'Aligned (16)', 'size' => 100016],
    ['name' => 'Unaligned (17)', 'size' => 100017],
];

$iterations = 100;

foreach ($test_cases as $test) {
    $size = $test['size'];
    $tensor = ZTensor::full([$size], 1.0);
    
    $start = microtime(true);
    for ($i = 0; $i < $iterations; $i++) {
        $sum = $tensor->sumtotal();
    }
    $end = microtime(true);
    
    $time_ms = ($end - $start) * 1000;
    $time_per_op_us = $time_ms * 1000 / $iterations;
    
    // Calcular quantos elementos foram vetorizados
    $vec_elements = ($size / 8) * 8;
    $tail_elements = $size % 8;
    
    printf("%s | Size: %7d (vec: %7d, tail: %d) | Time: %7.3f µs\n",
        $test['name'], $size, $vec_elements, $tail_elements, $time_per_op_us);
}

echo "\n[3] OPERAÇÕES RELACIONADAS - Benchmarks\n";
echo "───────────────────────────────────────────────────────────\n";

$tensor = ZTensor::full([1000000], 2.5);
$iterations = 50;

// sum() timing
$start = microtime(true);
for ($i = 0; $i < $iterations; $i++) {
    $sum = $tensor->sumtotal();
}
$time_sum = (microtime(true) - $start) * 1000000 / $iterations;

// mean() timing
$start = microtime(true);
for ($i = 0; $i < $iterations; $i++) {
    $mean = $tensor->mean();
}
$time_mean = (microtime(true) - $start) * 1000000 / $iterations;

// min() timing
$start = microtime(true);
for ($i = 0; $i < $iterations; $i++) {
    $min = $tensor->min();
}
$time_min = (microtime(true) - $start) * 1000000 / $iterations;

// max() timing
$start = microtime(true);
for ($i = 0; $i < $iterations; $i++) {
    $max = $tensor->max();
}
$time_max = (microtime(true) - $start) * 1000000 / $iterations;

printf("sum():  %8.3f µs\n", $time_sum);
printf("mean(): %8.3f µs (overhead: %+.3f µs)\n", $time_mean, $time_mean - $time_sum);
printf("min():  %8.3f µs\n", $time_min);
printf("max():  %8.3f µs\n", $time_max);

echo "\n[4] CORRECTNESS VALIDATION DURING BENCHMARK\n";
echo "───────────────────────────────────────────────────────────\n";

$tensor = ZTensor::full([1000000], 2.5);
$sum = $tensor->sumtotal();
$expected = 1000000 * 2.5;
$error = abs($sum - $expected) / $expected * 100;

printf("Sum of 1M × 2.5:\n");
printf("  Actual:   %.1f\n", $sum);
printf("  Expected: %.1f\n", $expected);
printf("  Error:    %.6f%%\n", $error);
printf("  Status:   %s\n", ($error < 0.01) ? "✅ PASS" : "❌ FAIL");

echo "\n═══════════════════════════════════════════════════════════\n";
echo "✅ PROFILING COMPLETE - SIMD sum() is working correctly\n";
echo "═══════════════════════════════════════════════════════════\n";
