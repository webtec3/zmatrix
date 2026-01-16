<?php
use ZMatrix\ZTensor;

echo "════════════════════════════════════════════════════════════════\n";
echo "DIA 6 - SCALAR OPERATIONS PERFORMANCE BENCHMARK\n";
echo "════════════════════════════════════════════════════════════════\n\n";

// Tamanhos para benchmark
$sizes = [1000, 10000, 100000, 1000000, 10000000];

foreach ($sizes as $size) {
    echo "┌─ Array Size: " . number_format($size) . " elements\n";
    
    // Benchmark scalarMultiply
    $t = ZTensor::full([$size], 2.5);
    $start = microtime(true);
    for ($i = 0; $i < 10; $i++) {
        $t->scalarMultiply(1.5);
    }
    $elapsed = (microtime(true) - $start) * 1000; // em ms
    $ops_per_sec = (10 * $size) / ($elapsed / 1000);
    $gflops = $ops_per_sec / 1e9;
    printf("│  scalarMultiply: %.2f ms | %.2f Gflops/s\n", $elapsed, $gflops);
    
    // Benchmark scalarDivide
    $t = ZTensor::full([$size], 10.0);
    $start = microtime(true);
    for ($i = 0; $i < 10; $i++) {
        $t->scalarDivide(2.0);
    }
    $elapsed = (microtime(true) - $start) * 1000;
    $ops_per_sec = (10 * $size) / ($elapsed / 1000);
    $gflops = $ops_per_sec / 1e9;
    printf("│  scalarDivide:   %.2f ms | %.2f Gflops/s\n", $elapsed, $gflops);
    
    // Benchmark element-wise divide
    $t_a = ZTensor::full([$size], 20.0);
    $t_b = ZTensor::full([$size], 2.0);
    $start = microtime(true);
    for ($i = 0; $i < 10; $i++) {
        // Clone para evitar que o tensor fique muito pequeno
        $t_test = ZTensor::full([$size], 20.0);
        $t_div = ZTensor::full([$size], 2.0);
        $t_test->divide($t_div);
    }
    $elapsed = (microtime(true) - $start) * 1000;
    $ops_per_sec = (10 * $size) / ($elapsed / 1000);
    $gflops = $ops_per_sec / 1e9;
    printf("│  divide (elem):  %.2f ms | %.2f Gflops/s\n", $elapsed, $gflops);
    
    echo "└\n";
}

echo "\n════════════════════════════════════════════════════════════════\n";
echo "✅ DIA 6 PERFORMANCE BENCHMARK COMPLETE\n";
echo "════════════════════════════════════════════════════════════════\n";
echo "\nNote: Throughput values are operational (GFLOPS) not absolute\n";
echo "Real performance depends on:\n";
echo "  - CPU cache effects\n";
echo "  - Memory bandwidth\n";
echo "  - SIMD instruction execution\n";
