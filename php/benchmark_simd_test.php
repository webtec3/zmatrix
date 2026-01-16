<?php

echo "\n=== TESTE SIMD AVX2 - Performance Comparativa ===\n";
echo "Iterações: 50\n\n";

// Teste com diferentes tamanhos
$test_configs = [
    [100, 100],
    [500, 500],
    [2500, 2500],
];

foreach ($test_configs as $shape) {
    $size = array_product($shape);
    echo str_repeat("=", 70) . "\n";
    echo sprintf("Array: %4d×%4d = %12d elementos\n", $shape[0], $shape[1], $size);
    echo str_repeat("=", 70) . "\n";
    
    // Test ADD
    $a = new \ZMatrix\ZTensor($shape);
    $b = new \ZMatrix\ZTensor($shape);
    
    echo "\n[ADD]\n";
    $start = microtime(true);
    for ($i = 0; $i < 50; $i++) {
        $a->add($b);
    }
    $total = (microtime(true) - $start) * 1000;
    $avg = $total / 50;
    $gflops = ($size / ($avg / 1000)) / 1e9;
    printf("  Per op: %.6f ms | Throughput: %.2f Gflops/s\n", $avg, $gflops);
    
    // Test MULTIPLY
    $a = new \ZMatrix\ZTensor($shape);
    $b = new \ZMatrix\ZTensor($shape);
    
    echo "\n[MUL]\n";
    $start = microtime(true);
    for ($i = 0; $i < 50; $i++) {
        $a->mul($b);
    }
    $total = (microtime(true) - $start) * 1000;
    $avg = $total / 50;
    $gflops = ($size / ($avg / 1000)) / 1e9;
    printf("  Per op: %.6f ms | Throughput: %.2f Gflops/s\n", $avg, $gflops);
    
    // Test SUBTRACT
    $a = new \ZMatrix\ZTensor($shape);
    $b = new \ZMatrix\ZTensor($shape);
    
    echo "\n[SUB]\n";
    $start = microtime(true);
    for ($i = 0; $i < 50; $i++) {
        $a->sub($b);
    }
    $total = (microtime(true) - $start) * 1000;
    $avg = $total / 50;
    $gflops = ($size / ($avg / 1000)) / 1e9;
    printf("  Per op: %.6f ms | Throughput: %.2f Gflops/s\n", $avg, $gflops);
    
    echo "\n";
}

echo "Done!\n";
