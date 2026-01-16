<?php

use ZMatrix\ZTensor;

echo "\n=== DIA 4: Extended SIMD Performance (PHP) ===\n\n";

$shape = [2500, 2500];
$size = array_product($shape);

// Test ABS
echo "[ABS]\n";
$a = ZTensor::full($shape, -2.5);  // Valores negativos
$start = microtime(true);
for ($i = 0; $i < 50; $i++) {
    $a->abs();
}
$total = (microtime(true) - $start) * 1000;
$avg = $total / 50;
printf("  50 ops: %.4f ms | per op: %.6f ms\n\n", $total, $avg);

// Verify correctness
$mean = $a->mean();
echo "  Mean of abs(-2.5) = " . $mean . " (expected ~2.5)\n\n";

// Test SQRT
echo "[SQRT]\n";
$b = ZTensor::full($shape, 4.0);  // Valores positivos

// Fazer sqrt uma única vez
$b->sqrt();
echo "  Single sqrt(4.0) = " . $b->mean() . " (expected 2.0)\n";

// Agora testar performance com reset
$b = ZTensor::full($shape, 4.0);
$start = microtime(true);
for ($i = 0; $i < 50; $i++) {
    $c = ZTensor::full($shape, 4.0);  // Reset cada iteração
    $c->sqrt();
}
$total = (microtime(true) - $start) * 1000;
$avg = $total / 50;
printf("  50 ops (com reset): %.4f ms | per op: %.6f ms\n\n", $total, $avg);

echo "Done!\n";
