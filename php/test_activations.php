<?php

echo "\n=== TESTE DIA 3: Funções de Ativação com SIMD ===\n\n";

$shape = [2500, 2500];
$size = array_product($shape);

// Test ReLU
echo "[ReLU]\n";
$a = new \ZMatrix\ZTensor($shape);
$start = microtime(true);
for ($i = 0; $i < 50; $i++) {
    $a->relu();
}
$total = (microtime(true) - $start) * 1000;
$avg = $total / 50;
printf("  50 ops: %.4f ms | per op: %.6f ms\n\n", $total, $avg);

// Test Sigmoid
echo "[Sigmoid]\n";
$a = new \ZMatrix\ZTensor($shape);
$start = microtime(true);
for ($i = 0; $i < 50; $i++) {
    $a->sigmoid();
}
$total = (microtime(true) - $start) * 1000;
$avg = $total / 50;
printf("  50 ops: %.4f ms | per op: %.6f ms\n\n", $total, $avg);

// Test Tanh
echo "[Tanh]\n";
$a = new \ZMatrix\ZTensor($shape);
$start = microtime(true);
for ($i = 0; $i < 50; $i++) {
    $a->tanh();
}
$total = (microtime(true) - $start) * 1000;
$avg = $total / 50;
printf("  50 ops: %.4f ms | per op: %.6f ms\n\n", $total, $avg);

echo "Done!\n";
