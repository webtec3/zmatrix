<?php
// Teste com SIMD/OpenMP ativado
require 'vendor/autoload.php';

echo "=== BENCHMARK COM SIMD + OpenMP ATIVADO ===\n";
echo "PHP: " . phpversion() . "\n\n";
use ZMatrix\ZTensor;

// Teste 1: Addition
echo "Teste 1: Addition\n";
$m1 = ZTensor::zeros([100000]);
$m2 = ZTensor::ones([100000]);

$start = microtime(true);
for ($i = 0; $i < 50; $i++) {
    $result = $m1->add($m2);
}
$elapsed = (microtime(true) - $start) * 1000;
echo "50 iterações: " . round($elapsed, 2) . " ms\n";
echo "Por iteração: " . round($elapsed / 50, 4) . " ms\n\n";

// Teste 2: ReLU
echo "Teste 2: ReLU\n";
$m3 = ZTensor::randn([100000]);

$start = microtime(true);
for ($i = 0; $i < 50; $i++) {
    $result = $m3->relu();
}
$elapsed = (microtime(true) - $start) * 1000;
echo "50 iterações: " . round($elapsed, 2) . " ms\n";
echo "Por iteração: " . round($elapsed / 50, 4) . " ms\n\n";

// Teste 3: Tanh
echo "Teste 3: Tanh\n";
$m4 = ZTensor::randn([100000]);

$start = microtime(true);
for ($i = 0; $i < 50; $i++) {
    $result = $m4->tanh();
}
$elapsed = (microtime(true) - $start) * 1000;
echo "50 iterações: " . round($elapsed, 2) . " ms\n";
echo "Por iteração: " . round($elapsed / 50, 4) . " ms\n\n";

// Teste 4: Sigmoid
echo "Teste 4: Sigmoid\n";
$m5 = ZTensor::randn([100000]);

$start = microtime(true);
for ($i = 0; $i < 50; $i++) {
    $result = $m5->sigmoid();
}
$elapsed = (microtime(true) - $start) * 1000;
echo "50 iterações: " . round($elapsed, 2) . " ms\n";
echo "Por iteração: " . round($elapsed / 50, 4) . " ms\n\n";

// Teste 5: Subtração
echo "Teste 5: Subtração\n";
$start = microtime(true);
for ($i = 0; $i < 50; $i++) {
    $result = $m1->sub($m2);
}
$elapsed = (microtime(true) - $start) * 1000;
echo "50 iterações: " . round($elapsed, 2) . " ms\n";
echo "Por iteração: " . round($elapsed / 50, 4) . " ms\n\n";

echo "=== Testes concluídos ===\n";
?>
