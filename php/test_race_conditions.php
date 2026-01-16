<?php
require 'vendor/autoload.php';

// Teste para reproduzir race conditions com OpenMP SIMD

echo "=== Teste de Race Conditions com OpenMP SIMD ===\n\n";

// Teste 1: Operações simples (seguras)
echo "Teste 1: Operações simples (ReLU, Sigmoid, etc)\n";
try {
    $m = new \ZMatrix\ZTensor([1000, 1000]);
    for ($i = 0; $i < 5; $i++) {
        $r1 = $m->relu();
        $r2 = $m->sigmoid();
        $r3 = $m->tanh();
    }
    echo "✓ Sem crash\n";
} catch (Throwable $e) {
    echo "✗ CRASH: " . $e->getMessage() . "\n";
}

// Teste 2: Redução (sum, mean, std) - PERIGOSO
echo "\nTeste 2: Redução (operações de sum/mean)\n";
try {
    $m = \ZMatrix\ZTensor::ones([5000, 5000]);
    for ($i = 0; $i < 3; $i++) {
        // sum() faz redução em paralelo - pode ter race condition
        $s = $m->sumtotal();
        $avg = $m->mean();
    }
    echo "✓ Sem crash\n";
} catch (Throwable $e) {
    echo "✗ CRASH: " . $e->getMessage() . "\n";
}

// Teste 3: Múltiplas threads simultâneas
echo "\nTeste 3: Stress test - múltiplas operações\n";
try {
    $m = new \ZMatrix\ZTensor([2000, 2000]);
    
    for ($iter = 0; $iter < 10; $iter++) {
        $r1 = $m->relu();
        $r2 = $m->add($m);
        $r3 = $m->mul($m);
        $r4 = $m->sigmoid();
    }
    echo "✓ Sem crash\n";
} catch (Throwable $e) {
    echo "✗ CRASH: " . $e->getMessage() . "\n";
}

echo "\n=== Teste concluído ===\n";
?>
