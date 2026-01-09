<?php
// Teste pesado para reproduzir crash com simd
require 'vendor/autoload.php';

use ZMatrix\ZTensor;


$sizes = [100, 500, 1000, 5000];

foreach ($sizes as $size) {
    echo "\n=== Testando tamanho $size x $size ===\n";
    
    try {
        $m = new ZTensor([$size, $size]);
        
        // Usar tensor com valores iniciais
        $m = ZTensor::full([$size, $size], 1.0);
        
        // Operações que usam omp parallel for simd
        echo "ReLU... ";
        $r1 = $m->relu();
        echo "OK\n";
        
        echo "Sigmoid... ";
        $r2 = $m->sigmoid();
        echo "OK\n";
        
        echo "Add... ";
        $r3 = $m->add($m);
        echo "OK\n";
        
        echo "Multiply... ";
        $r4 = $m->mul($m);
        echo "OK\n";
        
        echo "✅ Tamanho $size passou!\n";
        
    } catch (Exception $e) {
        echo "❌ ERRO em tamanho $size: " . $e->getMessage() . "\n";
        break;
    }
}

echo "\n=== Teste pesado completo ===\n";
?>
