<?php

use ZMatrix\ZTensor;

echo "\n=== STRESS TEST: Validação de Estabilidade ===\n\n";

// Teste 1: Múltiplas operações em sequência
echo "[1] Teste de Sequência (1000 ops)\n";
try {
    $a = new ZTensor([1000, 1000]);
    $b = new ZTensor([1000, 1000]);
    
    $ops = ['add', 'mul', 'sub', 'relu', 'sigmoid', 'tanh'];
    $start = microtime(true);
    
    for ($i = 0; $i < 1000; $i++) {
        $a->add($b);
        $a->mul($b);
        $a->sub($b);
        $a->relu();
        $a->sigmoid();
        $a->tanh();
    }
    
    $total = (microtime(true) - $start) * 1000;
    printf("  ✅ 6000 operações: %.2f ms (média: %.4f ms/op)\n\n", $total, $total / 6000);
} catch (Exception $e) {
    printf("  ❌ Erro: %s\n\n", $e->getMessage());
}

// Teste 2: Arrays grandes
echo "[2] Teste com Array Grande (5000×5000)\n";
try {
    $a = new ZTensor([5000, 5000]);
    $b = new ZTensor([5000, 5000]);
    
    $start = microtime(true);
    for ($i = 0; $i < 10; $i++) {
        $a->add($b);
    }
    $total = (microtime(true) - $start) * 1000;
    printf("  ✅ 10×add (25M floats): %.2f ms (média: %.4f ms)\n\n", $total, $total / 10);
} catch (Exception $e) {
    printf("  ❌ Erro: %s\n\n", $e->getMessage());
}

// Teste 3: Array pequeno (força SIMD sem OpenMP)
echo "[3] Teste com Array Pequeno (100×100, força SIMD)\n";
try {
    $a = new ZTensor([100, 100]);
    $b = new ZTensor([100, 100]);
    
    $start = microtime(true);
    for ($i = 0; $i < 1000; $i++) {
        $a->add($b);
    }
    $total = (microtime(true) - $start) * 1000;
    printf("  ✅ 1000×add (10k floats): %.2f ms (média: %.4f ms)\n\n", $total, $total / 1000);
} catch (Exception $e) {
    printf("  ❌ Erro: %s\n\n", $e->getMessage());
}

// Teste 4: Verificar valores corretos
echo "[4] Teste de Corretude\n";
try {
    // Criar tensores e inicializar com valores conhecidos
    $a = ZTensor::full([3, 3], 1.0);  // Preencher com 1.0
    $b = ZTensor::full([3, 3], 2.0);  // Preencher com 2.0
    
    // Operação conhecida: [1 1 1] + [2 2 2] = [3 3 3]
    $a->add($b);
    
    // Verificar resultado usando mean()
    // Esperado: média de 3.0 em todos os 9 elementos = 3.0
    $mean_val = $a->mean();
    $expected = 3.0;
    
    if (abs($mean_val - $expected) < 0.001) {
        printf("  ✅ Valores corretos (add: 1.0 + 2.0 = 3.0, mean = %.1f)\n\n", $mean_val);
    } else {
        printf("  ❌ Valor incorreto: esperado mean %.1f, obteve %.1f\n\n", $expected, $mean_val);
    }
} catch (Exception $e) {
    printf("  ❌ Erro: %s\n\n", $e->getMessage());
}

// Teste 5: Memory stability
echo "[5] Teste de Estabilidade de Memória\n";
try {
    $initial_mem = memory_get_usage();
    
    for ($i = 0; $i < 100; $i++) {
        $tmp = new ZTensor([1000, 1000]);
        $tmp->relu();
        unset($tmp);
    }
    
    $final_mem = memory_get_usage();
    $diff = ($final_mem - $initial_mem) / 1024 / 1024;
    
    if (abs($diff) < 10) {  // Menos de 10MB diferença
        printf("  ✅ Memória estável (diferença: %.2f MB)\n", $diff);
    } else {
        printf("  ⚠️  Possível leak (diferença: %.2f MB)\n", $diff);
    }
} catch (Exception $e) {
    printf("  ❌ Erro: %s\n", $e->getMessage());
}

echo "\n=== Stress Test Completo ===\n";
