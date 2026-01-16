<?php
/**
 * TEST: Verificar bug de sum() retornando 50% do esperado
 * 
 * DESCRIÇÃO:
 * - ZTensor::full([N], valor) cria tensor com N elementos iguais a 'valor'
 * - A soma esperada é: N * valor
 * - BUG OBSERVADO: suma retorna aproximadamente 50% do esperado
 * 
 * EXEMPLO:
 * - ZTensor::full([100], 2.5)->sumtotal() retorna 130
 * - Esperado: 100 * 2.5 = 250
 * - Erro: 130 / 250 = 52% (faltando 48%)
 */

use ZMatrix\ZTensor;

echo "╔════════════════════════════════════════════════════════════╗\n";
echo "║        TESTE: sum() com HAS_AVX2=1 e HAS_OPENMP=1         ║\n";
echo "╚════════════════════════════════════════════════════════════╝\n\n";

$test_cases = [
    ['size' => [8], 'value' => 1.0, 'expected' => 8.0],
    ['size' => [16], 'value' => 1.0, 'expected' => 16.0],
    ['size' => [100], 'value' => 2.5, 'expected' => 250.0],
    ['size' => [256], 'value' => 0.5, 'expected' => 128.0],
    ['size' => [1024], 'value' => 1.0, 'expected' => 1024.0],
    ['size' => [2048], 'value' => 0.25, 'expected' => 512.0],
];

$passed = 0;
$failed = 0;

foreach ($test_cases as $test) {
    $t = ZTensor::full($test['size'], $test['value']);
    $result = $t->sumtotal();
    $expected = $test['expected'];
    
    // Converter array para validar dados
    $arr = $t->toArray();
    $arr_sum = array_sum($arr);
    
    // Erro em relação ao esperado
    $error = abs($result - $expected);
    $pct = ($error / $expected) * 100;
    
    // Status
    $is_ok = $error < 0.5;
    $status = $is_ok ? "OK" : "FAIL";
    $passed += $is_ok ? 1 : 0;
    $failed += !$is_ok ? 1 : 0;
    
    $sz = $test['size'][0];
    $val = $test['value'];
    printf("Size: %4d  Value: %.1f  Result: %8.1f  Expected: %8.1f  Error: %5.1f%%  %s\n", 
        $sz, $val, $result, $expected, $pct, $status);
    
    // Se falhar, mostrar que os dados estão corretos
    if (!$is_ok) {
        echo "  └─ toArray() sum: $arr_sum (correto!) -> Bug esta em sum()\n";
    }
}

echo "\n";
echo "===============================================================\n";
echo "Resultados: $passed passado, $failed falhado\n";
echo "===============================================================\n\n";

if ($failed > 0) {
    echo "AVISO: sum() retorna ~50% do esperado!\n";
    echo "Problema em sum_simd_kernel() ou funcao sum()\n";
} else {
    echo "OK: TODOS OS TESTES PASSARAM!\n";
}
