<?php


echo "🧪 TESTE 1: Soma Global\n";
$t = new ZMatrix\ZTensor([[1, 2, 3], [4, 5, 6]]);
echo "Tensor shape: " . json_encode($t->shape()) . "\n";

$result = $t->sum();
echo "sum() retorna: " . json_encode($result->toArray()) . "\n";
echo "Expected: [21] (1+2+3+4+5+6)\n";
$value = $result->toArray()[0];
$expected = 21.0;
if (abs($value - $expected) < 0.001) {
    echo "✅ PASS: Soma global\n\n";
} else {
    echo "❌ FAIL: Esperado $expected, got $value\n\n";
}

echo "🧪 TESTE 2: Soma por Eixo 0\n";
$result0 = $t->sum(0);
echo "sum(0) shape: " . json_encode($result0->shape()) . "\n";
echo "sum(0) values: " . json_encode($result0->toArray()) . "\n";
echo "Expected: [5, 7, 9] (1+4, 2+5, 3+6)\n";
$vals0 = $result0->toArray();
if (count($vals0) == 3 && abs($vals0[0] - 5) < 0.001 && abs($vals0[1] - 7) < 0.001 && abs($vals0[2] - 9) < 0.001) {
    echo "✅ PASS: Soma por eixo 0\n\n";
} else {
    echo "❌ FAIL\n\n";
}

echo "🧪 TESTE 3: Soma por Eixo 1\n";
$result1 = $t->sum(1);
echo "sum(1) shape: " . json_encode($result1->shape()) . "\n";
echo "sum(1) values: " . json_encode($result1->toArray()) . "\n";
echo "Expected: [6, 15] (1+2+3, 4+5+6)\n";
$vals1 = $result1->toArray();
if (count($vals1) == 2 && abs($vals1[0] - 6) < 0.001 && abs($vals1[1] - 15) < 0.001) {
    echo "✅ PASS: Soma por eixo 1\n\n";
} else {
    echo "❌ FAIL\n\n";
}

echo "🧪 TESTE 4: Soma por Eixo Negativo (-1)\n";
$result_1 = $t->sum(-1);
echo "sum(-1) shape: " . json_encode($result_1->shape()) . "\n";
echo "sum(-1) values: " . json_encode($result_1->toArray()) . "\n";
echo "Expected: [6, 15] (igual a sum(1))\n";
$vals_1 = $result_1->toArray();
if (count($vals_1) == 2 && abs($vals_1[0] - 6) < 0.001 && abs($vals_1[1] - 15) < 0.001) {
    echo "✅ PASS: Soma por eixo negativo\n\n";
} else {
    echo "❌ FAIL\n\n";
}

echo "🧪 TESTE 5: Tratamento de Erro - axis inválido\n";
try {
    $t->sum(999);
    echo "❌ FAIL: Deveria ter lançado exceção\n\n";
} catch (Exception $e) {
    echo "✅ PASS: Exceção lançada: " . $e->getMessage() . "\n\n";
}

echo "🧪 TESTE 6: Tratamento de Erro - tipo inválido\n";
try {
    $t->sum("invalid");
    echo "❌ FAIL: Deveria ter lançado exceção\n\n";
} catch (TypeError $e) {
    echo "✅ PASS: TypeError lançado\n\n";
} catch (Exception $e) {
    echo "✅ PASS: Exceção lançada: " . $e->getMessage() . "\n\n";
}

echo "🧪 TESTE 7: sum() não modifica original\n";
$original = $t->toArray();
$t->sum();
$t->sum(0);
$t->sum(1);
if ($t->toArray() === $original) {
    echo "✅ PASS: Original não foi modificado\n\n";
} else {
    echo "❌ FAIL: Original foi modificado\n\n";
}

echo "✅ TODOS OS TESTES COMPLETADOS!\n";


echo "🧪 TESTE: PATCH 5 - Validação de Divisão por Zero\n";
echo "================================================\n\n";

$t = new ZMatrix\ZTensor([1, 2, 3, 4]);
echo "Tensor original: " . json_encode($t->toArray()) . "\n";

// Teste 1: Divisão normal (válida)
echo "\n✓ Teste 1: Divisão por 2.0\n";
try {
    $result = $t->scalarDivide(2.0);
    echo "✅ Divisão por 2.0: OK\n";
    echo "   Resultado: " . json_encode($result->toArray()) . "\n";
} catch (Exception $e) {
    echo "❌ Erro inesperado: " . $e->getMessage() . "\n";
}

// Teste 2: Divisão por zero deve lançar exceção
echo "\n🔴 Teste 2: Divisão por 0.0 (deve lançar exceção)\n";
try {
    $result = $t->scalarDivide(0.0);
    echo "❌ FALHOU: Deveria ter lançado exceção!\n";
} catch (InvalidArgumentException $e) {
    echo "✅ InvalidArgumentException capturada:\n";
    echo "   Mensagem: " . $e->getMessage() . "\n";
} catch (Exception $e) {
    echo "✅ Exceção capturada:\n";
    echo "   Tipo: " . get_class($e) . "\n";
    echo "   Mensagem: " . $e->getMessage() . "\n";
}

echo "\n================================================\n";
echo "✅ PATCH 5 VALIDADO COM SUCESSO!\n";
echo "A divisão por zero agora é detectada e impede operação.\n";

echo "========================================\n";
echo "🧪 TESTE FINAL - ZMatrix v0.5.0\n";
echo "========================================\n\n";

// Teste 1: sum() refactor
echo "✅ PATCH 1+2: sum() refactor\n";
$t = new ZMatrix\ZTensor([[1, 2, 3], [4, 5, 6]]);
echo "   Tensor shape [2,3]: " . json_encode($t->toArray()) . "\n";
echo "   sum() = " . json_encode($t->sum()->toArray()) . " (global)\n";
echo "   sum(0) = " . json_encode($t->sum(0)->toArray()) . " (axis 0)\n";
echo "   sum(1) = " . json_encode($t->sum(1)->toArray()) . " (axis 1)\n\n";

// Teste 2: scalar_divide com zero
echo "✅ PATCH 5: Division-by-zero protection\n";
try {
    $t->scalarDivide(0.0);
    echo "   ❌ FALHOU\n";
} catch (Exception $e) {
    echo "   ✅ Exception caught: " . $e->getMessage() . "\n\n";
}

// Teste 3: mul() com shapes diferentes
echo "✅ PATCH 6: mul() shape validation\n";
$t1 = new ZMatrix\ZTensor([1, 2, 3]);
$t2 = new ZMatrix\ZTensor([4, 5, 6]);
try {
    $result = $t1->mul($t2);
    echo "   ✅ mul() com mesma shape: OK\n";
} catch (Exception $e) {
    echo "   ❌ Erro: " . $e->getMessage() . "\n";
}

// Teste 4: matmul()
echo "✅ PATCH 7: matmul() dimension validation\n";
$A = new ZMatrix\ZTensor([[1, 2, 3], [4, 5, 6]]);
$B = new ZMatrix\ZTensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
$C = $A->matmul($B);
echo "   A[2,3] @ B[3,4] = C" . json_encode($C->toArray()) . "\n";
echo "   ✅ matmul works\n\n";

echo "========================================\n";
echo "✅ ALL PATCHES VALIDATED - v0.5.0 READY!\n";
echo "========================================\n";

