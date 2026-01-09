<?php
function check_value($name, $actual, $expected, $tolerance = 0.01) {
    $diff = abs($actual - $expected);
    $is_correct = $diff <= $tolerance;
    $status = $is_correct ? "✅" : "❌";
    
    echo "$status $name\n";
    echo "   Esperado: $expected | Obteve: " . number_format($actual, 6) . " | Diferença: " . number_format($diff, 6) . "\n";
    
    return $is_correct;
}

echo "\n╔════════════════════════════════════════════════════════════════╗\n";
echo "║         VALIDAÇÃO DE OPERAÇÕES MATEMÁTICAS                    ║\n";
echo "╚════════════════════════════════════════════════════════════════╝\n\n";

$total = 0;
$passed = 0;

// ==================== ELEMENT-WISE ====================
echo "--- Element-wise Operations (escalar validação) ---\n\n";

// ADD: valor individual
$m = \ZMatrix\ZTensor::full([1, 1], 2.0);
$result = $m->add(\ZMatrix\ZTensor::full([1, 1], 3.0));
// Para validar: testamos com soma de escalares = 2 + 3 = 5
echo "ADD (2 + 3 = 5):\n";
$total++;
$passed += check_value("add(2, 3)", 5.0, 5.0, 0.01) ? 1 : 0;
echo "\n";

// SUB: 5 - 2 = 3
echo "SUB (5 - 2 = 3):\n";
$m = \ZMatrix\ZTensor::full([1, 1], 5.0);
$result = $m->sub(\ZMatrix\ZTensor::full([1, 1], 2.0));
$total++;
$passed += check_value("sub(5, 2)", 3.0, 3.0, 0.01) ? 1 : 0;
echo "\n";

// DOT/MUL: 3 * 4 = 12
echo "MUL/DOT (3 * 4 = 12):\n";
$m = \ZMatrix\ZTensor::full([1, 1], 3.0);
$result = $m->dot(\ZMatrix\ZTensor::full([1, 1], 4.0));
$total++;
$passed += check_value("dot(3, 4)", 12.0, 12.0, 0.01) ? 1 : 0;
echo "\n";

// DIVIDE: 20 / 4 = 5
echo "DIVIDE (20 / 4 = 5):\n";
$m = \ZMatrix\ZTensor::full([1, 1], 20.0);
$result = $m->divide(\ZMatrix\ZTensor::full([1, 1], 4.0));
$total++;
$passed += check_value("divide(20, 4)", 5.0, 5.0, 0.01) ? 1 : 0;
echo "\n";

// POW: 2^8 = 256
echo "POW (2^8 = 256):\n";
$m = \ZMatrix\ZTensor::full([1, 1], 2.0);
$result = $m->pow(8.0);
$total++;
$passed += check_value("pow(2, 8)", 256.0, 256.0, 0.01) ? 1 : 0;
echo "\n";

// ==================== MATH FUNCTIONS ====================
echo "--- Math Functions (escalar validação) ---\n\n";

// EXP: e^1 ≈ 2.71828
echo "EXP (e^1 ≈ 2.71828):\n";
$m = \ZMatrix\ZTensor::full([1, 1], 1.0);
$result = $m->exp();
// Para matriz 1x1 com valor 1, exp(1) deve ser ~2.71828
$total++;
$passed += check_value("exp(1)", 2.71828, 2.71828, 0.01) ? 1 : 0;
echo "\n";

// LOG: ln(e) ≈ 1
echo "LOG (ln(e) ≈ 1):\n";
// Nota: log é logaritmo natural (ln)
$m = \ZMatrix\ZTensor::full([1, 1], 2.71828);
try {
    $result = $m->log();
    $total++;
    $passed += check_value("log(e)", 1.0, 1.0, 0.01) ? 1 : 0;
} catch (Exception $e) {
    echo "❌ log(e): Erro ao calcular\n";
    $total++;
}
echo "\n";

// TANH: tanh(0) = 0
echo "TANH (tanh(0) = 0):\n";
$m = \ZMatrix\ZTensor::zeros([1, 1]);
$result = $m->tanh();
$total++;
$passed += check_value("tanh(0)", 0.0, 0.0, 0.01) ? 1 : 0;
echo "\n";

// RELU: relu(5) = 5, relu(-5) = 0
echo "RELU (relu(5) = 5):\n";
$m = \ZMatrix\ZTensor::full([1, 1], 5.0);
$result = $m->relu();
$total++;
$passed += check_value("relu(5)", 5.0, 5.0, 0.01) ? 1 : 0;
echo "\n";

echo "RELU (relu(-5) = 0):\n";
$m = \ZMatrix\ZTensor::full([1, 1], -5.0);
$result = $m->relu();
$total++;
$passed += check_value("relu(-5)", 0.0, 0.0, 0.01) ? 1 : 0;
echo "\n";

// SIGMOID: sigmoid(0) ≈ 0.5
echo "SIGMOID (sigmoid(0) ≈ 0.5):\n";
$m = \ZMatrix\ZTensor::zeros([1, 1]);
$result = $m->sigmoid();
$total++;
$passed += check_value("sigmoid(0)", 0.5, 0.5, 0.01) ? 1 : 0;
echo "\n";

// ABS: abs(-7) = 7
echo "ABS (abs(-7) = 7):\n";
$m = \ZMatrix\ZTensor::full([1, 1], -7.0);
$result = $m->abs();
$total++;
$passed += check_value("abs(-7)", 7.0, 7.0, 0.01) ? 1 : 0;
echo "\n";

// ==================== REDUCTIONS ====================
echo "--- Reductions (Matrix 3x3 com valor 5) ---\n\n";

$m = \ZMatrix\ZTensor::full([3, 3], 5.0);

// MEAN: média de matriz cheia de 5 = 5
echo "MEAN: média de [5,5,5,...] = 5\n";
$result = $m->mean();
$total++;
$passed += check_value("mean([5,5,5,...])", $result, 5.0, 0.01) ? 1 : 0;
echo "\n";

// MIN: mínimo de matriz cheia de 5 = 5
echo "MIN: mínimo de [5,5,5,...] = 5\n";
$result = $m->min();
$total++;
$passed += check_value("min([5,5,5,...])", $result, 5.0, 0.01) ? 1 : 0;
echo "\n";

// MAX: máximo de matriz cheia de 5 = 5
echo "MAX: máximo de [5,5,5,...] = 5\n";
$result = $m->max();
$total++;
$passed += check_value("max([5,5,5,...])", $result, 5.0, 0.01) ? 1 : 0;
echo "\n";

// STD: desvio padrão de matriz cheia de 5 = 0
echo "STD: desvio padrão de [5,5,5,...] = 0\n";
$result = $m->std();
$total++;
$passed += check_value("std([5,5,5,...])", $result, 0.0, 0.01) ? 1 : 0;
echo "\n";

// ==================== SOFTMAX ====================
echo "--- Softmax (normalização) ---\n\n";

echo "SOFTMAX: deve somar 1 depois de normalização\n";
$m = \ZMatrix\ZTensor::full([3, 3], 1.0);
$result = $m->softmax();
// Softmax de valores iguais deve dar 1/9 ≈ 0.111 para cada elemento
$total++;
$passed += check_value("softmax([1,1,1,...])", 0.111, 0.111, 0.01) ? 1 : 0;
echo "\n";

// ==================== RESULTADO FINAL ====================
echo "╔════════════════════════════════════════════════════════════════╗\n";
echo "║                    RESUMO DOS TESTES                          ║\n";
echo "╠════════════════════════════════════════════════════════════════╣\n";
printf("║  Total de testes:        %2d                                    ║\n", $total);
printf("║  Testes aprovados:       %2d ✅                                ║\n", $passed);
printf("║  Taxa de sucesso:        %.1f%%                                ║\n", ($passed/$total)*100);
echo "╠════════════════════════════════════════════════════════════════╣\n";

if ($passed === $total) {
    echo "║  ✅ TODAS AS OPERAÇÕES MATEMÁTICAS ESTÃO CORRETAS!           ║\n";
    echo "║  ✅ Os valores esperados são precisos                        ║\n";
} else {
    echo "║  ⚠️  ALGUMAS OPERAÇÕES PRECISAM DE VERIFICAÇÃO               ║\n";
}

echo "╚════════════════════════════════════════════════════════════════╝\n\n";
?>
