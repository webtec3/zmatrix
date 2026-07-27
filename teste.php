<?php

use ZMatrix\ZTensor;

if (!extension_loaded('zmatrix')) {
    echo "❌ Extensão zmatrix não carregada!\n";
    exit(1);
}

echo "═══════════════════════════════════════════════════════════════════════\n";
echo "  GPU vs CPU BENCHMARK - Comparação Explícita com toGpu()\n";
echo "═══════════════════════════════════════════════════════════════════════\n\n";

// Teste com diferentes tamanhos
$tests = [
    ['name' => 'Pequeno (50K)', 'size' => 50_000, 'iter' => 20],
    ['name' => 'Médio (500K)', 'size' => 500_000, 'iter' => 10],
    ['name' => 'Grande (2M)', 'size' => 2_000_000, 'iter' => 3],
    ['name' => 'MuitoGrande (5M)', 'size' => 5_000_000, 'iter' => 2],
];

$results = [];

foreach ($tests as $test) {
    $name = $test['name'];
    $size = $test['size'];
    $iter = $test['iter'];

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    echo "Teste: $name (n=" . number_format($size) . " elementos)\n";
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    // Dados
    $a_data = array_fill(0, $size, 0.5);
    $b_data = array_fill(0, $size, 0.3);

    // ===== CPU BENCHMARK =====
    echo "  ⚙️  CPU (puro): ";
    flush();
    $a = ZTensor::arr($a_data);
    $b = ZTensor::arr($b_data); // FIX: era ZTensor::arr($a_data) — $b_data nunca era usado
    $start = microtime(true);
    for ($i = 0; $i < $iter; $i++) {

        $r1 = $a->add($b);
        $r2 = $r1->mul($b);
        $r3 = $r2->sub($b);
    }
    $time_cpu = (microtime(true) - $start) / $iter;
    $results[$name]['cpu'] = $time_cpu * 1000;

    printf("%.4f ms", $time_cpu * 1000);
    echo "\n";

    // ===== GPU BENCHMARK (com toGpu()) =====
    // NOTA: add/mul/sub são in-place — $r1/$r2/$r3 são o MESMO objeto que $a.
    // Isso significa que cada iteração do loop opera sobre o resultado da
    // iteração anterior, não sobre dados "frescos". Isso não afeta o tempo
    // medido (o custo por elemento independe do valor), mas vale saber.
    echo "  🎮 GPU (toGpu()): ";
    flush();

    $start = microtime(true);
    for ($i = 0; $i < $iter; $i++) {
        $a->toGpu();
        $b->toGpu();

        if (!$a->isOnGpu() || !$b->isOnGpu()) {
            echo "\n  ⚠️  AVISO: Dados não estão na GPU!\n";
        }

        $r1 = $a->add($b);
        $r2 = $r1->mul($b);
        $r3 = $r2->sub($b);
    }
    $time_gpu = (microtime(true) - $start) / $iter;
    $results[$name]['gpu'] = $time_gpu * 1000;

    printf("%.4f ms", $time_gpu * 1000);
    echo "\n";

    $speedup = $time_cpu / $time_gpu;
    $results[$name]['speedup'] = $speedup;

    echo "  📊 Resultado: ";
    if ($speedup >= 1.2) {
        printf("GPU %.2fx mais rápido ✅\n", $speedup);
    } elseif ($speedup >= 1.0) {
        printf("GPU %.2fx mais rápido ✓\n", $speedup);
    } else {
        printf("CPU %.2fx mais rápido (GPU overhead) ⚠️\n", 1.0 / $speedup);
    }

    echo "\n";
}

echo "\n" . str_repeat("═", 75) . "\n";
echo "COMPARATIVO FINAL\n";
echo str_repeat("═", 75) . "\n\n";

echo "┌──────────────────┬─────────────┬─────────────┬──────────────┐\n";
echo "│ Tamanho          │ CPU (ms)    │ GPU (ms)    │ Speedup      │\n";
echo "├──────────────────┼─────────────┼─────────────┼──────────────┤\n";

$total_speedup = 0;
$count = 0;

foreach ($results as $name => $data) {
    $speedup = $data['speedup'];
    if ($speedup >= 1.0) {
        $speedup_str = sprintf("GPU %.2fx", $speedup);
    } else {
        $speedup_str = sprintf("CPU %.2fx", 1.0 / $speedup);
    }

    printf("│ %-16s │ %11.4f │ %11.4f │ %-12s │\n",
        substr($name, 0, 16),
        $data['cpu'],
        $data['gpu'],
        $speedup_str
    );

    $total_speedup += $speedup;
    $count++;
}

echo "└──────────────────┴─────────────┴─────────────┴──────────────┘\n";

if ($count > 0) {
    $avg_speedup = $total_speedup / $count;
    echo "\n📈 Speedup médio GPU: " . sprintf("%.2fx", $avg_speedup) . "\n";

    if ($avg_speedup >= 2.0) {
        echo "   Status: GPU EXCELENTE ⭐⭐⭐\n";
    } elseif ($avg_speedup >= 1.5) {
        echo "   Status: GPU BOM ⭐⭐\n";
    } elseif ($avg_speedup >= 1.0) {
        echo "   Status: GPU ÚTIL ⭐\n";
    } else {
        echo "   Status: GPU COM OVERHEAD ⚠️\n";
        echo "   Dica: GPU pode ter overhead de transferência de dados\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════
// NOVO: Checagem de CORREÇÃO (não só velocidade) — CPU vs GPU
// devem produzir o MESMO resultado numérico, dentro da tolerância de float32.
// ═══════════════════════════════════════════════════════════════════════
echo "\n--- TESTE DE CORREÇÃO: CPU vs GPU produzem o mesmo resultado? ---\n\n";

$sizeCheck = 10_000;
$aa_data = array_fill(0, $sizeCheck, 0.7);
$bb_data = array_fill(0, $sizeCheck, 0.2);

$aa_cpu = ZTensor::arr($aa_data);
$bb_cpu = ZTensor::arr($bb_data);
$aa_cpu->add($bb_cpu)->mul($bb_cpu)->sub($bb_cpu);
$cpu_result = $aa_cpu->toArray();

$aa_gpu = ZTensor::arr($aa_data);
$bb_gpu = ZTensor::arr($bb_data);
$aa_gpu->toGpu();
$bb_gpu->toGpu();
$aa_gpu->add($bb_gpu)->mul($bb_gpu)->sub($bb_gpu);
$gpu_result = $aa_gpu->toArray();

$maxDiff = 0.0;
foreach ($cpu_result as $i => $v) {
    $diff = abs($v - $gpu_result[$i]);
    if ($diff > $maxDiff) $maxDiff = $diff;
}
printf("Maior diferença absoluta CPU vs GPU: %.8f\n", $maxDiff);
if ($maxDiff < 1e-4) {
    echo "✅ CPU e GPU produzem o mesmo resultado (dentro da tolerância de float32).\n\n";
} else {
    echo "❌ CPU e GPU DIVERGEM! Isso indica um bug real no kernel GPU ou na sincronização host/device.\n\n";
}

echo "--- TESTANDO O MÉTODO COLUMN() ---\n\n";

$matriz = ZTensor::arr([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
]);

echo "Matriz Original:\n" . $matriz . "\n";

$coluna1 = $matriz->column(1);
echo "\nColuna de índice 1 (esperado: [2, 5, 8]):\n";
echo $coluna1 . "\n";
echo "Shape da coluna: [" . implode(", ", $coluna1->shape()) . "]\n";

if ($coluna1->toArray() === [2.0, 5.0, 8.0]) {
    echo "✅ Teste Básico PASSOU!\n\n";
} else {
    echo "❌ Teste Básico FALHOU!\n\n";
}

echo "Gerando matriz massiva (1000x1000)...\n";
$bigMat = ZTensor::zeros([1000, 1000]);
$start = microtime(true);
$bigCol = $bigMat->column(500);
$time = microtime(true) - $start;
echo "Extração da coluna 500 concluída em " . number_format($time * 1000, 4) . " ms!\n";
echo "Shape extraído: [" . implode(", ", $bigCol->shape()) . "]\n";
echo "✅ Teste Estresse PASSOU!\n\n";

echo "Testando exceções (esperamos mensagens amigáveis e não um Crash):\n";

try {
    $matriz->column(10);
    echo "❌ FALHOU: Deveria ter dado erro de limite.\n";
} catch (Exception $e) {
    echo "✅ Erro capturado com sucesso: " . $e->getMessage() . "\n";
}

try {
    $vetor1D = ZTensor::arr([1, 2, 3]);
    $vetor1D->column(0);
    echo "❌ FALHOU: Deveria ter dado erro de dimensão.\n";
} catch (Exception $e) {
    echo "✅ Erro capturado com sucesso: " . $e->getMessage() . "\n";
}

// NOVO: valida o fix de requires_grad em column()/row()/gather()
try {
    $rg = ZTensor::arr([[1.0, 2.0], [3.0, 4.0]])->requiresGrad(true);
    $rg->column(0);
    echo "❌ FALHOU: column() em tensor requires_grad=true deveria lançar exceção.\n";
} catch (Throwable $e) {
    echo "✅ Erro esperado capturado (column + requiresGrad): " . $e->getMessage() . "\n";
}

echo "\n🚀 TODOS OS TESTES DE COLUMN() CONCLUÍDOS COM SUCESSO!\n";


echo "--- TESTANDO O MÉTODO row() ---\n\n";
$matriz = ZTensor::arr([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
]);

echo "Matriz Original:\n" . $matriz . "\n";

$linha1 = $matriz->row(1);
echo "\nLinha de índice 1 (esperado: [4, 5, 6]):\n";
echo $linha1 . "\n";
echo "Shape da linha: [" . implode(", ", $linha1->shape()) . "]\n";

if ($linha1->toArray() === [4.0, 5.0, 6.0]) {
    echo "✅ Teste Básico PASSOU!\n\n";
} else {
    echo "❌ Teste Básico FALHOU!\n\n";
}

echo "Gerando matriz massiva (1000x1000)...\n";
$bigMat = ZTensor::zeros([1000, 1000]);
$start = microtime(true);
$bigRow = $bigMat->row(500);
$time = microtime(true) - $start;
echo "Extração da linha 500 concluída em " . number_format($time * 1000, 4) . " ms!\n";
echo "Shape extraído: [" . implode(", ", $bigRow->shape()) . "]\n";
echo "✅ Teste Estresse PASSOU!\n\n";

echo "Testando exceções (esperamos mensagens amigáveis e não um Crash):\n";

try {
    $matriz->row(10);
    echo "❌ FALHOU: Deveria ter dado erro de limite.\n";
} catch (Exception $e) {
    echo "✅ Erro capturado com sucesso: " . $e->getMessage() . "\n";
}

try {
    $vetor1D = ZTensor::arr([1, 2, 3]);
    $vetor1D->row(0);
    echo "❌ FALHOU: Deveria ter dado erro de dimensão.\n";
} catch (Exception $e) {
    echo "✅ Erro capturado com sucesso: " . $e->getMessage() . "\n";
}

echo "\n🚀 TODOS OS TESTES DE ROW() CONCLUÍDOS COM SUCESSO!\n";

echo "--- TESTANDO O MÉTODO gather() ---\n\n";
$matriz = ZTensor::arr([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
]);

echo "Matriz Original:\n" . $matriz . "\n";

$indices = [0, 2];
$resultado = $matriz->gather($indices);

echo "\nGather dos índices [0, 2] (esperado: [[1, 2, 3], [7, 8, 9]]):\n";
echo $resultado . "\n";
echo "Shape do resultado: [" . implode(", ", $resultado->shape()) . "]\n";

$esperado = [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]];
if ($resultado->toArray() === $esperado) {
    echo "✅ Teste Básico PASSOU!\n\n";
} else {
    echo "❌ Teste Básico FALHOU!\n\n";
}

$indicesBootstrap = [3, 1, 1];
$resultBootstrap = $matriz->gather($indicesBootstrap);
echo "Gather de Bootstrap [3, 1, 1]:\n" . $resultBootstrap . "\n";
if (count($resultBootstrap->shape()) === 2 && $resultBootstrap->shape()[0] === 3) {
    echo "✅ Teste de Bootstrap (ordem e repetição) PASSOU!\n\n";
}

echo "Testando exceções:\n";

try {
    $matriz->gather([0, 99]);
    echo "❌ FALHOU: Deveria ter dado erro de limite.\n";
} catch (Exception $e) {
    echo "✅ Erro capturado com sucesso: " . $e->getMessage() . "\n";
}

try {
    $vetor1D = ZTensor::arr([1, 2, 3]);
    $vetor1D->gather([0]);
    echo "❌ FALHOU: Deveria ter dado erro de dimensão.\n";
} catch (Exception $e) {
    echo "✅ Erro capturado com sucesso: " . $e->getMessage() . "\n";
}

echo "\n🚀 TODOS OS TESTES DE GATHER() CONCLUÍDOS COM SUCESSO!\n";

echo "--- TESTE PESADO: ARGSORT + GATHER (PIPELINE DE DECISION TREE) ---\n\n";

$rows = 100000;
$cols = 5;
echo "Gerando matriz de $rows x $cols...\n";
$data = [];
for ($i = 0; $i < $rows; $i++) {
    $data[] = [
        (float)rand(0, 1000),
        (float)rand(0, 1000),
        (float)rand(0, 1000),
        (float)rand(0, 1000),
        (float)rand(0, 1000)
    ];
}
$X = ZTensor::arr($data);

echo "Executando argsort na coluna 0...\n";
$start = microtime(true);
// Semântica numpy: argsort(0) num tensor 2D ordena CADA COLUNA
// independentemente e devolve shape (rows, cols) — não um vetor 1D de
// índices de linha. Para o pipeline de decision tree, o que realmente
// queremos é a ordenação de UMA coluna específica.
$indices2D = $X->argsort(0);
$timeArgsort = microtime(true) - $start;
echo "Argsort concluído em " . number_format($timeArgsort * 1000, 2) . " ms.\n";
echo "Shape do resultado do argsort: [" . implode(", ", $indices2D->shape()) . "] (esperado: [$rows, $cols])\n";

echo "Executando gather para reordenar a matriz original...\n";
$start = microtime(true);
// FIX: o script original fazia array_map('intval', $indices->toArray())
// direto sobre um resultado 2D. intval() de um array PHP não-vazio sempre
// retorna 1, então isso produzia [1,1,1,...,1] e o gather() virava
// "repita a linha 1 cem mil vezes" — o que trivialmente "parece ordenado"
// (sequência constante) mas não testava NADA. Extraímos a coluna certa
// do resultado do argsort antes de usar no gather:
$indicesCol0 = $indices2D->column(0);
$indicesArray = array_map('intval', $indicesCol0->toArray());

if (count($indicesArray) !== $rows) {
    echo "❌ FALHOU: quantidade de índices (" . count($indicesArray) . ") diferente de $rows.\n";
}

$X_sorted = $X->gather($indicesArray);
$timeGather = microtime(true) - $start;

echo "Gather concluído em " . number_format($timeGather * 1000, 2) . " ms.\n";

echo "Validando se a coluna 0 está ordenada...\n";
$col0_arr = $X_sorted->column(0)->toArray();

$isSorted = true;
$isConstant = true;
for ($i = 0; $i < count($col0_arr) - 1; $i++) {
    if ($col0_arr[$i] > $col0_arr[$i + 1]) {
        $isSorted = false;
    }
    if ($col0_arr[$i] !== $col0_arr[$i + 1]) {
        $isConstant = false;
    }
}

// NOVO: checagem extra — uma sequência constante "passaria" no teste de
// ordenação sem provar nada. Com dados aleatórios de rand(0,1000) em
// 100.000 linhas, uma coluna real e corretamente ordenada NÃO deve ser
// constante. Se for, é sinal de que o bug do intval(array) voltou.
if ($isSorted && !$isConstant) {
    echo "✅ Sucesso! A matriz foi reordenada corretamente (e não é uma sequência degenerada constante).\n";
} elseif ($isSorted && $isConstant) {
    echo "❌ SUSPEITO: sequência ordenada mas constante — provável repetição da mesma linha (bug do intval/array).\n";
} else {
    echo "❌ Falha! A matriz não está ordenada.\n";
}

echo "\nPerformance total (Argsort + Gather): " . number_format(($timeArgsort + $timeGather) * 1000, 2) . " ms.\n";

// NOVO: caminho rápido para o caso comum (ordenar por UMA feature só) —
// evita pagar o custo de ordenar as 5 colunas quando só uma interessa.
echo "\nComparação: caminho rápido column(0)->argsort() (1D, um único sort):\n";
$start = microtime(true);
$fastIndices = $X->column(0)->argsort();
$timeFast = microtime(true) - $start;
echo "column(0)->argsort() concluído em " . number_format($timeFast * 1000, 2) . " ms";
echo " (vs " . number_format($timeArgsort * 1000, 2) . " ms do argsort(0) completo em todas as colunas)\n";

echo "🚀 TESTE PESADO CONCLUÍDO!\n\n";

// NOVO: testa também o ramo axis=1 do argsort (cada LINHA ordenada)
echo "--- TESTANDO ARGSORT AXIS=1 (cada linha ordenada) ---\n\n";
$small = ZTensor::arr([
    [3.0, 1.0, 2.0],
    [9.0, 7.0, 8.0],
]);
$sortedRows = $small->argsort(1);
echo "Matriz original:\n" . $small . "\n";
echo "argsort(axis=1) (esperado por linha: [1,2,0] e [1,2,0]):\n" . $sortedRows . "\n";
$expectedRows = [[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]];
if ($sortedRows->toArray() === $expectedRows) {
    echo "✅ argsort(axis=1) PASSOU!\n\n";
} else {
    echo "❌ argsort(axis=1) FALHOU!\n\n";
}

echo "--- TESTE DE CARGA: OPERAÇÃO WHERE (REALISTA - OTIMIZADO) ---\n\n";

$rows = 1000000;
$cols = 10;
echo "A gerar matriz de $rows x $cols com dados aleatórios (C++ Native)...\n";

$startGen = microtime(true);
$X = ZTensor::random([$rows, $cols], 0.0, 1000.0);
$timeGen = microtime(true) - $startGen;

echo "Matriz gerada em " . number_format($timeGen * 1000, 2) . " ms.\n";

$featureIndex = 5;
$threshold = 500.0;

echo "A executar operação where() na coluna $featureIndex...\n";

$start = microtime(true);
$mask = $X->where($featureIndex, $threshold);
$time = microtime(true) - $start;

echo "Operação where() concluída em " . number_format($time * 1000, 2) . " ms.\n";

$maskArray = $mask->toArray();
$sum = array_sum($maskArray);

echo "Número de amostras que satisfazem o split (<= $threshold): " . $sum . " (" . number_format(($sum / $rows) * 100, 2) . "%)\n";
echo "Performance: " . number_format($rows / ($time * 1000), 2) . " milhões de linhas/segundo.\n";

if (count($maskArray) === $rows) {
    echo "✅ Teste concluído com sucesso: Máscara tem o tamanho correto.\n";
} else {
    echo "❌ Erro: O tamanho da máscara é inconsistente.\n";
}

// NOVO: exceções de where() que não eram testadas
echo "\nTestando exceções de where():\n";
try {
    $X->where(99, 500.0); // coluna fora dos limites
    echo "❌ FALHOU: Deveria ter dado erro de limite.\n";
} catch (Exception $e) {
    echo "✅ Erro capturado com sucesso: " . $e->getMessage() . "\n";
}
try {
    $vetor1D = ZTensor::arr([1.0, 2.0, 3.0]);
    $vetor1D->where(0, 1.0); // não é 2D
    echo "❌ FALHOU: Deveria ter dado erro de dimensão.\n";
} catch (Exception $e) {
    echo "✅ Erro capturado com sucesso: " . $e->getMessage() . "\n";
}

echo "\n🚀 TESTE FINALIZADO!\n";

// ═══════════════════════════════════════════════════════════════════════
// NOVO: MATMUL + TRANSPOSE (com valores conhecidos, verificação exata)
// ═══════════════════════════════════════════════════════════════════════
echo "\n--- TESTANDO matmul() E transpose() ---\n\n";

$m = ZTensor::arr([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]); // (2,3)
$mt = $m->transpose(); // esperado (3,2): [[1,4],[2,5],[3,6]]
$expectedT = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
echo "transpose():\n" . $mt . "\n";
echo ($mt->toArray() === $expectedT) ? "✅ transpose() PASSOU!\n\n" : "❌ transpose() FALHOU!\n\n";

$m2 = ZTensor::arr([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]); // (3,2)
$prod = $m->matmul($m2); // (2,3)x(3,2) = (2,2), esperado [[4,5],[10,11]]
$expectedProd = [[4.0, 5.0], [10.0, 11.0]];
echo "matmul():\n" . $prod . "\n";
echo ($prod->toArray() === $expectedProd) ? "✅ matmul() PASSOU!\n\n" : "❌ matmul() FALHOU!\n\n";

// ═══════════════════════════════════════════════════════════════════════
// NOVO: reshape() e slice() — slice testa especificamente start > 0,
// que era o caso que estava incorreto na implementação antiga ("view"
// que na verdade não aplicava o offset).
// ═══════════════════════════════════════════════════════════════════════
echo "--- TESTANDO reshape() E slice() ---\n\n";

$flat = ZTensor::arr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
$reshaped = $flat->reshape([2, 3]);
$expectedReshape = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
echo "reshape([2,3]):\n" . $reshaped . "\n";
echo ($reshaped->toArray() === $expectedReshape) ? "✅ reshape() PASSOU!\n\n" : "❌ reshape() FALHOU!\n\n";

$t = ZTensor::arr([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]); // (4,2)
$s = $t->slice(0, 1, 3); // deve pegar as linhas 1 e 2 (start=1, não 0!)
$expectedSlice = [[3.0, 4.0], [5.0, 6.0]];
echo "slice(axis=0, start=1, end=3) (testa especificamente start>0):\n" . $s . "\n";
echo ($s->toArray() === $expectedSlice) ? "✅ slice() com start>0 PASSOU!\n\n" : "❌ slice() com start>0 FALHOU! (Se isso falhar, o bug antigo do offset voltou.)\n\n";

// ═══════════════════════════════════════════════════════════════════════
// NOVO: operações matemáticas element-wise com valores conhecidos
// ═══════════════════════════════════════════════════════════════════════
echo "--- TESTANDO divide(), pow(), exp(), log(), sqrt() ---\n\n";

$num = ZTensor::arr([10.0, 20.0, 30.0]);
$den = ZTensor::arr([2.0, 4.0, 5.0]);
$div = $num->copy()->divide($den); // esperado [5,5,6]
echo "divide(): " . $div . " (esperado [5,5,6])\n";
echo ($div->toArray() === [5.0, 5.0, 6.0]) ? "✅ divide() PASSOU!\n\n" : "❌ divide() FALHOU!\n\n";

$base = ZTensor::arr([1.0, 2.0, 3.0]);
$sq = $base->copy()->pow(2.0); // esperado [1,4,9]
echo "pow(2): " . $sq . " (esperado [1,4,9])\n";
echo ($sq->toArray() === [1.0, 4.0, 9.0]) ? "✅ pow() PASSOU!\n\n" : "❌ pow() FALHOU!\n\n";

$zero = ZTensor::arr([0.0]);
$expZero = $zero->copy()->exp(); // e^0 = 1
echo "exp(0): " . $expZero . " (esperado [1])\n";
echo (abs($expZero->toArray()[0] - 1.0) < 1e-5) ? "✅ exp() PASSOU!\n\n" : "❌ exp() FALHOU!\n\n";

$logInput = ZTensor::arr([1.0, M_E]);
$logResult = $logInput->copy()->log(); // esperado [0, 1]
echo "log([1, e]): " . $logResult . " (esperado [0,1])\n";
$logArr = $logResult->toArray();
echo (abs($logArr[0]) < 1e-5 && abs($logArr[1] - 1.0) < 1e-5) ? "✅ log() PASSOU!\n\n" : "❌ log() FALHOU!\n\n";

$sqrtInput = ZTensor::arr([4.0, 9.0, 16.0]);
$sqrtResult = $sqrtInput->copy()->sqrt(); // esperado [2,3,4]
echo "sqrt([4,9,16]): " . $sqrtResult . " (esperado [2,3,4])\n";
echo ($sqrtResult->toArray() === [2.0, 3.0, 4.0]) ? "✅ sqrt() PASSOU!\n\n" : "❌ sqrt() FALHOU!\n\n";

// ═══════════════════════════════════════════════════════════════════════
// NOVO: funções de ativação + derivadas
// ═══════════════════════════════════════════════════════════════════════
echo "--- TESTANDO relu(), sigmoid(), tanh() (+ derivadas) ---\n\n";

$actInput = ZTensor::arr([-2.0, -1.0, 0.0, 1.0, 2.0]);

$reluResult = $actInput->copy()->relu(); // esperado [0,0,0,1,2]
echo "relu(): " . $reluResult . " (esperado [0,0,0,1,2])\n";
echo ($reluResult->toArray() === [0.0, 0.0, 0.0, 1.0, 2.0]) ? "✅ relu() PASSOU!\n\n" : "❌ relu() FALHOU!\n\n";

$sigZero = ZTensor::arr([0.0])->sigmoid(); // sigmoid(0) = 0.5
echo "sigmoid(0): " . $sigZero . " (esperado [0.5])\n";
echo (abs($sigZero->toArray()[0] - 0.5) < 1e-5) ? "✅ sigmoid() PASSOU!\n\n" : "❌ sigmoid() FALHOU!\n\n";

$tanhZero = ZTensor::arr([0.0])->tanh(); // tanh(0) = 0
echo "tanh(0): " . $tanhZero . " (esperado [0])\n";
echo (abs($tanhZero->toArray()[0]) < 1e-5) ? "✅ tanh() PASSOU!\n\n" : "❌ tanh() FALHOU!\n\n";

$reluDeriv = $actInput->copy()->reluDerivative(); // esperado [0,0,0,1,1]
echo "reluDerivative(): " . $reluDeriv . " (esperado [0,0,0,1,1])\n";
echo ($reluDeriv->toArray() === [0.0, 0.0, 0.0, 1.0, 1.0]) ? "✅ reluDerivative() PASSOU!\n\n" : "❌ reluDerivative() FALHOU!\n\n";

// ═══════════════════════════════════════════════════════════════════════
// NOVO: reduções globais
// ═══════════════════════════════════════════════════════════════════════
echo "--- TESTANDO sumtotal(), mean(), min(), max(), std() ---\n\n";

$red = ZTensor::arr([1.0, 2.0, 3.0, 4.0, 5.0]);
printf("sumtotal(): %.4f (esperado 15)\n", $red->sumtotal());
printf("mean(): %.4f (esperado 3)\n", $red->mean());
printf("min(): %.4f (esperado 1)\n", $red->min());
printf("max(): %.4f (esperado 5)\n", $red->max());
// std amostral: sqrt(sum((x-3)^2)/(5-1)) = sqrt((4+1+0+1+4)/4) = sqrt(2.5) ≈ 1.5811
printf("std(): %.4f (esperado ≈1.5811)\n\n", $red->std());

$sumOk = abs($red->sumtotal() - 15.0) < 1e-4;
$meanOk = abs($red->mean() - 3.0) < 1e-4;
$minOk = abs($red->min() - 1.0) < 1e-4;
$maxOk = abs($red->max() - 5.0) < 1e-4;
$stdOk = abs($red->std() - 1.5811) < 1e-3;
echo ($sumOk && $meanOk && $minOk && $maxOk && $stdOk) ? "✅ Reduções PASSARAM!\n\n" : "❌ Alguma redução FALHOU!\n\n";

// ═══════════════════════════════════════════════════════════════════════
// NOVO: greater()
// ═══════════════════════════════════════════════════════════════════════
echo "--- TESTANDO greater() ---\n\n";

function run_test(string $label, callable $fn): void
{
    try {
        $fn();
    } catch (Throwable $e) {
        echo "❌ ERRO em '$label': " . get_class($e) . ": " . $e->getMessage() . "\n\n";
    }
}

run_test("greater() em tensor 1D com array [0.5]", function () {
    $g = ZTensor::arr([0.1, 0.6, 0.9]);
    $mask = $g->greater([0.5]);
    echo "greater(1D, [0.5]): " . $mask . "\n";
    echo ($mask->toArray() === [0.0, 1.0, 1.0]) ? "✅ PASSOU!\n\n" : "⚠️  Resultado inesperado.\n\n";
});

run_test("greater() em tensor 1D com escalar float direto", function () {
    $g = ZTensor::arr([0.1, 0.6, 0.9]);
    $mask = $g->greater(0.5);
    echo "greater(1D, 0.5): " . $mask . "\n";
    echo ($mask->toArray() === [0.0, 1.0, 1.0]) ? "✅ PASSOU!\n\n" : "⚠️  Resultado inesperado.\n\n";
});

run_test("greater() em tensor 2D (N,1) com array [0.5] (uso real da classe Metric)", function () {
    $g2d = ZTensor::arr([[0.1], [0.6], [0.9]]); // shape (3,1)
    $mask = $g2d->greater([0.5]);
    echo "greater(2D (3,1), [0.5]): " . $mask . "\n";
    echo ($mask->toArray() === [[0.0], [1.0], [1.0]]) ? "✅ PASSOU!\n\n" : "⚠️  Resultado inesperado.\n\n";
});

// ═══════════════════════════════════════════════════════════════════════
// AUTOGRAD — isolado em try/catch para não depender do resultado de greater()
// ═══════════════════════════════════════════════════════════════════════
echo "--- TESTANDO AUTOGRAD (addAutograd, mulAutograd, sumAutograd, backward) ---\n\n";

run_test("autograd completo (mul + sum + backward + grad)", function () {
    $a = ZTensor::arr([1.0, 2.0, 3.0]);
    $a->requiresGrad(true);
    $b = ZTensor::arr([4.0, 5.0, 6.0]);
    $b->requiresGrad(true);

    echo "a.isRequiresGrad(): " . ($a->isRequiresGrad() ? 'true' : 'false') . " (esperado true)\n";

    $c = ZTensor::mulAutograd($a, $b);   // c = a * b
    $loss = ZTensor::sumAutograd($c);    // loss = sum(c), escalar

    $loss->backward();

    $gradA = $a->getGrad()->toArray();  // esperado: b = [4,5,6]
    $gradB = $b->getGrad()->toArray();  // esperado: a = [1,2,3]

    echo "grad(a) = [" . implode(", ", $gradA) . "] (esperado [4, 5, 6])\n";
    echo "grad(b) = [" . implode(", ", $gradB) . "] (esperado [1, 2, 3])\n";

    $gradAOk = $gradA == [4.0, 5.0, 6.0];
    $gradBOk = $gradB == [1.0, 2.0, 3.0];
    echo ($gradAOk && $gradBOk) ? "✅ Autograd (mul + sum + backward) PASSOU!\n\n" : "❌ Autograd FALHOU!\n\n";

    // zeroGrad()
    $a->zeroGrad();
    $gradAAfterZero = $a->getGrad()->toArray();
    $allZero = true;
    foreach ($gradAAfterZero as $v) {
        if (abs($v) > 1e-9) {
            $allZero = false;
            break;
        }
    }
    echo "Após zeroGrad(): grad(a) = [" . implode(", ", $gradAAfterZero) . "]\n";
    echo $allZero ? "✅ zeroGrad() PASSOU!\n\n" : "❌ zeroGrad() FALHOU!\n\n";

    // Bloqueio de operação in-place em tensor com requires_grad=true
    try {
        $a->add($b);
        echo "❌ FALHOU: add() in-place em tensor requires_grad=true deveria lançar exceção.\n";
    } catch (Throwable $e) {
        echo "✅ Erro esperado capturado (add in-place + requiresGrad): " . $e->getMessage() . "\n";
    }
});

echo "\n🚀 TODOS OS TESTES ADICIONAIS CONCLUÍDOS!\n";

function assertTensorEquals(
    array   $expected,
    ZTensor $actual,
    string  $message
): void
{
    $actualArray = $actual->toArray();

    if ($actualArray !== $expected) {
        echo "❌ {$message}\n";
        echo "Esperado: " . json_encode($expected) . "\n";
        echo "Recebido: " . json_encode($actualArray) . "\n";
        exit(1);
    }

    echo "✅ {$message}\n";
}

echo str_repeat('=', 60) . PHP_EOL;
echo "MODE TEST" . PHP_EOL;
echo str_repeat('=', 60) . PHP_EOL;

$global = ZTensor::arr([2, 1, 2, 3, 2, 1]);
assertTensorEquals([2.0], $global->mode(), 'Moda global');

$tie = ZTensor::arr([2, 2, 1, 1]);
assertTensorEquals([1.0], $tie->mode(), 'Desempate escolhe o menor valor');

$negative = ZTensor::arr([-2, -2, 1, 1, -2]);
assertTensorEquals([-2.0], $negative->mode(), 'Moda com valores negativos');

$floats = ZTensor::arr([1.5, 2.5, 1.5]);
assertTensorEquals([1.5], $floats->mode(), 'Moda com floats');

$rows = ZTensor::arr([
    [1, 2, 2],
    [3, 3, 1],
    [5, 4, 5],
]);
assertTensorEquals([2.0, 3.0, 5.0], $rows->mode(1), 'Mode axis=1');

$columns = ZTensor::arr([
    [1, 2, 2],
    [1, 3, 2],
    [4, 3, 2],
]);
assertTensorEquals([1.0, 3.0, 2.0], $columns->mode(0), 'Mode axis=0');

assertTensorEquals([2.0, 3.0, 5.0], $rows->mode(-1), 'Mode axis=-1');

$ensemble = ZTensor::arr([
    [0, 1, 2, 1],
    [0, 2, 2, 1],
    [1, 1, 0, 1],
    [0, 2, 0, 2],
]);
assertTensorEquals([0.0, 1.0, 0.0, 1.0], $ensemble->mode(0), 'Votação de ensemble');

try {
    ZTensor::arr([])->mode();
    echo "❌ Tensor vazio deveria lançar exceção\n";
    exit(1);
} catch (Throwable $exception) {
    echo "✅ Exceção para tensor vazio: " . $exception->getMessage() . PHP_EOL;
}

try {
    $rows->mode(2);
    echo "❌ Axis inválido deveria lançar exceção\n";
    exit(1);
} catch (Throwable $exception) {
    echo "✅ Exceção para axis inválido: " . $exception->getMessage() . PHP_EOL;
}

try {
    ZTensor::arr([1.0, NAN, 2.0])->mode();
    echo "❌ NaN deveria lançar exceção\n";
    exit(1);
} catch (Throwable $exception) {
    echo "✅ Exceção para NaN: " . $exception->getMessage() . PHP_EOL;
}

// NOVO: valida especificamente que a checagem de NaN funciona também no
// caminho por eixo (mode(axis)) — o pedido original menciona a validação
// serial explicitamente para esse caminho, então vale testar em separado.
try {
    ZTensor::arr([[1.0, NAN], [2.0, 3.0]])->mode(1);
    echo "❌ NaN em mode(axis) deveria lançar exceção\n";
    exit(1);
} catch (Throwable $exception) {
    echo "✅ Exceção para NaN em mode(axis): " . $exception->getMessage() . PHP_EOL;
}

// NOVO: valida que a moda global retornada por mode() (sem axis) tem shape
// [1], não um float solto — é a regra explícita da API PHP (diferente do
// método C++ mode() que retorna float puro).
$shapeCheck = ZTensor::arr([1, 1, 2]);
$modeShape = $shapeCheck->mode()->shape();
if ($modeShape === [1]) {
    echo "✅ mode() sem axis retorna shape [1]\n";
} else {
    echo "❌ mode() sem axis deveria retornar shape [1], recebeu [" . implode(",", $modeShape) . "]\n";
    exit(1);
}

echo PHP_EOL;
echo "🚀 TODOS OS TESTES DE MODE PASSARAM!" . PHP_EOL;


$a = new ZTensor([1000, 1000]);
$b = new ZTensor([1000, 1000]);

// Com GPU: ~2ms
// Com CPU: ~50ms

$tensor = ZTensor::arr([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]);
$reshaped = $tensor->reshape([4, 2]);
/*
|--------------------------------------------------------------------------
| Helpers
|--------------------------------------------------------------------------
*/

function flattenArray(array $array): array
{
    $result = [];

    array_walk_recursive(
        $array,
        static function (mixed $value) use (&$result): void {
            $result[] = (float) $value;
        }
    );

    return $result;
}

function formatArray(array $array): string
{
    return json_encode(
        $array,
        JSON_UNESCAPED_UNICODE | JSON_PRESERVE_ZERO_FRACTION
    );
}

function assertTrue(bool $condition, string $label): void
{
    if (!$condition) {
        throw new RuntimeException("❌ {$label}");
    }

    echo "✅ {$label}\n";
}

function assertSameArray(
    array $expected,
    array $actual,
    string $label
): void {
    if ($expected !== $actual) {
        throw new RuntimeException(
            sprintf(
                "❌ %s\nEsperado: %s\nObtido:  %s",
                $label,
                formatArray($expected),
                formatArray($actual)
            )
        );
    }

    echo "✅ {$label}\n";
}

function assertArrayClose(
    array $expected,
    array $actual,
    string $label,
    float $epsilon = 1e-5
): void {
    $expectedFlat = flattenArray($expected);
    $actualFlat = flattenArray($actual);

    if (count($expectedFlat) !== count($actualFlat)) {
        throw new RuntimeException(
            sprintf(
                "❌ %s: esperado %d elementos, obtido %d",
                $label,
                count($expectedFlat),
                count($actualFlat)
            )
        );
    }

    foreach ($expectedFlat as $index => $expectedValue) {
        $actualValue = $actualFlat[$index];

        if (!is_finite($actualValue)) {
            throw new RuntimeException(
                sprintf(
                    "❌ %s: valor não finito no índice %d",
                    $label,
                    $index
                )
            );
        }

        if (abs($expectedValue - $actualValue) > $epsilon) {
            throw new RuntimeException(
                sprintf(
                    "❌ %s: índice %d, esperado %.8f, obtido %.8f",
                    $label,
                    $index,
                    $expectedValue,
                    $actualValue
                )
            );
        }
    }

    echo "✅ {$label}\n";
}

function assertShape(
    array $expectedShape,
    ZTensor $tensor,
    string $label
): void {
    assertSameArray(
        $expectedShape,
        $tensor->shape(),
        $label
    );
}

function assertTensorClose(
    array $expected,
    ZTensor $actual,
    string $label,
    float $epsilon = 1e-5
): void {
    assertArrayClose(
        $expected,
        $actual->toArray(),
        $label,
        $epsilon
    );
}

function assertThrows(
    callable $callback,
    string $label
): void {
    try {
        $callback();
    } catch (Throwable $exception) {
        echo "✅ {$label}: {$exception->getMessage()}\n";
        return;
    }

    throw new RuntimeException(
        "❌ {$label}: nenhuma exceção foi lançada"
    );
}

function section(string $title): void
{
    echo "\n";
    echo "==================================================\n";
    echo "{$title}\n";
    echo "==================================================\n";
}

/*
|--------------------------------------------------------------------------
| 1. permute()
|--------------------------------------------------------------------------
*/

section('1. permute');

$input3d = ZTensor::arr([
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ],
    [
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ],
]); // [2, 2, 3]

$input3dBefore = $input3d->toArray();

$permuted = $input3d->permute([2, 0, 1]);

assertShape(
    [3, 2, 2],
    $permuted,
    'permute produz o shape esperado'
);

assertTensorClose([
    [
        [1.0, 4.0],
        [7.0, 10.0],
    ],
    [
        [2.0, 5.0],
        [8.0, 11.0],
    ],
    [
        [3.0, 6.0],
        [9.0, 12.0],
    ],
], $permuted, 'permute reorganiza corretamente os valores');

/*
 * [-1, 0, 1] equivale a [2, 0, 1].
 */

$permutedNegative = $input3d->permute([-1, 0, 1]);

assertShape(
    [3, 2, 2],
    $permutedNegative,
    'permute aceita eixo negativo'
);

assertTensorClose(
    $permuted->toArray(),
    $permutedNegative,
    'permute negativo equivale ao eixo positivo'
);

/*
 * Permutação identidade.
 */

$permutedIdentity = $input3d->permute([0, 1, 2]);

assertShape(
    [2, 2, 3],
    $permutedIdentity,
    'permute identidade preserva o shape'
);

assertTensorClose(
    $input3dBefore,
    $permutedIdentity,
    'permute identidade preserva os valores'
);

assertArrayClose(
    $input3dBefore,
    $input3d->toArray(),
    'permute não altera o tensor original'
);

/*
|--------------------------------------------------------------------------
| 2. flatten()
|--------------------------------------------------------------------------
*/

section('2. flatten');

$flattenMiddle = $input3d->flatten(1, 2);

assertShape(
    [2, 6],
    $flattenMiddle,
    'flatten(1, 2) produz shape [2, 6]'
);

assertTensorClose([
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
], $flattenMiddle, 'flatten preserva a ordem dos elementos');

$flattenAll = $input3d->flatten();

assertShape(
    [12],
    $flattenAll,
    'flatten padrão combina todas as dimensões'
);

assertTensorClose([
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
], $flattenAll, 'flatten completo preserva a ordem linear');

$flattenNegative = $input3d->flatten(0, -2);

assertShape(
    [4, 3],
    $flattenNegative,
    'flatten aceita endAxis negativo'
);

assertTensorClose([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0],
], $flattenNegative, 'flatten com eixo negativo produz valores corretos');

/*
 * Achatar somente um eixo não deve alterar o shape.
 */

$flattenSingleAxis = $input3d->flatten(1, 1);

assertShape(
    [2, 2, 3],
    $flattenSingleAxis,
    'flatten de um único eixo preserva o shape'
);

assertTensorClose(
    $input3dBefore,
    $flattenSingleAxis,
    'flatten de um único eixo preserva os valores'
);

assertArrayClose(
    $input3dBefore,
    $input3d->toArray(),
    'flatten não altera o tensor original'
);

/*
|--------------------------------------------------------------------------
| 3. broadcastTo()
|--------------------------------------------------------------------------
*/

section('3. broadcastTo');

$broadcastInput = ZTensor::arr([
    [1.0, 2.0, 3.0],
]); // [1, 3]

$broadcastInputBefore = $broadcastInput->toArray();

$broadcasted = $broadcastInput->broadcastTo([2, 2, 3]);

assertShape(
    [2, 2, 3],
    $broadcasted,
    'broadcastTo produz o shape solicitado'
);

assertTensorClose([
    [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ],
    [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ],
], $broadcasted, 'broadcastTo replica corretamente os valores');

/*
 * Broadcasting de [2, 1] para [2, 3].
 */

$broadcastColumn = ZTensor::arr([
    [10.0],
    [20.0],
]);

$broadcastColumnResult = $broadcastColumn->broadcastTo([2, 3]);

assertShape(
    [2, 3],
    $broadcastColumnResult,
    'broadcastTo expande dimensão unitária'
);

assertTensorClose([
    [10.0, 10.0, 10.0],
    [20.0, 20.0, 20.0],
], $broadcastColumnResult, 'broadcastTo expande a dimensão correta');

assertArrayClose(
    $broadcastInputBefore,
    $broadcastInput->toArray(),
    'broadcastTo preserva o tensor original'
);

/*
|--------------------------------------------------------------------------
| 4. im2col()
|--------------------------------------------------------------------------
|
| Contrato:
|
| NCHW -> [N, C*KH*KW, OH*OW]
|
|--------------------------------------------------------------------------
*/

section('4. im2col');

$image3x3 = ZTensor::arr([
    [[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]],
]); // [1, 1, 3, 3]

$image3x3Before = $image3x3->toArray();

$columns = $image3x3->im2col(
    2,
    2,
    1,
    1,
    0,
    0
);

/*
 * OH = ((3 - 2) / 1) + 1 = 2
 * OW = ((3 - 2) / 1) + 1 = 2
 *
 * [N, C*KH*KW, OH*OW]
 * [1, 1*2*2, 2*2]
 * [1, 4, 4]
 */

assertShape(
    [1, 4, 4],
    $columns,
    'im2col retorna [N, C*KH*KW, OH*OW]'
);

assertTensorClose([
    [
        [1.0, 2.0, 4.0, 5.0],
        [2.0, 3.0, 5.0, 6.0],
        [4.0, 5.0, 7.0, 8.0],
        [5.0, 6.0, 8.0, 9.0],
    ],
], $columns, 'im2col extrai corretamente os patches');

/*
 * Teste com stride 2.
 */

$image4x4 = ZTensor::arr([
    [[
        [1.0,  2.0,  3.0,  4.0],
        [5.0,  6.0,  7.0,  8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]],
]);

$columnsStride = $image4x4->im2col(
    2,
    2,
    2,
    2,
    0,
    0
);

assertShape(
    [1, 4, 4],
    $columnsStride,
    'im2col com stride produz o shape correto'
);

assertTensorClose([
    [
        [1.0, 3.0, 9.0, 11.0],
        [2.0, 4.0, 10.0, 12.0],
        [5.0, 7.0, 13.0, 15.0],
        [6.0, 8.0, 14.0, 16.0],
    ],
], $columnsStride, 'im2col com stride extrai os patches corretos');

/*
 * Teste com múltiplos canais e kernel 1x1.
 */

$multiChannelImage = ZTensor::arr([
    [
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
    ],
]); // [1, 2, 2, 2]

$multiChannelColumns = $multiChannelImage->im2col(
    1,
    1,
    1,
    1,
    0,
    0
);

assertShape(
    [1, 2, 4],
    $multiChannelColumns,
    'im2col suporta múltiplos canais'
);

assertTensorClose([
    [
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ],
], $multiChannelColumns, 'im2col preserva a ordem dos canais');

assertArrayClose(
    $image3x3Before,
    $image3x3->toArray(),
    'im2col preserva o tensor original'
);

/*
|--------------------------------------------------------------------------
| 5. col2im()
|--------------------------------------------------------------------------
*/

section('5. col2im');

/*
 * Com overlap, as contribuições devem ser somadas.
 */

$ones3x3 = ZTensor::full([1, 1, 3, 3], 1.0);

$overlapColumns = $ones3x3->im2col(
    2,
    2,
    1,
    1,
    0,
    0
);

$overlapReconstructed = $overlapColumns->col2im(
    [1, 1, 3, 3],
    2,
    2,
    1,
    1,
    0,
    0
);

assertShape(
    [1, 1, 3, 3],
    $overlapReconstructed,
    'col2im recupera o shape NCHW'
);

assertTensorClose([
    [[
        [1.0, 2.0, 1.0],
        [2.0, 4.0, 2.0],
        [1.0, 2.0, 1.0],
    ]],
], $overlapReconstructed, 'col2im acumula regiões sobrepostas');

/*
 * Sem overlap:
 *
 * col2im(im2col(x)) deve recuperar exatamente x.
 */

$nonOverlapColumns = $image4x4->im2col(
    2,
    2,
    2,
    2,
    0,
    0
);

$nonOverlapReconstructed = $nonOverlapColumns->col2im(
    [1, 1, 4, 4],
    2,
    2,
    2,
    2,
    0,
    0
);

assertShape(
    [1, 1, 4, 4],
    $nonOverlapReconstructed,
    'col2im sem overlap recupera o shape original'
);

assertTensorClose(
    $image4x4->toArray(),
    $nonOverlapReconstructed,
    'col2im inverte im2col quando não existe overlap'
);

/*
|--------------------------------------------------------------------------
| 6. conv2d()
|--------------------------------------------------------------------------
|
| A stub declara cross-correlation NCHW com filtros OIHW.
|
|--------------------------------------------------------------------------
*/

section('6. conv2d');

/*
 * Identidade 1x1.
 */

$convInput = ZTensor::arr([
    [[
        [1.0, 2.0],
        [3.0, 4.0],
    ]],
]);

$convInputBefore = $convInput->toArray();

$identityFilter = ZTensor::arr([
    [[
        [1.0],
    ]],
]);

$identityFilterBefore = $identityFilter->toArray();

$identityOutput = $convInput->conv2d($identityFilter);

assertShape(
    [1, 1, 2, 2],
    $identityOutput,
    'conv2d identidade mantém o shape'
);

assertTensorClose(
    $convInput->toArray(),
    $identityOutput,
    'conv2d com kernel 1x1 igual a 1 preserva a entrada'
);

/*
 * Confirma cross-correlation.
 *
 * Entrada:
 *
 * 1 2 3
 * 4 5 6
 * 7 8 9
 *
 * Filtro:
 *
 * 1 2
 * 3 4
 *
 * Saída:
 *
 * 37 47
 * 67 77
 */

$correlationFilter = ZTensor::arr([
    [[
        [1.0, 2.0],
        [3.0, 4.0],
    ]],
]);

$correlationOutput = $image3x3->conv2d($correlationFilter);

assertShape(
    [1, 1, 2, 2],
    $correlationOutput,
    'conv2d calcula corretamente o shape espacial'
);

assertTensorClose([
    [[
        [37.0, 47.0],
        [67.0, 77.0],
    ]],
], $correlationOutput, 'conv2d executa cross-correlation sem inverter o kernel');

/*
 * Bias.
 */

$bias = ZTensor::arr([10.0]);
$biasBefore = $bias->toArray();

$outputWithBias = $image3x3->conv2d(
    $correlationFilter,
    $bias
);

assertTensorClose([
    [[
        [47.0, 57.0],
        [77.0, 87.0],
    ]],
], $outputWithBias, 'conv2d adiciona bias ao canal de saída');

/*
 * Múltiplos canais.
 */

$sumChannelsFilter = ZTensor::arr([
    [
        [[1.0]],
        [[1.0]],
    ],
]); // [O=1, I=2, H=1, W=1]

$sumChannelsOutput = $multiChannelImage->conv2d(
    $sumChannelsFilter
);

assertShape(
    [1, 1, 2, 2],
    $sumChannelsOutput,
    'conv2d suporta múltiplos canais de entrada'
);

assertTensorClose([
    [[
        [11.0, 22.0],
        [33.0, 44.0],
    ]],
], $sumChannelsOutput, 'conv2d soma as contribuições dos canais');

/*
 * Múltiplos filtros de saída.
 */

$twoOutputFilters = ZTensor::arr([
    [
        [[1.0]],
        [[1.0]],
    ],
    [
        [[1.0]],
        [[-1.0]],
    ],
]); // [2, 2, 1, 1]

$twoOutputBias = ZTensor::arr([
    0.0,
    100.0,
]);

$twoOutputResult = $multiChannelImage->conv2d(
    $twoOutputFilters,
    $twoOutputBias
);

assertShape(
    [1, 2, 2, 2],
    $twoOutputResult,
    'conv2d produz um canal para cada filtro'
);

assertTensorClose([
    [
        [
            [11.0, 22.0],
            [33.0, 44.0],
        ],
        [
            [91.0, 82.0],
            [73.0, 64.0],
        ],
    ],
], $twoOutputResult, 'conv2d calcula múltiplos canais de saída');

/*
 * Stride 2.
 */

$sum2x2Filter = ZTensor::arr([
    [[
        [1.0, 1.0],
        [1.0, 1.0],
    ]],
]);

$strideOutput = $image4x4->conv2d(
    $sum2x2Filter,
    null,
    2,
    2,
    0,
    0
);

assertShape(
    [1, 1, 2, 2],
    $strideOutput,
    'conv2d respeita o stride'
);

assertTensorClose([
    [[
        [14.0, 22.0],
        [46.0, 54.0],
    ]],
], $strideOutput, 'conv2d com stride produz valores corretos');

assertArrayClose(
    $convInputBefore,
    $convInput->toArray(),
    'conv2d preserva o input'
);

assertArrayClose(
    $identityFilterBefore,
    $identityFilter->toArray(),
    'conv2d preserva os filtros'
);

assertArrayClose(
    $biasBefore,
    $bias->toArray(),
    'conv2d preserva o bias'
);

/*
|--------------------------------------------------------------------------
| 7. conv2dBackward()
|--------------------------------------------------------------------------
*/

section('7. conv2dBackward');

/*
 * Caso analítico simples:
 *
 * output = input * 2
 * loss   = sum(output)
 *
 * gradOutput = 1
 *
 * gradInput   = 2 em todas as posições
 * gradFilter  = sum(input) = 10
 * gradBias    = 4
 */

$backwardInput = ZTensor::arr([
    [[
        [1.0, 2.0],
        [3.0, 4.0],
    ]],
]);

$backwardFilter = ZTensor::arr([
    [[
        [2.0],
    ]],
]);

$gradOutput = ZTensor::full(
    [1, 1, 2, 2],
    1.0
);

$backwardInputBefore = $backwardInput->toArray();
$backwardFilterBefore = $backwardFilter->toArray();
$gradOutputBefore = $gradOutput->toArray();

$gradients = $backwardInput->conv2dBackward(
    $gradOutput,
    $backwardFilter
);

assertTrue(
    count($gradients) === 3,
    'conv2dBackward retorna três tensores'
);

assertTrue(
    $gradients[0] instanceof ZTensor,
    'conv2dBackward retorna gradInput como ZTensor'
);

assertTrue(
    $gradients[1] instanceof ZTensor,
    'conv2dBackward retorna gradFilters como ZTensor'
);

assertTrue(
    $gradients[2] instanceof ZTensor,
    'conv2dBackward retorna gradBias como ZTensor'
);

[$gradInput, $gradFilters, $gradBias] = $gradients;

assertShape(
    [1, 1, 2, 2],
    $gradInput,
    'gradInput possui o shape do input'
);

assertShape(
    [1, 1, 1, 1],
    $gradFilters,
    'gradFilters possui o shape dos filtros'
);

assertShape(
    [1],
    $gradBias,
    'gradBias possui um valor por canal de saída'
);

assertTensorClose([
    [[
        [2.0, 2.0],
        [2.0, 2.0],
    ]],
], $gradInput, 'conv2dBackward calcula gradInput corretamente');

assertTensorClose([
    [[
        [10.0],
    ]],
], $gradFilters, 'conv2dBackward calcula gradFilters corretamente');

assertTensorClose(
    [4.0],
    $gradBias,
    'conv2dBackward calcula gradBias corretamente'
);

/*
 * GradOutput não uniforme.
 *
 * input:
 * 1 2
 * 3 4
 *
 * filtro = 2
 *
 * gradOutput:
 * 1 2
 * 3 4
 *
 * gradInput = gradOutput * 2
 *
 * gradFilter:
 * 1*1 + 2*2 + 3*3 + 4*4 = 30
 *
 * gradBias:
 * 1 + 2 + 3 + 4 = 10
 */

$customGradOutput = ZTensor::arr([
    [[
        [1.0, 2.0],
        [3.0, 4.0],
    ]],
]);

[
    $customGradInput,
    $customGradFilters,
    $customGradBias,
] = $backwardInput->conv2dBackward(
    $customGradOutput,
    $backwardFilter
);

assertTensorClose([
    [[
        [2.0, 4.0],
        [6.0, 8.0],
    ]],
], $customGradInput, 'conv2dBackward respeita gradOutput');

assertTensorClose([
    [[
        [30.0],
    ]],
], $customGradFilters, 'conv2dBackward pondera gradFilters por gradOutput');

assertTensorClose(
    [10.0],
    $customGradBias,
    'conv2dBackward soma gradOutput no gradBias'
);

assertArrayClose(
    $backwardInputBefore,
    $backwardInput->toArray(),
    'conv2dBackward preserva o input'
);

assertArrayClose(
    $backwardFilterBefore,
    $backwardFilter->toArray(),
    'conv2dBackward preserva os filtros'
);

assertArrayClose(
    $gradOutputBefore,
    $gradOutput->toArray(),
    'conv2dBackward preserva gradOutput'
);

/*
|--------------------------------------------------------------------------
| 8. maxPool2d()
|--------------------------------------------------------------------------
*/

section('8. maxPool2d');

$poolInput = ZTensor::arr([
    [[
        [1.0,  2.0,  3.0,  4.0],
        [5.0,  6.0,  7.0,  8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]],
]);

$poolInputBefore = $poolInput->toArray();

$poolResult = $poolInput->maxPool2d(
    2,
    2,
    2,
    2,
    0,
    0
);

assertTrue(
    count($poolResult) === 2,
    'maxPool2d retorna output e indices'
);

assertTrue(
    $poolResult[0] instanceof ZTensor,
    'maxPool2d retorna output como ZTensor'
);

assertTrue(
    $poolResult[1] instanceof ZTensor,
    'maxPool2d retorna indices como ZTensor'
);

[$poolOutput, $poolIndices] = $poolResult;

assertShape(
    [1, 1, 2, 2],
    $poolOutput,
    'maxPool2d calcula o shape da saída'
);

assertShape(
    [1, 1, 2, 2],
    $poolIndices,
    'maxPool2d retorna um índice para cada saída'
);

assertTensorClose([
    [[
        [6.0, 8.0],
        [14.0, 16.0],
    ]],
], $poolOutput, 'maxPool2d retorna os máximos das janelas');

/*
 * A stub informa que os índices são float32 exatos.
 *
 * O contrato não informa se são índices globais, locais ou lineares.
 * Portanto, validamos que são:
 *
 * - finitos;
 * - não negativos;
 * - inteiros representados exatamente como float.
 */

$poolIndicesValues = flattenArray($poolIndices->toArray());

foreach ($poolIndicesValues as $index => $value) {
    assertTrue(
        is_finite($value),
        "maxPool2d índice {$index} é finito"
    );

    assertTrue(
        $value >= 0.0,
        "maxPool2d índice {$index} não é negativo"
    );

    assertTrue(
        floor($value) === $value,
        "maxPool2d índice {$index} representa um inteiro exato"
    );
}

/*
 * Pooling com janela sobreposta.
 */

$poolOverlapInput = ZTensor::arr([
    [[
        [1.0, 5.0, 2.0],
        [4.0, 9.0, 3.0],
        [6.0, 8.0, 7.0],
    ]],
]);

[$poolOverlapOutput, $poolOverlapIndices] =
    $poolOverlapInput->maxPool2d(
        2,
        2,
        1,
        1,
        0,
        0
    );

assertShape(
    [1, 1, 2, 2],
    $poolOverlapOutput,
    'maxPool2d suporta janelas sobrepostas'
);

assertTensorClose([
    [[
        [9.0, 9.0],
        [9.0, 9.0],
    ]],
], $poolOverlapOutput, 'maxPool2d encontra máximos em janelas sobrepostas');

assertArrayClose(
    $poolInputBefore,
    $poolInput->toArray(),
    'maxPool2d preserva o input'
);

/*
|--------------------------------------------------------------------------
| 9. maxPool2dBackward()
|--------------------------------------------------------------------------
*/

section('9. maxPool2dBackward');

$poolGradOutput = ZTensor::arr([
    [[
        [1.0, 2.0],
        [3.0, 4.0],
    ]],
]);

$poolIndicesBefore = $poolIndices->toArray();
$poolGradOutputBefore = $poolGradOutput->toArray();

$poolGradInput = $poolInput->maxPool2dBackward(
    $poolGradOutput,
    $poolIndices,
    [1, 1, 4, 4]
);

assertShape(
    [1, 1, 4, 4],
    $poolGradInput,
    'maxPool2dBackward recupera o shape do input'
);

assertTensorClose([
    [[
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 4.0],
    ]],
], $poolGradInput, 'maxPool2dBackward direciona gradientes aos máximos');

/*
 * O mesmo máximo aparece nas quatro janelas sobrepostas.
 *
 * Como colabora com quatro saídas, os gradientes devem ser acumulados
 * na posição central.
 */

$overlapGradOutput = ZTensor::full(
    [1, 1, 2, 2],
    1.0
);

$overlapGradInput = $poolOverlapInput->maxPool2dBackward(
    $overlapGradOutput,
    $poolOverlapIndices,
    [1, 1, 3, 3]
);

assertShape(
    [1, 1, 3, 3],
    $overlapGradInput,
    'maxPool2dBackward suporta overlap'
);

assertTensorClose([
    [[
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0],
    ]],
], $overlapGradInput, 'maxPool2dBackward acumula gradientes sobrepostos');

assertArrayClose(
    $poolGradOutputBefore,
    $poolGradOutput->toArray(),
    'maxPool2dBackward preserva gradOutput'
);

assertArrayClose(
    $poolIndicesBefore,
    $poolIndices->toArray(),
    'maxPool2dBackward preserva indices'
);

/*
|--------------------------------------------------------------------------
| 10. randomUniform()
|--------------------------------------------------------------------------
|
| Contrato:
|
| - shape solicitado;
| - valores em [minimum, maximum);
| - mesma seed produz os mesmos valores;
| - seeds diferentes devem produzir sequências diferentes;
| - valores determinísticos.
|
|--------------------------------------------------------------------------
*/

section('10. randomUniform');

$randomA = ZTensor::randomUniform(
    [2, 3, 4],
    -2.5,
    7.5,
    12345
);

$randomB = ZTensor::randomUniform(
    [2, 3, 4],
    -2.5,
    7.5,
    12345
);

$randomC = ZTensor::randomUniform(
    [2, 3, 4],
    -2.5,
    7.5,
    54321
);

assertShape(
    [2, 3, 4],
    $randomA,
    'randomUniform produz o shape solicitado'
);

$randomAValues = flattenArray($randomA->toArray());
$randomBValues = flattenArray($randomB->toArray());
$randomCValues = flattenArray($randomC->toArray());

assertTrue(
    count($randomAValues) === 24,
    'randomUniform produz a quantidade correta de valores'
);

foreach ($randomAValues as $index => $value) {
    assertTrue(
        is_finite($value),
        "randomUniform valor {$index} é finito"
    );

    assertTrue(
        $value >= -2.5,
        "randomUniform valor {$index} respeita o mínimo inclusivo"
    );

    assertTrue(
        $value < 7.5,
        "randomUniform valor {$index} respeita o máximo exclusivo"
    );
}

/*
 * Determinismo exato para a mesma seed.
 */

assertSameArray(
    $randomA->toArray(),
    $randomB->toArray(),
    'randomUniform é determinístico com a mesma seed'
);

/*
 * Seeds diferentes devem alterar a sequência.
 */

assertTrue(
    $randomA->toArray() !== $randomC->toArray(),
    'randomUniform produz sequência diferente com outra seed'
);

/*
 * Intervalo positivo pequeno.
 */

$randomSmallRange = ZTensor::randomUniform(
    [100],
    0.25,
    0.50,
    999
);

assertShape(
    [100],
    $randomSmallRange,
    'randomUniform suporta tensor unidimensional'
);

foreach (
    flattenArray($randomSmallRange->toArray())
    as $index => $value
) {
    assertTrue(
        $value >= 0.25 && $value < 0.50,
        "randomUniform valor {$index} pertence a [0.25, 0.50)"
    );
}

/*
 * Confirma que a sequência não contém somente um valor repetido.
 */

$uniqueRandomValues = array_unique(
    array_map(
        static fn(float $value): string => sprintf('%.9f', $value),
        $randomAValues
    )
);

assertTrue(
    count($uniqueRandomValues) > 1,
    'randomUniform produz uma sequência não constante'
);

/*
|--------------------------------------------------------------------------
| 11. Validações de argumentos
|--------------------------------------------------------------------------
|
| Estes testes verificam erros que são necessários para manter os contratos.
|
|--------------------------------------------------------------------------
*/

section('11. validações de argumentos');

assertThrows(
    static fn() => $input3d->permute([0, 0, 1]),
    'permute rejeita eixos duplicados'
);

assertThrows(
    static fn() => $input3d->permute([0, 1]),
    'permute rejeita quantidade incorreta de eixos'
);

assertThrows(
    static fn() => $input3d->permute([0, 1, 3]),
    'permute rejeita eixo fora do intervalo'
);

assertThrows(
    static fn() => $input3d->flatten(2, 1),
    'flatten rejeita intervalo invertido'
);

assertThrows(
    static fn() => $input3d->flatten(0, 3),
    'flatten rejeita eixo fora do intervalo'
);

assertThrows(
    static fn() => $broadcastInput->broadcastTo([2, 2]),
    'broadcastTo rejeita shape incompatível'
);

assertThrows(
    static fn() => ZTensor::zeros([1, 3, 3])->im2col(
        2,
        2
    ),
    'im2col rejeita tensor que não seja NCHW'
);

assertThrows(
    static fn() => $image3x3->im2col(
        0,
        2
    ),
    'im2col rejeita kernel de altura zero'
);

assertThrows(
    static fn() => $image3x3->im2col(
        2,
        2,
        0,
        1
    ),
    'im2col rejeita stride zero'
);

assertThrows(
    static function () use ($image3x3): void {
        $invalidFilters = ZTensor::ones([1, 2, 1, 1]);

        $image3x3->conv2d($invalidFilters);
    },
    'conv2d rejeita canais incompatíveis'
);

assertThrows(
    static function () use ($image3x3): void {
        $filters = ZTensor::ones([2, 1, 1, 1]);
        $invalidBias = ZTensor::zeros([3]);

        $image3x3->conv2d(
            $filters,
            $invalidBias
        );
    },
    'conv2d rejeita bias incompatível'
);

assertThrows(
    static fn() => ZTensor::zeros([1, 3, 3])->maxPool2d(
        2,
        2,
        1,
        1
    ),
    'maxPool2d rejeita tensor que não seja NCHW'
);

assertThrows(
    static fn() => $poolInput->maxPool2d(
        0,
        2,
        1,
        1
    ),
    'maxPool2d rejeita kernel zero'
);

assertThrows(
    static fn() => $poolInput->maxPool2d(
        2,
        2,
        0,
        1
    ),
    'maxPool2d rejeita stride zero'
);

assertThrows(
    static fn() => ZTensor::randomUniform(
        [2, 2],
        10.0,
        5.0,
        123
    ),
    'randomUniform rejeita minimum maior que maximum'
);

/*
|--------------------------------------------------------------------------
| Resultado final
|--------------------------------------------------------------------------
*/

echo "\n";
echo "==================================================\n";
echo "🚀 TODOS OS NOVOS MÉTODOS FORAM VALIDADOS\n";
echo "==================================================\n";

function assertClose(array $expected, array $actual, string $label, float $eps = 1e-3): void {
    $fe = []; array_walk_recursive($expected, function($v) use (&$fe){ $fe[]=$v; });
    $fa = []; array_walk_recursive($actual, function($v) use (&$fa){ $fa[]=$v; });
    if (count($fe) !== count($fa)) { echo "❌ $label: tamanhos diferentes\n"; return; }
    foreach ($fe as $i => $v) {
        if (abs($v - $fa[$i]) > $eps) { echo "❌ $label: idx $i esperado $v obtido {$fa[$i]}\n"; return; }
    }
    echo "✅ $label\n";
}

echo "--- globalAveragePool2d: caso manual ---\n";
$x = ZTensor::arr([[
    [[1.0, 2.0], [3.0, 4.0]],       // canal 0: média = 2.5
    [[10.0, 20.0], [30.0, 40.0]],   // canal 1: média = 25.0
]]); // (1,2,2,2)
$pooled = $x->globalAveragePool2d();
assertClose([1,2], $pooled->shape(), "globalAveragePool2d shape");
assertClose([[2.5, 25.0]], $pooled->toArray(), "globalAveragePool2d valores");

echo "\n--- globalAveragePool2d: batch > 1 ---\n";
$batch = ZTensor::arr([
    [[[1.0,1.0],[1.0,1.0]]],
    [[[2.0,2.0],[2.0,2.0]]],
]); // (2,1,2,2)
assertClose([[1.0],[2.0]], $batch->globalAveragePool2d()->toArray(), "globalAveragePool2d batch");

echo "\n--- globalAveragePool2dBackward: distribuição uniforme ---\n";
$gradOut = ZTensor::arr([[4.0, 8.0]]); // (1,2)
$gradIn = ZTensor::globalAveragePool2dBackward($gradOut, [1,2,2,2]);
assertClose([[
    [[1.0,1.0],[1.0,1.0]], // 4.0/4
    [[2.0,2.0],[2.0,2.0]], // 8.0/4
]], $gradIn->toArray(), "globalAveragePool2dBackward valores");

echo "\n--- Gradiente numérico (diferenças finitas) ---\n";
function golLoss(ZTensor $x): float {
    return $x->globalAveragePool2d()->sumtotal(); // dLoss/dPooled = 1 em todo lugar
}
$xArr = $x->toArray();
$eps = 1e-3;
$numGrad = $xArr;
for ($c=0;$c<2;$c++) for ($i=0;$i<2;$i++) for ($j=0;$j<2;$j++) {
    $plus = $xArr; $plus[0][$c][$i][$j] += $eps;
    $minus = $xArr; $minus[0][$c][$i][$j] -= $eps;
    $lp = golLoss(ZTensor::arr($plus));
    $lm = golLoss(ZTensor::arr($minus));
    $numGrad[0][$c][$i][$j] = ($lp - $lm) / (2*$eps);
}
$gradOutOnes = ZTensor::full([1,2], 1.0);
$analyticalGrad = ZTensor::globalAveragePool2dBackward($gradOutOnes, [1,2,2,2]);
assertClose($numGrad, $analyticalGrad->toArray(), "gradiente numérico globalAveragePool2d", 5e-2);

echo "\n--- Preservação da entrada ---\n";
$before = $x->toArray();
$x->globalAveragePool2d();
assertClose($before, $x->toArray(), "globalAveragePool2d preserva input");

echo "\n--- Encadeamento com conv2d ---\n";
$img = ZTensor::arr([[[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]]]); // (1,1,3,3)
$kernel = ZTensor::arr([[[[1.0,0.0],[0.0,1.0]],[[0.0,1.0],[1.0,0.0]]]]); // (2,1,2,2)
$features = $img->conv2d($kernel); // (1,2,2,2)
$gap = $features->globalAveragePool2d(); // (1,2)
assertClose([1,2], $gap->shape(), "conv2d -> globalAveragePool2d encadeado, shape final");

echo "\n--- Validações/exceções ---\n";
try {
    ZTensor::zeros([2,2])->globalAveragePool2d();
    echo "❌ deveria lançar (rank != 4)\n";
} catch (Throwable $e) { echo "✅ globalAveragePool2d rank inválido: {$e->getMessage()}\n"; }

try {
    ZTensor::globalAveragePool2dBackward(ZTensor::arr([[1.0,2.0,3.0]]), [1,2,2,2]);
    echo "❌ deveria lançar (gradOutput incompatível)\n";
} catch (Throwable $e) { echo "✅ globalAveragePool2dBackward shape incompatível: {$e->getMessage()}\n"; }

try {
    $rg = $x->copy()->requiresGrad(true);
    $rg->globalAveragePool2d();
    echo "❌ deveria lançar (requires_grad)\n";
} catch (Throwable $e) { echo "✅ globalAveragePool2d + requiresGrad: {$e->getMessage()}\n"; }

echo "\n🚀 TESTES DE globalAveragePool2d CONCLUÍDOS!\n";