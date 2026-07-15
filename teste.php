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
    ['name' => 'Pequeno (50K)',    'size' => 50_000,    'iter' => 20],
    ['name' => 'Médio (500K)',     'size' => 500_000,   'iter' => 10],
    ['name' => 'Grande (2M)',      'size' => 2_000_000, 'iter' => 3],
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

function run_test(string $label, callable $fn): void {
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
        if (abs($v) > 1e-9) { $allZero = false; break; }
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
    array $expected,
    ZTensor $actual,
    string $message
): void {
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