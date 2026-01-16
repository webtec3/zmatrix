<?php

echo "\n=== ANÁLISE: POR QUE ESTÁ TÃO RÁPIDO? ===\n\n";

// 1. Teste a velocidade básica de allocação + loop vazio
echo "[1] Tempo de alocação + loop (sem operações):\n";
$shape = [2500, 2500];
$size = array_product($shape);

$start = microtime(true);
for ($i = 0; $i < 50; $i++) {
    $a = new \ZMatrix\ZTensor($shape);
}
$alloc_time = (microtime(true) - $start) * 1000;
printf("  50 alocações: %.4f ms | por alocação: %.6f ms\n", $alloc_time, $alloc_time / 50);

// 2. Teste com valores reais
echo "\n[2] Performance com valores inicializados:\n";
$a = new \ZMatrix\ZTensor($shape);
$b = new \ZMatrix\ZTensor($shape);

$start = microtime(true);
for ($i = 0; $i < 50; $i++) {
    $a->add($b);
}
$total = (microtime(true) - $start) * 1000;
$avg = $total / 50;
printf("  50 adds: %.4f ms | por operação: %.6f ms\n", $total, $avg);
printf("  Elementos: %d\n", $size);
printf("  Throughput: %.2f Gflops/s\n", ($size / ($avg / 1000)) / 1e9);

// 3. Teste com números reais visíveis
echo "\n[3] Verificar se operação está sendo realizada:\n";
$a = new \ZMatrix\ZTensor([2, 2]);
$b = new \ZMatrix\ZTensor([2, 2]);
echo "  Antes: a=" . $a->toArray()[0][0] . ", b=" . $b->toArray()[0][0] . "\n";
$a->add($b);
echo "  Depois add: a=" . $a->toArray()[0][0] . "\n";

echo "\nAnálise concluída!\n";
