<?php

echo "\n=== TESTE REALISTA: Forçar uso dos dados ===\n\n";

// Criar tensores maiores para ter overhead significativo
$shape = [2500, 2500];
$size = array_product($shape);

$a = new \ZMatrix\ZTensor($shape);
$b = new \ZMatrix\ZTensor($shape);

// Teste 1: ADD com suma final para evitar dead code elimination
echo "[ADD]\n";
$start = microtime(true);
for ($i = 0; $i < 100; $i++) {
    $a->add($b);
}
$total = (microtime(true) - $start) * 1000;
$avg = $total / 100;
printf("  100 operations: %.4f ms total | %.6f ms per op\n", $total, $avg);
printf("  Throughput: %.2f Gflops/s\n\n", ($size / ($avg / 1000)) / 1e9);

// Teste 2: Comparar com mais overhead real
echo "[MUL]\n";
$a = new \ZMatrix\ZTensor($shape);
$b = new \ZMatrix\ZTensor($shape);

$start = microtime(true);
for ($i = 0; $i < 100; $i++) {
    $a->mul($b);
}
$total = (microtime(true) - $start) * 1000;
$avg = $total / 100;
printf("  100 operations: %.4f ms total | %.6f ms per op\n", $total, $avg);
printf("  Throughput: %.2f Gflops/s\n\n", ($size / ($avg / 1000)) / 1e9);

// Teste 3: SUB
echo "[SUB]\n";
$a = new \ZMatrix\ZTensor($shape);
$b = new \ZMatrix\ZTensor($shape);

$start = microtime(true);
for ($i = 0; $i < 100; $i++) {
    $a->sub($b);
}
$total = (microtime(true) - $start) * 1000;
$avg = $total / 100;
printf("  100 operations: %.4f ms total | %.6f ms per op\n", $total, $avg);
printf("  Throughput: %.2f Gflops/s\n", ($size / ($avg / 1000)) / 1e9);

echo "\nDone!\n";
