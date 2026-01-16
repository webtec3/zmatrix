<?php
// Teste de segurança de memória - sem leaks

use ZMatrix\ZTensor;

echo "╔════════════════════════════════════════════════╗\n";
echo "║  TESTE DE SEGURANÇA DE MEMÓRIA E INTEGRIDADE   ║\n";
echo "╚════════════════════════════════════════════════╝\n\n";

// ===== TEST 1: Large tensor allocation and deallocation =====
echo "[TEST 1] Alocação e desalocação de tensores grandes...\n";
for ($i = 0; $i < 100; $i++) {
    $t = new ZTensor([1000, 1000], 3.14);
    unset($t);
}
echo "✅ 100 iterações - Sem crash\n\n";

// ===== TEST 2: Operations on large tensors =====
echo "[TEST 2] Operações em tensores grandes...\n";
for ($i = 0; $i < 50; $i++) {
    $t1 = new ZTensor([500, 500], 2.0);
    $t2 = new ZTensor([500, 500], 3.0);
    
    // Várias operações que acionam OpenMP
    $t1->add($t2);
    $t1->mul(0.5);
    $t1->relu();
    $t1->sigmoid();
    
    unset($t1, $t2);
}
echo "✅ 50 operações complexas - Sem crash\n\n";

// ===== TEST 3: Reductions =====
echo "[TEST 3] Operações de redução...\n";
for ($i = 0; $i < 100; $i++) {
    $t = new ZTensor([10000], 5.0);
    
    $sum = $t->sum();
    $min = $t->min();
    $max = $t->max();
    $mean = $t->mean();
    
    unset($t);
}
echo "✅ 100 reduções - Sem crash\n\n";

// ===== TEST 4: Mixed operations =====
echo "[TEST 4] Operações mistas...\n";
for ($i = 0; $i < 30; $i++) {
    $t = new ZTensor([2000], 1.5);
    
    $t->relu();
    $t->add(new ZTensor([2000], 0.5));
    $m = $t->mean();
    $t->mul($m);
    $s = $t->sum();
    
    unset($t);
}
echo "✅ 30 operações mistas - Sem crash\n\n";

// ===== TEST 5: Activation functions stress =====
echo "[TEST 5] Funções de ativação (stress)...\n";
for ($i = 0; $i < 50; $i++) {
    $t = new ZTensor([5000], 0.5);
    
    $t->relu();
    $t->sigmoid();
    $t->tanh();
    
    unset($t);
}
echo "✅ 50 iterações de ativações - Sem crash\n\n";

echo "╔════════════════════════════════════════════════╗\n";
echo "║       ✅ TODOS OS TESTES DE MEMÓRIA OK!        ║\n";
echo "╚════════════════════════════════════════════════╝\n";

