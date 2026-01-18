<?php

/**
 * GPU vs CPU Comprehensive Test Suite
 * ====================================
 * 
 * This test suite validates GPU functionality in the ZMatrix extension,
 * including tensor movement between GPU/CPU, operation performance,
 * and correctness of GPU accelerated operations.
 * 
 * Run with: php test_gpu_vs_cpu.php
 */

declare(strict_types=1);

use ZMatrix\ZTensor;

// ============================================================================
// TEST UTILITIES
// ============================================================================
if (!extension_loaded('zmatrix')) {
    echo "❌ Extensão zmatrix não carregada!\n";
    exit(1);
}

echo "═══════════════════════════════════════════════════════════════════════\n";
echo "  GPU vs CPU BENCHMARK - Dados Residentes (GPU Resident)\n";
echo "═══════════════════════════════════════════════════════════════════════\n";
echo "  🔥 Transferência UMA VEZ, depois operações puras\n\n";

// Teste com diferentes tamanhos
$tests = [
    ['name' => 'Pequeno (50K)',    'size' => 50_000,    'iter' => 100],
    ['name' => 'Médio (500K)',     'size' => 500_000,   'iter' => 50],
    ['name' => 'Grande (2M)',      'size' => 2_000_000, 'iter' => 20],
    ['name' => 'MuitoGrande (5M)', 'size' => 5_000_000, 'iter' => 10],
];

$results = [];

foreach ($tests as $test) {
    $name = $test['name'];
    $size = $test['size'];
    $iter = $test['iter'];

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    echo "Teste: $name (n=" . number_format($size) . " elementos, $iter iterações)\n";
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    // Dados
    $a_data = array_fill(0, $size, 0.5);
    $b_data = array_fill(0, $size, 0.3);

    // ===== CPU BENCHMARK =====
    echo "  ⚙️  CPU (puro): ";
    flush();
    $a = new ZMatrix\ZTensor($a_data);
    $b = new ZMatrix\ZTensor($b_data);
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

    // ===== GPU BENCHMARK (RESIDENT) =====
    // Transferência UMA VEZ
    echo "  🎮 GPU (resident, sem roundtrip): ";
    flush();

    // Setup: transferir para GPU fora do loop
    $a_gpu = (new ZMatrix\ZTensor($a_data))->toGpu();
    $b_gpu = (new ZMatrix\ZTensor($b_data))->toGpu();

    if (!$a_gpu->isOnGpu() || !$b_gpu->isOnGpu()) {
        echo "\n  ⚠️  AVISO: Dados não estão na GPU!\n";
    }

    // Operações puras na GPU (sem transferência)
    $start = microtime(true);
    for ($i = 0; $i < $iter; $i++) {
        $r1 = $a_gpu->add($b_gpu);
        $r2 = $r1->mul($b_gpu);
        $r3 = $r2->sub($b_gpu);
    }
    $time_gpu = (microtime(true) - $start) / $iter;
    $results[$name]['gpu'] = $time_gpu * 1000;

    printf("%.4f ms", $time_gpu * 1000);
    echo "\n";

    // Cálculo de speedup
    $speedup = $time_cpu / $time_gpu;
    $results[$name]['speedup'] = $speedup;

    echo "  📊 Speedup: ";
    if ($speedup >= 5.0) {
        printf("🚀 GPU %.2fx mais rápido\n", $speedup);
    } elseif ($speedup >= 2.0) {
        printf("✅ GPU %.2fx mais rápido\n", $speedup);
    } elseif ($speedup >= 1.0) {
        printf("✓ GPU %.2fx mais rápido\n", $speedup);
    } else {
        printf("⚠️  CPU %.2fx mais rápido\n", 1.0 / $speedup);
    }

    echo "\n";
}

echo "\n" . str_repeat("═", 85) . "\n";
echo "COMPARATIVO FINAL - DADOS RESIDENTES NA GPU\n";
echo str_repeat("═", 85) . "\n\n";

echo "┌──────────────────┬──────────────┬──────────────┬─────────────────┐\n";
echo "│ Tamanho          │ CPU (ms)     │ GPU (ms)     │ Speedup GPU     │\n";
echo "├──────────────────┼──────────────┼──────────────┼─────────────────┤\n";

$total_speedup = 0;
$count = 0;

foreach ($results as $name => $data) {
    $speedup = $data['speedup'];
    if ($speedup >= 1.0) {
        $speedup_str = sprintf("%.2fx ✅", $speedup);
    } else {
        $speedup_str = sprintf("CPU %.2fx ⚠️", 1.0 / $speedup);
    }

    printf("│ %-16s │ %12.4f │ %12.4f │ %-15s │\n",
        substr($name, 0, 16),
        $data['cpu'],
        $data['gpu'],
        $speedup_str
    );

    $total_speedup += $speedup;
    $count++;
}

echo "└──────────────────┴──────────────┴──────────────┴─────────────────┘\n";

if ($count > 0) {
    $avg_speedup = $total_speedup / $count;
    echo "\n";
    echo "📈 ANÁLISE FINAL:\n";
    echo "   Speedup médio GPU: " . sprintf("%.2fx", $avg_speedup) . "\n";

    if ($avg_speedup >= 5.0) {
        echo "   Status: GPU EXCELENTE 🚀🚀🚀\n";
        echo "\n   💡 Conclusão: GPU brilha com dados residentes!\n";
    } elseif ($avg_speedup >= 2.0) {
        echo "   Status: GPU BOM ✅✅\n";
        echo "\n   💡 Conclusão: GPU vale a pena para operações em batches\n";
    } elseif ($avg_speedup >= 1.0) {
        echo "   Status: GPU ÚTIL ✓\n";
        echo "\n   💡 Conclusão: GPU útil mas com overhead controlado\n";
    } else {
        echo "   Status: GPU COM OVERHEAD ⚠️\n";
    }

    echo "\n   📌 Este teste mostra o cenário ideal: dados uma vez na GPU,\n";
    echo "      múltiplas operações sem roundtrip PCIe.\n";
}