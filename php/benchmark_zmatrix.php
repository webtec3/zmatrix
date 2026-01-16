#!/usr/bin/env php
<?php
/**
 * ZMatrix Benchmark Suite
 * Compara performance de CPU vs GPU com operações equivalentes ao benchmark Python
 */

use ZMatrix\ZTensor;

error_reporting(E_ALL);
ini_set('display_errors', 1);

// Setup GPU environment
putenv('ZMATRIX_GPU_DEBUG=0');
putenv('LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:' . getenv('LD_LIBRARY_PATH'));

$RESULTS = [];

// Utility functions
function benchmark_php(string $name, callable $func, int $iterations = 5) {
    $times = [];
    
    for ($i = 0; $i < $iterations; $i++) {
        $start = microtime(true);
        $result = $func();
        $end = microtime(true);
        $times[] = ($end - $start) * 1000;  // ms
    }
    
    $avg = array_sum($times) / count($times);
    $std = sqrt(array_reduce($times, fn($carry, $x) => $carry + pow($x - $avg, 2), 0) / count($times));
    
    return [
        'name' => $name,
        'avg_ms' => round($avg, 3),
        'std_ms' => round($std, 3),
        'min_ms' => round(min($times), 3),
        'max_ms' => round(max($times), 3)
    ];
}

echo "╔════════════════════════════════════════════════════════════════╗\n";
echo "║   ZMatrix Benchmark Suite (PHP)                               ║\n";
echo "╚════════════════════════════════════════════════════════════════╝\n\n";

// Check GPU availability
echo "Checking GPU availability...\n";
$test_tensor = ZTensor::arr([1, 2, 3]);
try {
    $test_tensor->toGpu();
    $gpu_available = $test_tensor->isOnGpu();
    $test_tensor->toCpu();
    if ($gpu_available) {
        echo "✅ GPU available - Full benchmarks will run\n\n";
    }
} catch (Exception $e) {
    echo "⚠️  GPU not available - CPU benchmarks only\n";
    echo "   Error: " . $e->getMessage() . "\n\n";
    $gpu_available = false;
}

// ─── TESTE 1: CRIAÇÃO ───
echo "═══════════════════════════════════════════════════════════════\n";
echo "TEST 1: Creation and Initialization\n";
echo "═══════════════════════════════════════════════════════════════\n\n";

// ZMatrix Random
$result = benchmark_php("ZMatrix Random [1M]", function() {
    return ZTensor::random([1_000_000], -1.0, 1.0);
});
printf("ZMatrix random: %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['creation_zmatrix_random'] = $result;

// ZMatrix Zeros
$result = benchmark_php("ZMatrix Zeros [1M]", function() {
    return ZTensor::zeros([1_000_000]);
});
printf("ZMatrix zeros:  %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['creation_zmatrix_zeros'] = $result;

// ZMatrix Ones
$result = benchmark_php("ZMatrix Ones [1M]", function() {
    return ZTensor::ones([1_000_000]);
});
printf("ZMatrix ones:   %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['creation_zmatrix_ones'] = $result;

// ─── TESTE 2: OPERAÇÕES ARITMÉTICAS ───
echo "\n═══════════════════════════════════════════════════════════════\n";
echo "TEST 2: Arithmetic Operations [5M elements]\n";
echo "═══════════════════════════════════════════════════════════════\n\n";

$size = 5_000_000;
$a = ZTensor::random([$size], -1.0, 1.0);
$b = ZTensor::random([$size], -1.0, 1.0);

// ZMatrix Add (CPU)
$result = benchmark_php("ZMatrix Add (CPU)", function() use ($a, $b) {
    return $a->copy()->add($b);
}, 10);
printf("ZMatrix add:    %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['arithmetic_zmatrix_add_cpu'] = $result;

// ZMatrix Sub (CPU)
$result = benchmark_php("ZMatrix Sub (CPU)", function() use ($a, $b) {
    return $a->copy()->sub($b);
}, 10);
printf("ZMatrix sub:    %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['arithmetic_zmatrix_sub_cpu'] = $result;

// ZMatrix Mul (CPU)
$result = benchmark_php("ZMatrix Mul (CPU)", function() use ($a, $b) {
    return $a->copy()->mul($b);
}, 10);
printf("ZMatrix mul:    %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['arithmetic_zmatrix_mul_cpu'] = $result;

// ZMatrix Div (CPU)
$result = benchmark_php("ZMatrix Div (CPU)", function() use ($a, $b) {
    return $a->copy()->divide($b);
}, 10);
printf("ZMatrix div:    %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['arithmetic_zmatrix_div_cpu'] = $result;

if ($gpu_available) {
    echo "\n";
    
    // ZMatrix Add (GPU)
    $a_gpu = $a->copy();
    $b_gpu = $b->copy();
    $a_gpu->toGpu();
    $b_gpu->toGpu();
    
    $result = benchmark_php("ZMatrix Add (GPU)", function() use ($a_gpu, $b_gpu) {
        $temp = $a_gpu->copy();
        $temp->add($b_gpu);
        return $temp;
    }, 10);
    printf("ZMatrix add (GPU): %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
    $RESULTS['arithmetic_zmatrix_add_gpu'] = $result;
    
    // ZMatrix Sub (GPU)
    $result = benchmark_php("ZMatrix Sub (GPU)", function() use ($a_gpu, $b_gpu) {
        $temp = $a_gpu->copy();
        $temp->sub($b_gpu);
        return $temp;
    }, 10);
    printf("ZMatrix sub (GPU): %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
    $RESULTS['arithmetic_zmatrix_sub_gpu'] = $result;
    
    // ZMatrix Mul (GPU)
    $result = benchmark_php("ZMatrix Mul (GPU)", function() use ($a_gpu, $b_gpu) {
        $temp = $a_gpu->copy();
        $temp->mul($b_gpu);
        return $temp;
    }, 10);
    printf("ZMatrix mul (GPU): %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
    $RESULTS['arithmetic_zmatrix_mul_gpu'] = $result;
}

// ─── TESTE 3: ATIVAÇÕES ───
echo "\n═══════════════════════════════════════════════════════════════\n";
echo "TEST 3: Activation Functions [5M elements]\n";
echo "═══════════════════════════════════════════════════════════════\n\n";

// ZMatrix ReLU (CPU)
$result = benchmark_php("ZMatrix ReLU (CPU)", function() use ($a) {
    return $a->copy()->relu();
}, 10);
printf("ZMatrix ReLU:   %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['activation_zmatrix_relu_cpu'] = $result;

// ZMatrix Sigmoid (CPU)
$result = benchmark_php("ZMatrix Sigmoid (CPU)", function() use ($a) {
    return $a->copy()->sigmoid();
}, 5);
printf("ZMatrix Sigmoid: %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['activation_zmatrix_sigmoid_cpu'] = $result;

// ZMatrix Tanh (CPU)
$result = benchmark_php("ZMatrix Tanh (CPU)", function() use ($a) {
    return $a->copy()->tanh();
}, 10);
printf("ZMatrix Tanh:   %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['activation_zmatrix_tanh_cpu'] = $result;

// ZMatrix Softmax (small, CPU)
$a_small = ZTensor::random([10_000], -1.0, 1.0);
$result = benchmark_php("ZMatrix Softmax (CPU)", function() use ($a_small) {
    return $a_small->copy()->softmax();
}, 10);
printf("ZMatrix Softmax: %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['activation_zmatrix_softmax_cpu'] = $result;

if ($gpu_available) {
    echo "\n";
    
    // ZMatrix ReLU (GPU)
    $result = benchmark_php("ZMatrix ReLU (GPU)", function() use ($a_gpu) {
        $temp = $a_gpu->copy();
        $temp->relu();
        return $temp;
    }, 10);
    printf("ZMatrix ReLU (GPU):   %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
    $RESULTS['activation_zmatrix_relu_gpu'] = $result;
    
    // ZMatrix Sigmoid (GPU)
    $result = benchmark_php("ZMatrix Sigmoid (GPU)", function() use ($a_gpu) {
        $temp = $a_gpu->copy();
        $temp->sigmoid();
        return $temp;
    }, 5);
    printf("ZMatrix Sigmoid (GPU): %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
    $RESULTS['activation_zmatrix_sigmoid_gpu'] = $result;
    
    // ZMatrix Tanh (GPU)
    $result = benchmark_php("ZMatrix Tanh (GPU)", function() use ($a_gpu) {
        $temp = $a_gpu->copy();
        $temp->tanh();
        return $temp;
    }, 10);
    printf("ZMatrix Tanh (GPU):   %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
    $RESULTS['activation_zmatrix_tanh_gpu'] = $result;
}

// ─── TESTE 4: ÁLGEBRA LINEAR ───
echo "\n═══════════════════════════════════════════════════════════════\n";
echo "TEST 4: Linear Algebra\n";
echo "═══════════════════════════════════════════════════════════════\n\n";

// MatMul [1000x1000]
$size = 1000;
$A = ZTensor::random([$size, $size], -1.0, 1.0);
$B = ZTensor::random([$size, $size], -1.0, 1.0);

$result = benchmark_php("ZMatrix MatMul [1Kx1K]", function() use ($A, $B) {
    return $A->matmul($B);
}, 3);
printf("ZMatrix MatMul [1Kx1K]: %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['linalg_zmatrix_matmul_1k_cpu'] = $result;

// Dot Product [1M]
$vec1 = ZTensor::random([1_000_000], -1.0, 1.0);
$vec2 = ZTensor::random([1_000_000], -1.0, 1.0);

$result = benchmark_php("ZMatrix Dot [1M]", function() use ($vec1, $vec2) {
    return $vec1->dot($vec2);
}, 10);
printf("ZMatrix Dot [1M]:       %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['linalg_zmatrix_dot_1m_cpu'] = $result;

// ─── TESTE 5: ESTATÍSTICAS ───
echo "\n═══════════════════════════════════════════════════════════════\n";
echo "TEST 5: Statistics [5M elements]\n";
echo "═══════════════════════════════════════════════════════════════\n\n";

// ZMatrix Sum
$result = benchmark_php("ZMatrix Sum", function() use ($a) {
    return $a->sumtotal();
}, 10);
printf("ZMatrix Sum:   %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['stats_zmatrix_sum'] = $result;

// ZMatrix Mean
$result = benchmark_php("ZMatrix Mean", function() use ($a) {
    return $a->mean();
}, 10);
printf("ZMatrix Mean:  %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['stats_zmatrix_mean'] = $result;

// ZMatrix Std
$result = benchmark_php("ZMatrix Std", function() use ($a) {
    return $a->std();
}, 10);
printf("ZMatrix Std:   %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['stats_zmatrix_std'] = $result;

// ZMatrix Min/Max
$result = benchmark_php("ZMatrix Min/Max", function() use ($a) {
    return [$a->min(), $a->max()];
}, 10);
printf("ZMatrix Min/Max: %.3f ms ± %.3f ms\n", $result['avg_ms'], $result['std_ms']);
$RESULTS['stats_zmatrix_minmax'] = $result;

// ─── SALVAR RESULTADOS ───
echo "\n═══════════════════════════════════════════════════════════════\n";
echo "Saving results...\n";
echo "═══════════════════════════════════════════════════════════════\n\n";

$output_file = dirname(__FILE__) . '/benchmark_zmatrix_results.json';
file_put_contents($output_file, json_encode($RESULTS, JSON_PRETTY_PRINT));

echo "✅ Results saved to: $output_file\n";
echo "   Total benchmarks: " . count($RESULTS) . "\n";

echo "\n" . str_repeat("═", 65) . "\n";
echo "ZMatrix benchmarks completed!\n";
echo str_repeat("═", 65) . "\n\n";
