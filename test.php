<?php
/**
 * Performance Test: GPU Matrix Multiplication (cuBLAS) vs CPU BLAS
 *
 * Testes de desempenho comparando:
 * - Pequenas matrizes: CPU puro (abaixo de 200K elementos)
 * - Grandes matrizes: GPU cuBLAS (acima de 200K elementos)
 *
 * Métricas:
 * - Tempo de execução (ms)
 * - Consumo de memória (MB)
 * - Speedup GPU vs CPU
 */

if (!extension_loaded('zmatrix')) {
    die("ERROR: ZMatrix extension not loaded\n");
}
use ZMatrix\ZTensor;
echo "╔════════════════════════════════════════════════════════════════╗\n";
echo "║  GPU Matrix Multiplication (cuBLAS) Performance Test           ║\n";
echo "║  Date: " . date('Y-m-d H:i:s') . "                               ║\n";
echo "╚════════════════════════════════════════════════════════════════╝\n\n";

// Helper function to format bytes
function format_bytes($bytes) {
    $units = ['B', 'KB', 'MB', 'GB'];
    $bytes = max($bytes, 0);
    $pow = floor(($bytes ? log($bytes) : 0) / log(1024));
    $pow = min($pow, count($units) - 1);
    $bytes /= (1 << (10 * $pow));
    return round($bytes, 2) . ' ' . $units[$pow];
}

// Helper function to format time
function format_time($ms) {
    if ($ms < 1) {
        return round($ms * 1000, 2) . ' μs';
    } elseif ($ms < 1000) {
        return round($ms, 2) . ' ms';
    } else {
        return round($ms / 1000, 2) . ' s';
    }
}

// Test configuration
$tests = [
    ['name' => 'Tiny (10×10)', 'size' => 10, 'elements' => 100, 'expect_gpu' => false],
    ['name' => 'Small (50×50)', 'size' => 50, 'elements' => 2500, 'expect_gpu' => false],
    ['name' => 'Medium (200×200)', 'size' => 200, 'elements' => 40000, 'expect_gpu' => false],
    ['name' => 'Large (400×400)', 'size' => 400, 'elements' => 160000, 'expect_gpu' => false],
    ['name' => 'XLarge (500×500)', 'size' => 500, 'elements' => 250000, 'expect_gpu' => true],
    ['name' => 'XXLarge (700×700)', 'size' => 700, 'elements' => 490000, 'expect_gpu' => true],
    ['name' => 'Huge (1000×1000)', 'size' => 1000, 'elements' => 1000000, 'expect_gpu' => true],
];

// Force CPU mode if requested
$force_cpu = getenv('FORCE_CPU');
if ($force_cpu) {
    echo "[INFO] FORCE_CPU mode enabled - all tests will use CPU\n\n";
}

$results = [];

foreach ($tests as $test) {
    $size = $test['size'];
    $name = $test['name'];
    $elements = $test['elements'];
    $expect_gpu = $test['expect_gpu'] && !$force_cpu;

    echo "─────────────────────────────────────────────────────────────\n";
    echo "Test: $name ($size × $size × $size × $size = $elements result elements)\n";
    echo "─────────────────────────────────────────────────────────────\n";

    // Create matrices with random values
    $mem_before = memory_get_usage(true);

    $a = ZTensor::random([$size, $size], 0.0, 1.0);
    $b = ZTensor::random([$size, $size], 0.0, 1.0);
    
    $mem_after_create = memory_get_usage(true);
    $mem_input = $mem_after_create - $mem_before;
    
    echo sprintf("Memory allocated for inputs: %s\n", format_bytes($mem_input));
    
    // Warm up (optional, helps with JIT)
    if ($size <= 200) {
        $_ = $a->matmul($b);
        unset($_);
    }
    
    // Perform matmul
    gc_collect_cycles();  // Clean up first
    $mem_before_compute = memory_get_usage(true);
    
    $time_start = microtime(true);
    $c = $a->matmul($b);
    $time_end = microtime(true);
    
    $time_ms = ($time_end - $time_start) * 1000;
    $mem_after_compute = memory_get_usage(true);
    $mem_compute = $mem_after_compute - $mem_before_compute;
    
    // Check if GPU was used
    $gpu_used = method_exists($c, 'is_on_gpu') && $c->is_on_gpu();
    
    echo sprintf("Time elapsed: %s\n", format_time($time_ms));
    echo sprintf("Memory used: %s\n", format_bytes($mem_compute));
    echo sprintf("GPU Used: %s", $gpu_used ? "YES ✓" : "NO (CPU)");
    
    if ($expect_gpu && !$gpu_used && !$force_cpu) {
        echo " ⚠ WARNING: Expected GPU but got CPU\n";
    } elseif (!$expect_gpu && $gpu_used) {
        echo " ⚠ WARNING: Expected CPU but got GPU\n";
    } else {
        echo "\n";
    }
    
    // Verify result shape
    $shape = $c->shape();
    echo sprintf("Result shape: [%d, %d]\n", $shape[0], $shape[1]);
    
    // Store for comparison
    $results[] = [
        'name' => $name,
        'size' => $size,
        'elements' => $elements,
        'time_ms' => $time_ms,
        'memory_mb' => $mem_compute / (1024 * 1024),
        'gpu_used' => $gpu_used,
        'expect_gpu' => $expect_gpu,
    ];
    
    // Cleanup
    unset($a, $b, $c);
    gc_collect_cycles();
    
    echo "\n";
}

// Summary and comparison
echo "\n";
echo "╔════════════════════════════════════════════════════════════════╗\n";
echo "║  SUMMARY & COMPARISON                                          ║\n";
echo "╚════════════════════════════════════════════════════════════════╝\n\n";

echo "┌─────────────────────────────────────────────────────────────────────────────────┐\n";
printf("│ %-20s │ %-12s │ %-15s │ %-15s │\n", "Test", "Time", "Memory", "GPU");
echo "├─────────────────────────────────────────────────────────────────────────────────┤\n";

foreach ($results as $r) {
    $gpu_status = $r['gpu_used'] ? '✓ GPU' : '  CPU';
    printf("│ %-20s │ %12s │ %15s │ %15s │\n",
        $r['name'],
        format_time($r['time_ms']),
        format_bytes($r['memory_mb'] * 1024 * 1024),
        $gpu_status
    );
}

echo "└─────────────────────────────────────────────────────────────────────────────────┘\n\n";

// Calculate speedup
$cpu_times = array_filter($results, function($r) { return !$r['gpu_used'] && $r['size'] <= 200; });
$gpu_times = array_filter($results, function($r) { return $r['gpu_used'] && $r['size'] >= 500; });

if (!empty($cpu_times) && !empty($gpu_times)) {
    $avg_cpu_time = array_sum(array_column($cpu_times, 'time_ms')) / count($cpu_times);
    $avg_gpu_time = array_sum(array_column($gpu_times, 'time_ms')) / count($gpu_times);

    // More fair comparison: use similar sized matrices if possible
    $cpu_200 = array_filter($results, function($r) { return $r['size'] == 200; })[0] ?? null;
    $gpu_500 = array_filter($results, function($r) { return $r['size'] == 500; })[0] ?? null;

    if ($cpu_200 && $gpu_500) {
        // Estimate for same size (500x500)
        // 500x500 has 2.5x more result elements than 200x200
        // Time roughly scales with O(n^2.37) or O(n^3) for matmul
        // For rough comparison: time ∝ size^3

        $size_ratio = 500.0 / 200.0;
        $time_scale = pow($size_ratio, 2.8);  // Empirical exponent ~2.8 for mature BLAS

        $estimated_cpu_500 = $cpu_200['time_ms'] * $time_scale;
        $speedup = $estimated_cpu_500 / $gpu_500['time_ms'];

        echo "Performance Analysis:\n";
        echo "─────────────────────\n";
        printf("CPU (200×200): %s\n", format_time($cpu_200['time_ms']));
        printf("GPU (500×500): %s\n", format_time($gpu_500['time_ms']));
        printf("Estimated CPU (500×500): %s (extrapolated)\n", format_time($estimated_cpu_500));
        printf("Speedup (GPU vs estimated CPU): %.1fx\n\n", $speedup);

        if ($speedup > 1) {
            echo "✓ GPU acceleration is EFFECTIVE!\n";
            echo "  GPU is " . number_format($speedup, 1) . "x faster than CPU for large matrices\n";
        } else {
            echo "⚠ GPU overhead not compensated for this workload\n";
            echo "  Consider increasing threshold for GPU usage\n";
        }
    }
} else {
    echo "⚠ Not enough data to compare CPU vs GPU performance\n";
    echo "  Run without FORCE_CPU=1 to see GPU performance\n";
}

echo "\n";
echo "Environment:\n";
echo "─────────────\n";
printf("PHP Version: %s\n", PHP_VERSION);
printf("ZMatrix Extension: %s\n", phpversion('zmatrix'));
printf("CPU Force Mode: %s\n", $force_cpu ? "YES" : "NO");
printf("Memory Peak: %s\n", format_bytes(memory_get_peak_usage(true)));
printf("Timestamp: %s\n", date('Y-m-d H:i:s'));

echo "\n✓ Test completed successfully\n";
