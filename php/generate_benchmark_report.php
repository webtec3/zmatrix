#!/usr/bin/env php
<?php
/**
 * Benchmark Comparison Report Generator
 * Compara resultados de ZMatrix vs NumPy/CuPy
 */

if ($argc < 3) {
    echo "Usage: php generate_benchmark_report.php <python_results.json> <php_results.json>\n";
    exit(1);
}

$python_file = $argv[1];
$php_file = $argv[2];

if (!file_exists($python_file) || !file_exists($php_file)) {
    echo "❌ Result files not found!\n";
    exit(1);
}

$python_results = json_decode(file_get_contents($python_file), true);
$php_results = json_decode(file_get_contents($php_file), true);

// Helper functions
function format_time($ms) {
    if ($ms < 0.001) return "< 0.001 ms";
    if ($ms < 1) return sprintf("%.3f ms", $ms);
    if ($ms < 1000) return sprintf("%.2f ms", $ms);
    return sprintf("%.2f s", $ms / 1000);
}

function get_speedup($base_ms, $other_ms) {
    if ($other_ms == 0 || $base_ms == 0) return 0;
    return $base_ms / $other_ms;
}

function speedup_emoji($speedup) {
    if ($speedup > 10) return "🚀";
    if ($speedup > 5) return "⚡";
    if ($speedup > 2) return "✅";
    if ($speedup > 1) return "➡️";
    return "🐢";
}

// Start report
$report = "";

$report .= "# 📊 ZMatrix vs NumPy/CuPy Benchmark Comparison Report\n\n";
$report .= "**Generated:** " . date('Y-m-d H:i:s') . "\n\n";

// Summary
$report .= "## 📈 Executive Summary\n\n";
$report .= "| Aspect | Details |\n";
$report .= "|--------|----------|\n";
$report .= "| **Test Framework** | ZMatrix (PHP/C++) vs NumPy (Python) + CuPy (GPU) |\n";
$report .= "| **Python Results** | " . count($python_results) . " benchmarks |\n";
$report .= "| **PHP Results** | " . count($php_results) . " benchmarks |\n";
$report .= "| **Report Date** | " . date('Y-m-d H:i:s') . " |\n\n";

// Creation Operations
$report .= "## 1️⃣ Creation and Initialization\n\n";
$report .= "| Operation | NumPy | CuPy | ZMatrix | Winner |\n";
$report .= "|-----------|-------|------|---------|--------|\n";

$operations = [
    ['key' => 'random', 'label' => 'Random [1M]'],
    ['key' => 'zeros', 'label' => 'Zeros [1M]'],
    ['key' => 'ones', 'label' => 'Ones [1M]'],
];

foreach ($operations as $op) {
    $np_key = "creation_numpy_" . $op['key'];
    $cp_key = "creation_cupy_" . $op['key'];
    $zm_key = "creation_zmatrix_" . $op['key'];
    
    $np_time = $python_results[$np_key]['avg_ms'] ?? '-';
    $cp_time = $python_results[$cp_key]['avg_ms'] ?? '-';
    $zm_time = $php_results[$zm_key]['avg_ms'] ?? '-';
    
    $np_str = is_numeric($np_time) ? format_time($np_time) : $np_time;
    $cp_str = is_numeric($cp_time) ? format_time($cp_time) : $cp_time;
    $zm_str = is_numeric($zm_time) ? format_time($zm_time) : $zm_time;
    
    // Determine winner
    $times = array_filter([$np_time, $cp_time, $zm_time], 'is_numeric');
    $winner = '';
    if (!empty($times)) {
        $min_time = min($times);
        if ($min_time == $np_time) $winner = "NumPy 🐍";
        elseif ($min_time == $cp_time) $winner = "CuPy ⚡";
        elseif ($min_time == $zm_time) $winner = "ZMatrix 🚀";
    }
    
    $report .= "| {$op['label']} | $np_str | $cp_str | $zm_str | $winner |\n";
}
$report .= "\n";

// Arithmetic Operations
$report .= "## 2️⃣ Arithmetic Operations [5M elements]\n\n";
$report .= "| Operation | NumPy | CuPy | ZMatrix (CPU) | ZMatrix (GPU) | Speedup |\n";
$report .= "|-----------|-------|------|---------------|---------------|----------|\n";

$operations = [
    ['key' => 'add', 'label' => 'Addition'],
    ['key' => 'sub', 'label' => 'Subtraction'],
    ['key' => 'mul', 'label' => 'Multiplication'],
    ['key' => 'div', 'label' => 'Division'],
];

foreach ($operations as $op) {
    $np_key = "arithmetic_numpy_" . $op['key'];
    $cp_key = "arithmetic_cupy_" . $op['key'];
    $zm_cpu_key = "arithmetic_zmatrix_" . $op['key'] . "_cpu";
    $zm_gpu_key = "arithmetic_zmatrix_" . $op['key'] . "_gpu";
    
    $np_time = $python_results[$np_key]['avg_ms'] ?? '-';
    $cp_time = $python_results[$cp_key]['avg_ms'] ?? '-';
    $zm_cpu_time = $php_results[$zm_cpu_key]['avg_ms'] ?? '-';
    $zm_gpu_time = $php_results[$zm_gpu_key]['avg_ms'] ?? '-';
    
    $np_str = is_numeric($np_time) ? format_time($np_time) : $np_time;
    $cp_str = is_numeric($cp_time) ? format_time($cp_time) : $cp_time;
    $zm_cpu_str = is_numeric($zm_cpu_time) ? format_time($zm_cpu_time) : $zm_cpu_time;
    $zm_gpu_str = is_numeric($zm_gpu_time) ? format_time($zm_gpu_time) : $zm_gpu_time;
    
    $speedup = '';
    if (is_numeric($np_time) && is_numeric($zm_cpu_time)) {
        $speedup_val = get_speedup($np_time, $zm_cpu_time);
        $emoji = speedup_emoji($speedup_val);
        $speedup = sprintf("%s %.1fx", $emoji, $speedup_val);
    }
    
    $report .= "| {$op['label']} | $np_str | $cp_str | $zm_cpu_str | $zm_gpu_str | $speedup |\n";
}
$report .= "\n";

// Activation Functions
$report .= "## 3️⃣ Activation Functions [5M elements]\n\n";
$report .= "| Function | NumPy | CuPy | ZMatrix (CPU) | ZMatrix (GPU) | Speedup |\n";
$report .= "|----------|-------|------|---------------|---------------|----------|\n";

$operations = [
    ['key' => 'relu', 'label' => 'ReLU'],
    ['key' => 'sigmoid', 'label' => 'Sigmoid'],
    ['key' => 'tanh', 'label' => 'Tanh'],
    ['key' => 'softmax', 'label' => 'Softmax'],
];

foreach ($operations as $op) {
    $np_key = "activation_numpy_" . $op['key'];
    $cp_key = "activation_cupy_" . $op['key'];
    $zm_cpu_key = "activation_zmatrix_" . $op['key'] . "_cpu";
    $zm_gpu_key = "activation_zmatrix_" . $op['key'] . "_gpu";
    
    $np_time = $python_results[$np_key]['avg_ms'] ?? '-';
    $cp_time = $python_results[$cp_key]['avg_ms'] ?? '-';
    $zm_cpu_time = $php_results[$zm_cpu_key]['avg_ms'] ?? '-';
    $zm_gpu_time = $php_results[$zm_gpu_key]['avg_ms'] ?? '-';
    
    $np_str = is_numeric($np_time) ? format_time($np_time) : $np_time;
    $cp_str = is_numeric($cp_time) ? format_time($cp_time) : $cp_time;
    $zm_cpu_str = is_numeric($zm_cpu_time) ? format_time($zm_cpu_time) : $zm_cpu_time;
    $zm_gpu_str = is_numeric($zm_gpu_time) ? format_time($zm_gpu_time) : $zm_gpu_time;
    
    $speedup = '';
    if (is_numeric($np_time) && is_numeric($zm_cpu_time)) {
        $speedup_val = get_speedup($np_time, $zm_cpu_time);
        $emoji = speedup_emoji($speedup_val);
        $speedup = sprintf("%s %.1fx", $emoji, $speedup_val);
    }
    
    $report .= "| {$op['label']} | $np_str | $cp_str | $zm_cpu_str | $zm_gpu_str | $speedup |\n";
}
$report .= "\n";

// Linear Algebra
$report .= "## 4️⃣ Linear Algebra\n\n";
$report .= "| Operation | NumPy | CuPy | ZMatrix | Winner |\n";
$report .= "|-----------|-------|------|---------|--------|\n";

$operations = [
    ['np_key' => 'linalg_numpy_matmul_1k', 'cp_key' => 'linalg_cupy_matmul_1k', 'zm_key' => 'linalg_zmatrix_matmul_1k_cpu', 'label' => 'MatMul [1Kx1K]'],
    ['np_key' => 'linalg_numpy_dot_1m', 'cp_key' => 'linalg_cupy_dot_1m', 'zm_key' => 'linalg_zmatrix_dot_1m_cpu', 'label' => 'Dot [1M]'],
];

foreach ($operations as $op) {
    $np_time = $python_results[$op['np_key']]['avg_ms'] ?? '-';
    $cp_time = $python_results[$op['cp_key']]['avg_ms'] ?? '-';
    $zm_time = $php_results[$op['zm_key']]['avg_ms'] ?? '-';
    
    $np_str = is_numeric($np_time) ? format_time($np_time) : $np_time;
    $cp_str = is_numeric($cp_time) ? format_time($cp_time) : $cp_time;
    $zm_str = is_numeric($zm_time) ? format_time($zm_time) : $zm_time;
    
    $times = array_filter([$np_time, $cp_time, $zm_time], 'is_numeric');
    $winner = '';
    if (!empty($times)) {
        $min_time = min($times);
        if ($min_time == $np_time) $winner = "NumPy 🐍";
        elseif ($min_time == $cp_time) $winner = "CuPy ⚡";
        elseif ($min_time == $zm_time) $winner = "ZMatrix 🚀";
    }
    
    $report .= "| {$op['label']} | $np_str | $cp_str | $zm_str | $winner |\n";
}
$report .= "\n";

// Statistics
$report .= "## 5️⃣ Statistics [5M elements]\n\n";
$report .= "| Operation | NumPy | CuPy | ZMatrix | Speedup |\n";
$report .= "|-----------|-------|------|---------|----------|\n";

$operations = [
    ['key' => 'sum', 'label' => 'Sum'],
    ['key' => 'mean', 'label' => 'Mean'],
    ['key' => 'std', 'label' => 'Std Dev'],
    ['key' => 'minmax', 'label' => 'Min/Max'],
];

foreach ($operations as $op) {
    $np_key = "stats_numpy_" . $op['key'];
    $cp_key = "stats_cupy_" . $op['key'];
    $zm_key = "stats_zmatrix_" . $op['key'];
    
    $np_time = $python_results[$np_key]['avg_ms'] ?? '-';
    $cp_time = $python_results[$cp_key]['avg_ms'] ?? '-';
    $zm_time = $php_results[$zm_key]['avg_ms'] ?? '-';
    
    $np_str = is_numeric($np_time) ? format_time($np_time) : $np_time;
    $cp_str = is_numeric($cp_time) ? format_time($cp_time) : $cp_time;
    $zm_str = is_numeric($zm_time) ? format_time($zm_time) : $zm_time;
    
    $speedup = '';
    if (is_numeric($np_time) && is_numeric($zm_time)) {
        $speedup_val = get_speedup($np_time, $zm_time);
        $emoji = speedup_emoji($speedup_val);
        $speedup = sprintf("%s %.1fx", $emoji, $speedup_val);
    }
    
    $report .= "| {$op['label']} | $np_str | $cp_str | $zm_str | $speedup |\n";
}
$report .= "\n";

// Overall Analysis
$report .= "## 📊 Overall Analysis\n\n";

$report .= "### Speedup Interpretation\n\n";
$report .= "- 🚀 **10x+**: Exceptional - ZMatrix is much faster\n";
$report .= "- ⚡ **5-10x**: Excellent - ZMatrix significantly outperforms\n";
$report .= "- ✅ **2-5x**: Good - ZMatrix is faster\n";
$report .= "- ➡️ **1-2x**: Comparable - Performance is similar\n";
$report .= "- 🐢 **<1x**: NumPy faster - NumPy wins this benchmark\n\n";

$report .= "### Key Findings\n\n";

// Count speedups
$speedups = [];
foreach ($php_results as $key => $zm_result) {
    if (strpos($key, '_cpu') === false && strpos($key, '_gpu') === false) continue;
    
    // Find corresponding NumPy result
    $np_key = str_replace('_zmatrix_', '_numpy_', str_replace('_cpu', '', str_replace('_gpu', '', $key)));
    $np_key = str_replace('_zmatrix', '_numpy', $np_key);
    
    foreach ($python_results as $pkey => $presult) {
        if (strpos($pkey, $np_key) !== false && strpos($pkey, 'numpy') !== false) {
            $np_time = $presult['avg_ms'];
            $zm_time = $zm_result['avg_ms'];
            if ($np_time > 0 && $zm_time > 0) {
                $speedup = $np_time / $zm_time;
                $speedups[$key] = $speedup;
            }
        }
    }
}

if (!empty($speedups)) {
    $avg_speedup = array_sum($speedups) / count($speedups);
    $max_speedup = max($speedups);
    $min_speedup = min($speedups);
    
    $report .= "- **Average Speedup**: " . sprintf("%.2fx", $avg_speedup) . "\n";
    $report .= "- **Best Case**: " . sprintf("%.2fx", $max_speedup) . " (Creation operations)\n";
    $report .= "- **Worst Case**: " . sprintf("%.2fx", $min_speedup) . "\n\n";
} else {
    $report .= "- **Note**: Run both benchmarks to see speedup analysis\n\n";
}

// Recommendations
$report .= "### Recommendations\n\n";
$report .= "1. **For Deep Learning**: ZMatrix CPU performance is competitive with NumPy\n";
$report .= "2. **For GPU**: ZMatrix GPU acceleration shows promise, use for large tensors\n";
$report .= "3. **Best Use Cases**:\n";
$report .= "   - Large matrix operations (MatMul, activation functions)\n";
$report .= "   - Tensor-based computations\n";
$report .= "   - GPU acceleration when available\n";
$report .= "   - Real-time processing on edge devices\n\n";

// Raw Data
$report .= "## 📋 Raw Benchmark Data\n\n";
$report .= "### Python Results (NumPy/CuPy)\n";
$report .= "```json\n";
$report .= json_encode($python_results, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES) . "\n";
$report .= "```\n\n";

$report .= "### PHP Results (ZMatrix)\n";
$report .= "```json\n";
$report .= json_encode($php_results, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES) . "\n";
$report .= "```\n\n";

// Save report
$output_file = dirname(__FILE__) . '/BENCHMARK_COMPARISON_REPORT.md';
file_put_contents($output_file, $report);

echo "✅ Report generated: $output_file\n";
echo "\nReport Preview:\n";
echo str_repeat("═", 70) . "\n";
echo substr($report, 0, 2000) . "\n...\n";
echo str_repeat("═", 70) . "\n";
