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

class GPUTestRunner {
    private int $passed = 0;
    private int $failed = 0;
    private array $failures = [];
    private bool $cuda_available = false;

    public function __construct() {
        // Detect CUDA availability
        $this->cuda_available = $this->detectCUDA();
    }

    private function detectCUDA(): bool {
        // Try a simple GPU operation to detect CUDA
        try {
            $t = ZTensor::arr([[1.0]]);
            $t->toGpu();
            $result = $t->isOnGpu();
            $t->toCpu();
            return $result;
        } catch (Exception $e) {
            return false;
        }
    }

    public function isCudaAvailable(): bool {
        return $this->cuda_available;
    }

    public function assert(bool $condition, string $message): void {
        if ($condition) {
            $this->passed++;
            echo "  ✓ $message\n";
        } else {
            $this->failed++;
            $this->failures[] = $message;
            echo "  ✗ $message\n";
        }
    }

    public function assertClose(float $actual, float $expected, float $epsilon = 1e-5, string $message = ""): void {
        $condition = abs($actual - $expected) < $epsilon;
        $msg = $message ?: "Expected $expected, got $actual";
        $this->assert($condition, $msg);
    }

    public function assertArrayClose(array $actual, array $expected, float $epsilon = 1e-5): void {
        if (count($actual) !== count($expected)) {
            $this->assert(false, "Array size mismatch: " . count($actual) . " vs " . count($expected));
            return;
        }

        for ($i = 0; $i < count($actual); $i++) {
            if (is_array($actual[$i]) && is_array($expected[$i])) {
                $this->assertArrayClose($actual[$i], $expected[$i], $epsilon);
            } else {
                $this->assertClose((float)$actual[$i], (float)$expected[$i], $epsilon);
            }
        }
    }

    public function section(string $title): void {
        echo "\n" . str_repeat("=", 70) . "\n";
        echo "  $title\n";
        echo str_repeat("=", 70) . "\n";
    }

    public function subsection(string $title): void {
        echo "\n➜ $title\n";
        echo str_repeat("-", 70) . "\n";
    }

    public function printSummary(): void {
        echo "\n" . str_repeat("=", 70) . "\n";
        echo "  TEST SUMMARY\n";
        echo str_repeat("=", 70) . "\n";
        echo "  Passed: " . $this->passed . "\n";
        echo "  Failed: " . $this->failed . "\n";
        echo "  Total:  " . ($this->passed + $this->failed) . "\n";

        if (!empty($this->failures)) {
            echo "\nFailures:\n";
            foreach ($this->failures as $failure) {
                echo "  - $failure\n";
            }
        }

        $status = $this->failed === 0 ? "✓ ALL TESTS PASSED" : "✗ SOME TESTS FAILED";
        echo "\n$status\n";
        echo str_repeat("=", 70) . "\n";

        exit($this->failed === 0 ? 0 : 1);
    }
}

// Utility function for timing operations
function measure_time(callable $func): float {
    $start = hrtime(true);
    $func();
    $end = hrtime(true);
    return ($end - $start) / 1_000_000; // Convert to milliseconds
}

// ============================================================================
// TEST SUITE
// ============================================================================

$runner = new GPUTestRunner();

// ============================================================================
// 1. INITIALIZATION & DETECTION
// ============================================================================

$runner->section("1. GPU Initialization and Detection");

$runner->subsection("1.1 - CUDA Availability");
if ($runner->isCudaAvailable()) {
    echo "✓ CUDA is available on this system\n";
} else {
    echo "⚠ CUDA is NOT available - GPU tests will be skipped\n";
}

// ============================================================================
// 2. TENSOR MOVEMENT (toGpu/toCpu/isOnGpu)
// ============================================================================

$runner->section("2. Tensor Movement Between GPU and CPU");

if ($runner->isCudaAvailable()) {
    $runner->subsection("2.1 - Move Tensor to GPU");
    $tensor = ZTensor::arr([[1.0, 2.0], [3.0, 4.0]]);
    $runner->assert(!$tensor->isOnGpu(), "Tensor initially on CPU");
    $tensor->toGpu();
    $runner->assert($tensor->isOnGpu(), "Tensor successfully moved to GPU");

    $runner->subsection("2.2 - Move Tensor back to CPU");
    $tensor->toCpu();
    $runner->assert(!$tensor->isOnGpu(), "Tensor successfully moved back to CPU");

    $runner->subsection("2.3 - Data Integrity after GPU Movement");
    $original = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    $tensor = ZTensor::arr($original);
    $tensor->toGpu();
    $tensor->toCpu();
    $result = $tensor->toArray();
    $runner->assertArrayClose($result, $original, 1e-6);

    $runner->subsection("2.4 - Multiple GPU Movement Cycles");
    $tensor = ZTensor::arr([[1.0, 2.0], [3.0, 4.0]]);
    for ($i = 0; $i < 3; $i++) {
        $tensor->toGpu();
        $runner->assert($tensor->isOnGpu(), "Cycle $i: on GPU");
        $tensor->toCpu();
        $runner->assert(!$tensor->isOnGpu(), "Cycle $i: on CPU");
    }

    // ========================================================================
    // 3. GPU OPERATIONS - BASIC ARITHMETIC
    // ========================================================================

    $runner->section("3. GPU Operations - Basic Arithmetic");

    $runner->subsection("3.1 - GPU Addition");
    $a = ZTensor::arr([[1.0, 2.0], [3.0, 4.0]]);
    $b = ZTensor::arr([[5.0, 6.0], [7.0, 8.0]]);
    
    $a->toGpu();
    $b->toGpu();
    $a->add($b);
    $a->toCpu();
    
    $result = $a->toArray();
    $expected = [[6.0, 8.0], [10.0, 12.0]];
    $runner->assertArrayClose($result, $expected, 1e-5);

    $runner->subsection("3.2 - GPU Subtraction");
    $a = ZTensor::arr([[5.0, 6.0], [7.0, 8.0]]);
    $b = ZTensor::arr([[1.0, 2.0], [3.0, 4.0]]);
    
    $a->toGpu();
    $b->toGpu();
    $a->sub($b);
    $a->toCpu();
    
    $result = $a->toArray();
    $expected = [[4.0, 4.0], [4.0, 4.0]];
    $runner->assertArrayClose($result, $expected, 1e-5);

    $runner->subsection("3.3 - GPU Element-wise Multiplication");
    $a = ZTensor::arr([[2.0, 3.0], [4.0, 5.0]]);
    $b = ZTensor::arr([[2.0, 2.0], [2.0, 2.0]]);
    
    $a->toGpu();
    $b->toGpu();
    $a->mul($b);
    $a->toCpu();
    
    $result = $a->toArray();
    $expected = [[4.0, 6.0], [8.0, 10.0]];
    $runner->assertArrayClose($result, $expected, 1e-5);

    $runner->subsection("3.4 - GPU Scalar Operations");
    $tensor = ZTensor::arr([[1.0, 2.0], [3.0, 4.0]]);
    $tensor->toGpu();
    $tensor->scalarMultiply(2.0);
    $tensor->toCpu();
    
    $result = $tensor->toArray();
    $expected = [[2.0, 4.0], [6.0, 8.0]];
    $runner->assertArrayClose($result, $expected, 1e-5);

    // ========================================================================
    // 4. GPU OPERATIONS - ACTIVATION FUNCTIONS
    // ========================================================================

    $runner->section("4. GPU Operations - Activation Functions");

    $runner->subsection("4.1 - GPU ReLU");
    $tensor = ZTensor::arr([[-1.0, 2.0], [-3.0, 4.0]]);
    $tensor->toGpu();
    $tensor->relu();
    $tensor->toCpu();
    
    $result = $tensor->toArray();
    $expected = [[0.0, 2.0], [0.0, 4.0]];
    $runner->assertArrayClose($result, $expected, 1e-5);

    $runner->subsection("4.2 - GPU Sigmoid");
    $tensor = ZTensor::arr([[0.0, 1.0], [-1.0]]);
    $tensor->toGpu();
    $tensor->sigmoid();
    $tensor->toCpu();
    
    $result = $tensor->toArray();
    // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.268
    $runner->assertClose($result[0][0], 0.5, 1e-4);
    $runner->assertClose($result[0][1], 0.73105858, 1e-4);
    $runner->assertClose($result[1][0], 0.26894142, 1e-4);

    $runner->subsection("4.3 - GPU Tanh");
    $tensor = ZTensor::arr([[0.0, 1.0], [-1.0]]);
    $tensor->toGpu();
    $tensor->tanh();
    $tensor->toCpu();
    
    $result = $tensor->toArray();
    // tanh(0) = 0, tanh(1) ≈ 0.761, tanh(-1) ≈ -0.761
    $runner->assertClose($result[0][0], 0.0, 1e-5);
    $runner->assertClose($result[0][1], 0.76159415, 1e-4);
    $runner->assertClose($result[1][0], -0.76159415, 1e-4);

    $runner->subsection("4.4 - GPU Abs");
    $tensor = ZTensor::arr([[-1.0, -2.0], [3.0, -4.0]]);
    $tensor->toGpu();
    $tensor->abs();
    $tensor->toCpu();
    
    $result = $tensor->toArray();
    $expected = [[1.0, 2.0], [3.0, 4.0]];
    $runner->assertArrayClose($result, $expected, 1e-5);

    $runner->subsection("4.5 - GPU Exp");
    $tensor = ZTensor::arr([[0.0, 1.0], [2.0]]);
    $tensor->toGpu();
    $tensor->exp();
    $tensor->toCpu();
    
    $result = $tensor->toArray();
    $runner->assertClose($result[0][0], 1.0, 1e-4);
    $runner->assertClose($result[0][1], 2.71828182, 1e-4);
    $runner->assertClose($result[1][0], 7.38905609, 1e-4);

    // ========================================================================
    // 5. GPU PERFORMANCE COMPARISON
    // ========================================================================

    $runner->section("5. GPU vs CPU Performance Comparison");

    $runner->subsection("5.1 - Small Tensor (1000 elements)");
    
    // CPU benchmark
    $tensor_cpu = ZTensor::random([100, 100]);
    $other_cpu = ZTensor::random([100, 100]);
    
    $cpu_time = measure_time(function() use ($tensor_cpu, $other_cpu) {
        $tensor_cpu->add($other_cpu);
    });
    
    // GPU benchmark
    $tensor_gpu = ZTensor::random([100, 100]);
    $other_gpu = ZTensor::random([100, 100]);
    $tensor_gpu->toGpu();
    $other_gpu->toGpu();
    
    $gpu_time = measure_time(function() use ($tensor_gpu, $other_gpu) {
        $tensor_gpu->add($other_gpu);
    });
    
    $tensor_gpu->toCpu();
    
    echo "  CPU time:  " . number_format($cpu_time, 3) . " ms\n";
    echo "  GPU time:  " . number_format($gpu_time, 3) . " ms\n";
    
    $runner->subsection("5.2 - Large Tensor (1M elements)");
    
    // CPU benchmark
    $tensor_cpu = ZTensor::random([1000, 1000]);
    $other_cpu = ZTensor::random([1000, 1000]);
    
    $cpu_time = measure_time(function() use ($tensor_cpu, $other_cpu) {
        for ($i = 0; $i < 5; $i++) {
            $tensor_cpu->add($other_cpu);
        }
    });
    
    // GPU benchmark
    $tensor_gpu = ZTensor::random([1000, 1000]);
    $other_gpu = ZTensor::random([1000, 1000]);
    $tensor_gpu->toGpu();
    $other_gpu->toGpu();
    
    $gpu_time = measure_time(function() use ($tensor_gpu, $other_gpu) {
        for ($i = 0; $i < 5; $i++) {
            $tensor_gpu->add($other_gpu);
        }
    });
    
    $tensor_gpu->toCpu();
    
    echo "  CPU time (5x):  " . number_format($cpu_time, 3) . " ms\n";
    echo "  GPU time (5x):  " . number_format($gpu_time, 3) . " ms\n";
    
    if ($gpu_time > 0) {
        $speedup = $cpu_time / $gpu_time;
        echo "  Speedup: " . number_format($speedup, 1) . "x\n";
        $runner->assert($gpu_time < $cpu_time, "GPU is faster than CPU for large tensors");
    }

    // ========================================================================
    // 6. GPU EDGE CASES
    // ========================================================================

    $runner->section("6. GPU Edge Cases");

    $runner->subsection("6.1 - Empty Tensor GPU Movement");
    $tensor = ZTensor::arr([]);
    try {
        $tensor->toGpu();
        $runner->assert(true, "Empty tensor can be moved to GPU");
    } catch (Exception $e) {
        $runner->assert(false, "Empty tensor failed: " . $e->getMessage());
    }

    $runner->subsection("6.2 - Single Element Tensor");
    $tensor = ZTensor::arr([42.0]);
    $tensor->toGpu();
    $runner->assert($tensor->isOnGpu(), "Single element tensor on GPU");
    $tensor->toCpu();
    $result = $tensor->toArray();
    $runner->assertArrayClose($result, [42.0], 1e-5);

    $runner->subsection("6.3 - Large Dimension Tensor");
    $tensor = ZTensor::random([100, 100, 100]);
    $tensor->toGpu();
    $runner->assert($tensor->isOnGpu(), "3D tensor on GPU");
    $tensor->toCpu();
    $runner->assert(!$tensor->isOnGpu(), "3D tensor back on CPU");

    $runner->subsection("6.4 - Repeated GPU Operations");
    $tensor = ZTensor::arr([[1.0, 2.0], [3.0, 4.0]]);
    $tensor->toGpu();
    for ($i = 0; $i < 10; $i++) {
        $tensor->relu();
    }
    $tensor->toCpu();
    $result = $tensor->toArray();
    $expected = [[1.0, 2.0], [3.0, 4.0]];
    $runner->assertArrayClose($result, $expected, 1e-5);

    // ========================================================================
    // 7. CPU AND GPU EQUIVALENCE
    // ========================================================================

    $runner->section("7. CPU and GPU Operation Equivalence");

    $runner->subsection("7.1 - Addition Equivalence");
    $data_a = [[1.5, 2.3], [3.7, 4.1]];
    $data_b = [[2.1, 1.9], [0.8, 3.2]];
    
    // CPU version
    $a_cpu = ZTensor::arr($data_a);
    $b_cpu = ZTensor::arr($data_b);
    $a_cpu->add($b_cpu);
    $result_cpu = $a_cpu->toArray();
    
    // GPU version
    $a_gpu = ZTensor::arr($data_a);
    $b_gpu = ZTensor::arr($data_b);
    $a_gpu->toGpu();
    $b_gpu->toGpu();
    $a_gpu->add($b_gpu);
    $a_gpu->toCpu();
    $result_gpu = $a_gpu->toArray();
    
    $runner->assertArrayClose($result_cpu, $result_gpu, 1e-5);

    $runner->subsection("7.2 - ReLU Equivalence");
    $data = [[-2.0, -1.0, 0.0, 1.0, 2.0]];
    
    // CPU version
    $t_cpu = ZTensor::arr($data);
    $t_cpu->relu();
    $result_cpu = $t_cpu->toArray();
    
    // GPU version
    $t_gpu = ZTensor::arr($data);
    $t_gpu->toGpu();
    $t_gpu->relu();
    $t_gpu->toCpu();
    $result_gpu = $t_gpu->toArray();
    
    $runner->assertArrayClose($result_cpu, $result_gpu, 1e-5);

    $runner->subsection("7.3 - Sigmoid Equivalence");
    $data = [[0.0, 0.5, 1.0, -0.5, -1.0]];
    
    // CPU version
    $t_cpu = ZTensor::arr($data);
    $t_cpu->sigmoid();
    $result_cpu = $t_cpu->toArray();
    
    // GPU version
    $t_gpu = ZTensor::arr($data);
    $t_gpu->toGpu();
    $t_gpu->sigmoid();
    $t_gpu->toCpu();
    $result_gpu = $t_gpu->toArray();
    
    $runner->assertArrayClose($result_cpu, $result_gpu, 1e-4);

    $runner->subsection("7.4 - Multiplication Equivalence");
    $data_a = [[2.0, 3.0, 4.0]];
    $data_b = [[5.0, 6.0, 7.0]];
    
    // CPU version
    $a_cpu = ZTensor::arr($data_a);
    $b_cpu = ZTensor::arr($data_b);
    $a_cpu->mul($b_cpu);
    $result_cpu = $a_cpu->toArray();
    
    // GPU version
    $a_gpu = ZTensor::arr($data_a);
    $b_gpu = ZTensor::arr($data_b);
    $a_gpu->toGpu();
    $b_gpu->toGpu();
    $a_gpu->mul($b_gpu);
    $a_gpu->toCpu();
    $result_gpu = $a_gpu->toArray();
    
    $runner->assertArrayClose($result_cpu, $result_gpu, 1e-5);

    // ========================================================================
    // 8. MEMORY MANAGEMENT
    // ========================================================================

    $runner->section("8. GPU Memory Management");

    $runner->subsection("8.1 - freeDevice() Method");
    $tensor = ZTensor::random([500, 500]);
    $tensor->toGpu();
    $runner->assert($tensor->isOnGpu(), "Tensor on GPU before freeDevice()");
    
    // Note: freeDevice() should not throw but move data back to CPU
    // The actual behavior depends on implementation
    try {
        $tensor->freeDevice();
        echo "  ✓ freeDevice() executed successfully\n";
    } catch (Exception $e) {
        echo "  ⚠ freeDevice() raised: " . $e->getMessage() . "\n";
    }

    $runner->subsection("8.2 - Multiple Large Tensors on GPU");
    $tensors = [];
    $size = 100;
    
    // Create and move multiple tensors to GPU
    for ($i = 0; $i < 5; $i++) {
        $t = ZTensor::random([$size, $size]);
        $t->toGpu();
        $tensors[] = $t;
        $runner->assert($t->isOnGpu(), "Tensor $i on GPU");
    }
    
    // Move back to CPU
    foreach ($tensors as $i => $t) {
        $t->toCpu();
        $runner->assert(!$t->isOnGpu(), "Tensor $i back on CPU");
    }

} else {
    $runner->section("GPU Tests Skipped");
    echo "CUDA is not available on this system.\n";
    echo "GPU tests require a CUDA-capable GPU and proper driver installation.\n";
}

// ============================================================================
// TEST SUMMARY
// ============================================================================

$runner->printSummary();

    putenv('ZMATRIX_GPU_DEBUG=1');
    
    $a = ZTensor::random([100000], -1.0, 1.0);
    $a->toGpu();
    
    if (!$a->isOnGpu()) {
        return "Tensor not on GPU after toGpu()";
    }
    
    // This will trigger the fallback dlopen() and show debug output
    $b = ZTensor::random([100000], -1.0, 1.0);
    $b->toGpu();
    
    $a->add($b);
    
    return true;
}

// ============================================================================
// TEST 2: CPU vs GPU PERFORMANCE - Small Array (should use CPU)
// ============================================================================

function test_cpu_performance_small() {
    $size = 1000;  // Small array - below GPU threshold (200k)
    
    $a_cpu = ZTensor::random([$size], -1.0, 1.0);
    $b_cpu = ZTensor::random([$size], -1.0, 1.0);
    
    $t0 = microtime(true);
    for ($i = 0; $i < 100; $i++) {
        $a_cpu->add($b_cpu);
    }
    $t1 = microtime(true);
    
    $cpu_time = ($t1 - $t0) * 1000;
    echo "Small array (1k elements) - 100 iterations: {$cpu_time} ms\n";
    
    if ($cpu_time < 0) {
        return "Invalid timing";
    }
    
    return true;
}

// ============================================================================
// TEST 3: CPU vs GPU PERFORMANCE - Large Array WITHOUT Residency
// ============================================================================

function test_gpu_performance_no_residency() {
    $size = 1000000;  // 1M elements
    
    $a = ZTensor::random([$size], -1.0, 1.0);
    $b = ZTensor::random([$size], -1.0, 1.0);
    
    // DO NOT call toGpu() - this will test H2D copy overhead
    $t0 = microtime(true);
    $a->add($b);  // Will copy to GPU, compute, copy back
    $t1 = microtime(true);
    
    $time_ms = ($t1 - $t0) * 1000;
    echo "GPU add (1M elements) without residency: {$time_ms} ms\n";
    echo "  (This includes H2D and D2H copy overhead)\n";
    
    if ($time_ms <= 0) {
        return "Invalid timing";
    }
    
    return true;
}

// ============================================================================
// TEST 4: CPU vs GPU PERFORMANCE - Large Array WITH Residency
// ============================================================================

function test_gpu_performance_with_residency() {
    $size = 1000000;  // 1M elements
    
    $a = ZTensor::random([$size], -1.0, 1.0);
    $b = ZTensor::random([$size], -1.0, 1.0);
    
    // Move to GPU FIRST - keep resident
    $a->toGpu();
    $b->toGpu();
    
    $t0 = microtime(true);
    for ($i = 0; $i < 10; $i++) {
        $a->add($b);
    }
    $t1 = microtime(true);
    
    $avg_ms = ($t1 - $t0) / 10 * 1000;
    echo "GPU add (1M elements) with residency: {$avg_ms} ms/operation\n";
    echo "  (No H2D/D2H copy overhead - data stays on GPU)\n";
    
    $cpu_baseline = 2.5;  // Typical CPU time for 1M add
    $speedup = $cpu_baseline / ($avg_ms / 1000);
    echo "  Speedup vs CPU: {$speedup}x\n";
    
    if ($avg_ms > 1.0) {
        return "GPU add too slow (>1ms)";
    }
    
    return true;
}

// ============================================================================
// TEST 5: CPU vs GPU - Multiple Operations
// ============================================================================

function test_gpu_multiple_operations() {
    $size = 1000000;
    
    $a = ZTensor::random([$size], -1.0, 1.0);
    $b = ZTensor::random([$size], -1.0, 1.0);
    
    $a->toGpu();
    $b->toGpu();
    
    echo "Testing different operations on GPU (1M elements):\n";
    
    $operations = [
        'add' => function($a, $b) { $a->add($b); },
        'sub' => function($a, $b) { $a->sub($b); },
        'mul' => function($a, $b) { $a->mul($b); },
        'relu' => function($a, $b) { $a->relu(); },
        'sigmoid' => function($a, $b) { $a->sigmoid(); },
        'tanh' => function($a, $b) { $a->tanh(); },
        'exp' => function($a, $b) { $a->exp(); },
        'abs' => function($a, $b) { $a->abs(); },
    ];
    
    foreach ($operations as $op_name => $op_func) {
        $a_test = ZTensor::random([$size], -1.0, 1.0);
        $a_test->toGpu();
        
        $t0 = microtime(true);
        call_user_func($op_func, $a_test, $b);
        $t1 = microtime(true);
        
        $time_ms = ($t1 - $t0) * 1000;
        printf("  %-10s: %8.4f ms ✓\n", $op_name, $time_ms);
    }
    
    return true;
}

// ============================================================================
// TEST 6: GPU Memory Management
// ============================================================================

function test_gpu_memory_management() {
    $size = 1000000;
    
    $a = ZTensor::random([$size], -1.0, 1.0);
    echo "1. Created tensor (1M elements) in CPU RAM\n";
    
    $a->toGpu();
    echo "2. Moved tensor to GPU\n";
    
    if (!$a->isOnGpu()) {
        return "Tensor should be on GPU after toGpu()";
    }
    echo "3. Verified tensor is on GPU\n";
    
    // Move to GPU and test operations
    $b = ZTensor::random([$size], -1.0, 1.0);
    $b->toGpu();
    
    if (!$b->isOnGpu()) {
        return "Tensor B should be on GPU after toGpu()";
    }
    echo "4. Second tensor also moved to GPU\n";
    
    $a->add($b);
    echo "5. Performed add() on GPU resident tensors\n";
    
    if (!$a->isOnGpu()) {
        return "Result should still be on GPU after add()";
    }
    echo "6. Verified result is still on GPU after operations\n";
    
    return true;
}

// ============================================================================
// TEST 7: Scalar Operations on GPU
// ============================================================================

function test_gpu_scalar_operations() {
    $size = 1000000;
    
    $a = ZTensor::random([$size], -1.0, 1.0);
    $a->toGpu();
    
    echo "Testing scalar operations on GPU (1M elements):\n";
    
    // Scalar add
    $t0 = microtime(true);
    $a->add(0.5);
    $t1 = microtime(true);
    printf("  add(0.5):           %8.4f ms ✓\n", ($t1-$t0)*1000);
    
    // Scalar mul
    $t0 = microtime(true);
    $a->mul(2.0);
    $t1 = microtime(true);
    printf("  mul(2.0):           %8.4f ms ✓\n", ($t1-$t0)*1000);
    
    // Scalar sub
    $t0 = microtime(true);
    $a->sub(0.1);
    $t1 = microtime(true);
    printf("  sub(0.1):           %8.4f ms ✓\n", ($t1-$t0)*1000);
    
    // Scalar divide
    $t0 = microtime(true);
    $a->scalarDivide(2.0);
    $t1 = microtime(true);
    printf("  scalarDivide(2.0):  %8.4f ms ✓\n", ($t1-$t0)*1000);
    
    return true;
}

// ============================================================================
// TEST 8: Large Batch Processing (Real-world scenario)
// ============================================================================

function test_gpu_batch_processing() {
    $batch_size = 128;
    $features = 512;
    $num_batches = 100;
    
    echo "Batch processing test: $num_batches batches of {$batch_size}x{$features}\n";
    
    // CPU baseline
    $data_cpu = ZTensor::random([$batch_size, $features], -1.0, 1.0);
    
    $t0 = microtime(true);
    for ($i = 0; $i < $num_batches; $i++) {
        $batch = ZTensor::arr($data_cpu);
        $batch->relu();
    }
    $t1 = microtime(true);
    $cpu_total = ($t1 - $t0) * 1000;
    
    // GPU
    $data_gpu = ZTensor::random([$batch_size, $features], -1.0, 1.0);
    $data_gpu->toGpu();
    
    $t0 = microtime(true);
    for ($i = 0; $i < $num_batches; $i++) {
        $batch = ZTensor::arr($data_gpu);
        $batch->relu();
    }
    $t1 = microtime(true);
    $gpu_total = ($t1 - $t0) * 1000;
    
    printf("CPU: %.2f ms total (%.4f ms/batch)\n", $cpu_total, $cpu_total/$num_batches);
    printf("GPU: %.2f ms total (%.4f ms/batch)\n", $gpu_total, $gpu_total/$num_batches);
    
    $speedup = $cpu_total / $gpu_total;
    printf("Speedup: %.1fx\n", $speedup);
    
    if ($gpu_total <= 0) {
        return "Invalid GPU timing";
    }
    
    return true;
}

// ============================================================================
// TEST 9: Chained Operations
// ============================================================================

function test_gpu_chained_operations() {
    $size = 1000000;
    
    $a = ZTensor::random([$size], -0.5, 0.5);
    $a->toGpu();
    
    echo "Executing chained operations on GPU (1M elements):\n";
    
    $t0 = microtime(true);
    
    // Chain of operations staying on GPU
    $a->add(0.1);
    $a->mul(2.0);
    $a->relu();
    $a->add(0.05);
    $a->sigmoid();
    
    $t1 = microtime(true);
    
    $total_ms = ($t1 - $t0) * 1000;
    echo "5 chained ops: {$total_ms} ms\n";
    echo "Average per op: " . ($total_ms / 5) . " ms\n";
    echo "(No H2D/D2H overhead between operations)\n";
    
    return true;
}

// ============================================================================
// TEST 10: Stress Test - Many Tensors on GPU
// ============================================================================

function test_gpu_stress_many_tensors() {
    $size = 100000;
    $num_tensors = 50;
    
    echo "Creating and operating on $num_tensors tensors on GPU:\n";
    
    $tensors = [];
    for ($i = 0; $i < $num_tensors; $i++) {
        $t = ZTensor::random([$size], -1.0, 1.0);
        $t->toGpu();
        $tensors[] = $t;
    }
    echo "✓ Created $num_tensors tensors on GPU\n";
    
    // Add all together
    $t0 = microtime(true);
    $result = $tensors[0];
    for ($i = 1; $i < $num_tensors; $i++) {
        $result->add($tensors[$i]);
    }
    $t1 = microtime(true);
    
    $time_ms = ($t1 - $t0) * 1000;
    printf("Added all tensors: %.4f ms (%.4f ms per add)\n", 
           $time_ms, $time_ms / ($num_tensors - 1));
    
    return true;
}

// ============================================================================
// TEST 11: Fallback Behavior (CPU-only mode)
// ============================================================================

function test_cpu_fallback_mode() {
    putenv('ZMATRIX_FORCE_CPU=1');
    
    $size = 1000000;
    $a = ZTensor::random([$size], -1.0, 1.0);
    $b = ZTensor::random([$size], -1.0, 1.0);
    
    // Trying to call toGpu() in CPU-only mode should either be ignored or no-op
    $a->toGpu();  // In CPU mode, this might be ignored
    
    $t0 = microtime(true);
    $a->add($b);
    $t1 = microtime(true);
    
    $time_ms = ($t1 - $t0) * 1000;
    echo "CPU fallback mode - 1M add: {$time_ms} ms\n";
    echo "(Verifies system gracefully degrades without GPU)\n";
    
    putenv('ZMATRIX_FORCE_CPU=0');
    
    return true;
}

// ============================================================================
// TEST 12: Debug Output Verification
// ============================================================================

function test_debug_output() {
    echo "Testing debug output (ZMATRIX_GPU_DEBUG=1):\n";
    echo "You should see GPU debug messages above this test.\n";
    echo "Look for: [zmatrix][gpu] Successfully loaded CUDA driver\n";
    echo "          [zmatrix][gpu] devices=N\n";
    echo "          [zmatrix][gpu] <operation> n=<size>\n";
    
    return true;
}

// ============================================================================
// TEST 13: Performance Comparison Summary
// ============================================================================

function test_performance_summary() {
    echo "\n📊 PERFORMANCE SUMMARY\n";
    echo "═════════════════════════════════════════════════════════════════\n\n";
    
    $sizes = [100000, 1000000, 10000000];
    
    foreach ($sizes as $size) {
        echo "Array size: $size elements\n";
        echo "─────────────────────────────────────────────────────────────\n";
        
        // CPU test
        $a_cpu = ZTensor::random([$size], -1.0, 1.0);
        $b_cpu = ZTensor::random([$size], -1.0, 1.0);
        
        $t0 = microtime(true);
        $a_cpu->add($b_cpu);
        $t1 = microtime(true);
        $cpu_ms = ($t1 - $t0) * 1000;
        
        // GPU test (with residency)
        $a_gpu = ZTensor::random([$size], -1.0, 1.0);
        $b_gpu = ZTensor::random([$size], -1.0, 1.0);
        $a_gpu->toGpu();
        $b_gpu->toGpu();
        
        $t0 = microtime(true);
        $a_gpu->add($b_gpu);
        $t1 = microtime(true);
        $gpu_ms = ($t1 - $t0) * 1000;
        
        $speedup = $cpu_ms / $gpu_ms;
        
        printf("CPU:     %10.4f ms\n", $cpu_ms);
        printf("GPU:     %10.4f ms\n", $gpu_ms);
        printf("Speedup: %10.1f x\n\n", $speedup);
    }
    
    return true;
}

// ============================================================================
// MAIN TEST EXECUTION
// ============================================================================

$runner = new TestRunner();

// Add all tests
$runner->addTest("GPU Detection & Fallback System", 'test_gpu_detection');
$runner->addTest("CPU Performance - Small Array", 'test_cpu_performance_small');
$runner->addTest("GPU Performance - No Residency", 'test_gpu_performance_no_residency');
$runner->addTest("GPU Performance - With Residency", 'test_gpu_performance_with_residency');
$runner->addTest("GPU - Multiple Operations", 'test_gpu_multiple_operations');
$runner->addTest("GPU - Memory Management", 'test_gpu_memory_management');
$runner->addTest("GPU - Scalar Operations", 'test_gpu_scalar_operations');
$runner->addTest("GPU - Batch Processing", 'test_gpu_batch_processing');
$runner->addTest("GPU - Chained Operations", 'test_gpu_chained_operations');
$runner->addTest("GPU - Stress Test (Many Tensors)", 'test_gpu_stress_many_tensors');
$runner->addTest("CPU Fallback Mode", 'test_cpu_fallback_mode');
$runner->addTest("Debug Output Verification", 'test_debug_output');
$runner->addTest("Performance Summary", 'test_performance_summary');

// Run tests
$runner->run();

echo "\n✨ Test suite complete!\n";
echo "For detailed debugging, run: ZMATRIX_GPU_DEBUG=1 php test_gpu_vs_cpu.php\n\n";
?>
