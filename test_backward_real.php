<?php
/**
 * ZMatrix Backward Pass Real Gradient Computation Tests
 * 
 * This tests ACTUAL gradient computation, not just infrastructure.
 */

namespace ZMatrix\Tests;

use ZMatrix\ZTensor;

class BackwardRealTests
{
    private int $passed = 0;
    private int $failed = 0;

    public function run(): void
    {
        echo "════════════════════════════════════════════════\n";
        echo "🧪 BACKWARD PASS REAL GRADIENT TESTS\n";
        echo "════════════════════════════════════════════════\n\n";

        $this->testSimpleAdd();
        $this->testSum();
        $this->testMultiplication();
        $this->testComposition();

        $this->printSummary();
    }

    /**
     * Test 1: Simple addition backward
     * y = a + b
     * dy/da = 1, dy/db = 1
     */
    private function testSimpleAdd(): void
    {
        echo "📌 TEST 1: Simple Addition (y = a + b)\n";
        
        try {
            $a = ZTensor::arr([2.0])->requiresGrad(true);
            $b = ZTensor::arr([3.0])->requiresGrad(true);
            
            echo "   a = 2.0, b = 3.0\n";
            echo "   y = a + b = ";
            
            $y = ZTensor::addAutograd($a, $b);
            echo $y->toArray()[0] . "\n";
            
            // Call backward
            echo "   Calling backward()...\n";
            $y->backward();
            
            // Check gradients
            $grad_a = $a->getGrad();
            $grad_b = $b->getGrad();
            
            if ($grad_a === null || $grad_b === null) {
                echo "   ❌ FAILED: Gradients are null after backward()\n";
                echo "   grad_a: " . ($grad_a === null ? "NULL" : json_encode($grad_a->toArray())) . "\n";
                echo "   grad_b: " . ($grad_b === null ? "NULL" : json_encode($grad_b->toArray())) . "\n\n";
                $this->failed++;
                return;
            }
            
            $grad_a_val = $grad_a->toArray()[0];
            $grad_b_val = $grad_b->toArray()[0];
            
            echo "   dy/da = " . $grad_a_val . " (expected: 1.0)\n";
            echo "   dy/db = " . $grad_b_val . " (expected: 1.0)\n";
            
            if (abs($grad_a_val - 1.0) < 1e-5 && abs($grad_b_val - 1.0) < 1e-5) {
                echo "   ✅ PASSED\n\n";
                $this->passed++;
            } else {
                echo "   ❌ FAILED: Incorrect gradient values\n\n";
                $this->failed++;
            }
        } catch (Exception $e) {
            echo "   ❌ FAILED: " . $e->getMessage() . "\n\n";
            $this->failed++;
        }
    }

    /**
     * Test 2: Sum backward
     * loss = sum(x)
     * dloss/dx[i] = 1 for all i
     */
    private function testSum(): void
    {
        echo "📌 TEST 2: Sum Backward (loss = sum(x))\n";
        
        try {
            $x = ZTensor::arr([1.0, 2.0, 3.0])->requiresGrad(true);
            echo "   x = [1, 2, 3]\n";
            
            $loss = ZTensor::sumAutograd($x);
            echo "   loss = sum(x) = " . $loss->toArray()[0] . "\n";
            
            // Call backward
            echo "   Calling backward()...\n";
            $loss->backward();
            
            // Check gradients
            $grad = $x->getGrad();
            
            if ($grad === null) {
                echo "   ❌ FAILED: Gradient is null\n\n";
                $this->failed++;
                return;
            }
            
            $grad_vals = $grad->toArray();
            echo "   dloss/dx = " . json_encode($grad_vals) . "\n";
            echo "   Expected: [1, 1, 1]\n";
            
            $expected = [1.0, 1.0, 1.0];
            $match = true;
            foreach ($grad_vals as $i => $val) {
                if (abs($val - $expected[$i]) > 1e-5) {
                    $match = false;
                    break;
                }
            }
            
            if ($match) {
                echo "   ✅ PASSED\n\n";
                $this->passed++;
            } else {
                echo "   ❌ FAILED: Incorrect gradients\n\n";
                $this->failed++;
            }
        } catch (Exception $e) {
            echo "   ❌ FAILED: " . $e->getMessage() . "\n\n";
            $this->failed++;
        }
    }

    /**
     * Test 3: Multiplication backward
     * z = a * b
     * dz/da = b, dz/db = a
     */
    private function testMultiplication(): void
    {
        echo "📌 TEST 3: Multiplication (z = a * b)\n";
        
        try {
            $a = ZTensor::arr([2.0])->requiresGrad(true);
            $b = ZTensor::arr([3.0])->requiresGrad(true);
            
            echo "   a = 2.0, b = 3.0\n";
            
            $z = ZTensor::mulAutograd($a, $b);
            echo "   z = a * b = " . $z->toArray()[0] . "\n";
            
            // Call backward
            echo "   Calling backward()...\n";
            $z->backward();
            
            // Check gradients
            $grad_a = $a->getGrad();
            $grad_b = $b->getGrad();
            
            if ($grad_a === null || $grad_b === null) {
                echo "   ❌ FAILED: Gradients are null\n\n";
                $this->failed++;
                return;
            }
            
            $grad_a_val = $grad_a->toArray()[0];
            $grad_b_val = $grad_b->toArray()[0];
            
            echo "   dz/da = " . $grad_a_val . " (expected: " . 3.0 . ")\n";
            echo "   dz/db = " . $grad_b_val . " (expected: " . 2.0 . ")\n";
            
            if (abs($grad_a_val - 3.0) < 1e-5 && abs($grad_b_val - 2.0) < 1e-5) {
                echo "   ✅ PASSED\n\n";
                $this->passed++;
            } else {
                echo "   ❌ FAILED: Incorrect gradients\n\n";
                $this->failed++;
            }
        } catch (Exception $e) {
            echo "   ❌ FAILED: " . $e->getMessage() . "\n\n";
            $this->failed++;
        }
    }

    /**
     * Test 4: Composition (chain rule)
     * x = [1, 2]
     * y = x + x = [2, 4]
     * loss = sum(y) = 6
     * dloss/dx = [2, 2] (from chain rule)
     */
    private function testComposition(): void
    {
        echo "📌 TEST 4: Composition (Chain Rule)\n";
        
        try {
            $x = ZTensor::arr([1.0, 2.0])->requiresGrad(true);
            echo "   x = [1, 2]\n";
            
            // y = x + x
            $y = ZTensor::addAutograd($x, $x);
            echo "   y = x + x = " . json_encode($y->toArray()) . "\n";
            
            // loss = sum(y)
            $loss = ZTensor::sumAutograd($y);
            echo "   loss = sum(y) = " . $loss->toArray()[0] . "\n";
            
            // Backward
            echo "   Calling backward()...\n";
            $loss->backward();
            
            // Check gradient
            $grad = $x->getGrad();
            
            if ($grad === null) {
                echo "   ❌ FAILED: Gradient is null\n\n";
                $this->failed++;
                return;
            }
            
            $grad_vals = $grad->toArray();
            echo "   dloss/dx = " . json_encode($grad_vals) . "\n";
            echo "   Expected: [2, 2] (each element contributes twice)\n";
            
            if (abs($grad_vals[0] - 2.0) < 1e-5 && abs($grad_vals[1] - 2.0) < 1e-5) {
                echo "   ✅ PASSED\n\n";
                $this->passed++;
            } else {
                echo "   ❌ FAILED: Incorrect gradients (chain rule not working)\n\n";
                $this->failed++;
            }
        } catch (Exception $e) {
            echo "   ❌ FAILED: " . $e->getMessage() . "\n\n";
            $this->failed++;
        }
    }

    private function printSummary(): void
    {
        $total = $this->passed + $this->failed;
        
        echo "════════════════════════════════════════════════\n";
        echo "📊 SUMMARY\n";
        echo "════════════════════════════════════════════════\n\n";
        echo "✅ Passed: $this->passed/$total\n";
        echo "❌ Failed: $this->failed/$total\n\n";
        
        if ($this->failed === 0) {
            echo "🎉 ALL TESTS PASSED! Backward pass is working!\n";
        } else {
            echo "⚠️  Some tests failed. Gradients not propagating correctly.\n";
        }
    }
}

// Run tests
if (php_sapi_name() === 'cli') {
    $tests = new BackwardRealTests();
    $tests->run();
}
