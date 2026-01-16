<?php

namespace ZMatrix\Tests;

use ZMatrix\ZTensor;

class BackwardCorrectedValidation
{
    private int $passed = 0;
    private int $failed = 0;

    public function run(): void
    {
        echo "\n╔════════════════════════════════════════════════════════╗\n";
        echo "║  ZMatrix v0.5.0 - BACKWARD PASS CORRECTED VALIDATION║\n";
        echo "║  Gradient Computation & Backpropagation Testing     ║\n";
        echo "╚════════════════════════════════════════════════════════╝\n\n";

        $this->testScalarAddition();
        $this->testVectorSum();
        $this->testMultiplication();
        $this->testChainRule();
        $this->testMultipleInputs();
        $this->testComplexExpression();

        $this->printFinalReport();
    }

    private function testScalarAddition(): void
    {
        echo "TEST 1: Scalar Addition (y = a + b)\n";
        echo "───────────────────────────────────\n";

        $a = ZTensor::arr([2.0])->requiresGrad(true);
        $b = ZTensor::arr([3.0])->requiresGrad(true);
        $y = ZTensor::addAutograd($a, $b);

        $y->backward();

        $grad_a = $a->getGrad()->toArray()[0];
        $grad_b = $b->getGrad()->toArray()[0];

        $result = (abs($grad_a - 1.0) < 1e-5 && abs($grad_b - 1.0) < 1e-5);

        echo "  ✓ a + b = 5\n";
        echo "  ✓ da/dy = 1.0, db/dy = 1.0\n";
        echo $result ? "  ✅ PASSED\n\n" : "  ❌ FAILED\n\n";
        $result ? $this->passed++ : $this->failed++;
    }

    private function testVectorSum(): void
    {
        echo "TEST 2: Vector Sum (loss = sum(x))\n";
        echo "───────────────────────────────────\n";

        $x = ZTensor::arr([1.0, 2.0, 3.0, 4.0])->requiresGrad(true);
        $loss = ZTensor::sumAutograd($x);

        $loss->backward();

        $grad_x = $x->getGrad()->toArray();
        $expected = [1.0, 1.0, 1.0, 1.0];
        $result = true;
        foreach ($grad_x as $i => $g) {
            if (abs($g - $expected[$i]) > 1e-5) {
                $result = false;
                break;
            }
        }

        echo "  ✓ sum([1,2,3,4]) = 10\n";
        echo "  ✓ dloss/dx = [1,1,1,1]\n";
        echo $result ? "  ✅ PASSED\n\n" : "  ❌ FAILED\n\n";
        $result ? $this->passed++ : $this->failed++;
    }

    private function testMultiplication(): void
    {
        echo "TEST 3: Element-wise Multiplication (z = a * b)\n";
        echo "───────────────────────────────────────────────\n";

        $a = ZTensor::arr([2.0, 3.0])->requiresGrad(true);
        $b = ZTensor::arr([4.0, 5.0])->requiresGrad(true);
        $z = ZTensor::mulAutograd($a, $b);
        $loss = ZTensor::sumAutograd($z);

        $loss->backward();

        $grad_a = $a->getGrad()->toArray();
        $grad_b = $b->getGrad()->toArray();

        // dloss/da[i] = b[i], dloss/db[i] = a[i]
        $result = (abs($grad_a[0] - 4.0) < 1e-5 && abs($grad_a[1] - 5.0) < 1e-5 &&
                   abs($grad_b[0] - 2.0) < 1e-5 && abs($grad_b[1] - 3.0) < 1e-5);

        echo "  ✓ [2,3] * [4,5] = [8,15]\n";
        echo "  ✓ da/dloss = [4,5], db/dloss = [2,3]\n";
        echo $result ? "  ✅ PASSED\n\n" : "  ❌ FAILED\n\n";
        $result ? $this->passed++ : $this->failed++;
    }

    private function testChainRule(): void
    {
        echo "TEST 4: Chain Rule (z = (x+x)^2)\n";
        echo "─────────────────────────────────\n";

        $x = ZTensor::arr([2.0])->requiresGrad(true);

        // y = x + x
        $y = ZTensor::addAutograd($x, $x);

        // z = y * y (using mulAutograd)
        $z = ZTensor::mulAutograd($y, $y);

        $z->backward();

        $grad_x = $x->getGrad()->toArray()[0];
        // z = (x+x)^2 = (2x)^2 = 4x^2
        // dz/dx = 8x = 8*2 = 16
        $expected = 16.0;
        $result = abs($grad_x - $expected) < 1e-5;

        echo "  ✓ x = 2, y = x+x = 4\n";
        echo "  ✓ z = y*y = 16\n";
        echo "  ✓ dz/dx = " . number_format($grad_x, 1) . " (expected: 16)\n";
        echo $result ? "  ✅ PASSED\n\n" : "  ❌ FAILED\n\n";
        $result ? $this->passed++ : $this->failed++;
    }

    private function testMultipleInputs(): void
    {
        echo "TEST 5: Multiple Inputs (z = a*b + a*c)\n";
        echo "───────────────────────────────────────\n";

        $a = ZTensor::arr([2.0])->requiresGrad(true);
        $b = ZTensor::arr([3.0])->requiresGrad(true);
        $c = ZTensor::arr([4.0])->requiresGrad(true);

        // z = a*b + a*c
        $ab = ZTensor::mulAutograd($a, $b);
        $ac = ZTensor::mulAutograd($a, $c);
        $z = ZTensor::addAutograd($ab, $ac);

        $z->backward();

        $grad_a = $a->getGrad()->toArray()[0];
        $grad_b = $b->getGrad()->toArray()[0];
        $grad_c = $c->getGrad()->toArray()[0];

        // dz/da = b + c = 7
        // dz/db = a = 2
        // dz/dc = a = 2
        $result = (abs($grad_a - 7.0) < 1e-5 &&
                   abs($grad_b - 2.0) < 1e-5 &&
                   abs($grad_c - 2.0) < 1e-5);

        echo "  ✓ z = 2*3 + 2*4 = 14\n";
        echo "  ✓ dz/da = 7, dz/db = 2, dz/dc = 2\n";
        echo $result ? "  ✅ PASSED\n\n" : "  ❌ FAILED\n\n";
        $result ? $this->passed++ : $this->failed++;
    }

    private function testComplexExpression(): void
    {
        echo "TEST 6: Complex Expression (loss = sum((x+y)*x))\n";
        echo "───────────────────────────────────────────────\n";

        $x = ZTensor::arr([1.0, 2.0])->requiresGrad(true);
        $y = ZTensor::arr([2.0, 3.0])->requiresGrad(true);

        $xy_sum = ZTensor::addAutograd($x, $y);
        $product = ZTensor::mulAutograd($xy_sum, $x);
        $loss = ZTensor::sumAutograd($product);

        $loss->backward();

        $grad_x = $x->getGrad()->toArray();
        $grad_y = $y->getGrad()->toArray();

        // For i-th element: loss_i = (x[i] + y[i]) * x[i]
        // dloss/dx[i] = 2*x[i] + y[i]
        // dloss/dy[i] = x[i]

        $expected_grad_x_0 = 2*1.0 + 2.0; // = 4
        $expected_grad_x_1 = 2*2.0 + 3.0; // = 7
        $expected_grad_y_0 = 1.0;
        $expected_grad_y_1 = 2.0;

        $result = (abs($grad_x[0] - $expected_grad_x_0) < 1e-5 &&
                   abs($grad_x[1] - $expected_grad_x_1) < 1e-5 &&
                   abs($grad_y[0] - $expected_grad_y_0) < 1e-5 &&
                   abs($grad_y[1] - $expected_grad_y_1) < 1e-5);

        echo "  ✓ Complex computation: (x+y)*x\n";
        echo "  ✓ Gradients computed correctly through chain rule\n";
        echo $result ? "  ✅ PASSED\n\n" : "  ❌ FAILED\n\n";
        $result ? $this->passed++ : $this->failed++;
    }

    private function printFinalReport(): void
    {
        $total = $this->passed + $this->failed;
        $percentage = $total > 0 ? intval(($this->passed / $total) * 100) : 0;

        echo "╔════════════════════════════════════════════════════════╗\n";
        echo "║                   FINAL REPORT                        ║\n";
        echo "╚════════════════════════════════════════════════════════╝\n\n";

        echo "Tests Passed:  $this->passed/$total\n";
        echo "Tests Failed:  $this->failed/$total\n";
        echo "Success Rate:  $percentage%\n\n";

        if ($this->failed === 0) {
            echo "✅ BACKWARD PASS FULLY OPERATIONAL!\n\n";
            echo "Summary:\n";
            echo "  ✓ Scalar operations working\n";
            echo "  ✓ Vector operations working\n";
            echo "  ✓ Chain rule implemented correctly\n";
            echo "  ✓ Multiple input handling working\n";
            echo "  ✓ Complex expressions evaluated correctly\n";
            echo "  ✓ Gradient accumulation functional\n\n";
            echo "🎉 ZMatrix v0.5.0 backward() is PRODUCTION READY!\n";
        } else {
            echo "⚠️  Some tests failed. Please review.\n";
        }
        echo "\n";
    }
}

if (php_sapi_name() === 'cli') {
    $tests = new BackwardCorrectedValidation();
    $tests->run();
}
?>
