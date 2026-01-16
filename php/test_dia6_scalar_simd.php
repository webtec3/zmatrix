<?php
use ZMatrix\ZTensor;

echo "════════════════════════════════════════════════════════════════\n";
echo "DIA 6 - SCALAR OPERATIONS SIMD OPTIMIZATION TEST\n";
echo "════════════════════════════════════════════════════════════════\n\n";

// Test 1: Scalar Multiply
echo "[1] Testing scalarMultiply()...\n";
$t1 = ZTensor::full([100], 4.0);
$t1->scalarMultiply(2.5);
$expected = 100 * 10.0; // (4.0 * 2.5) * 100
$actual = $t1->sumtotal();
echo "  Result: sum = $actual, expected = $expected\n";
echo "  Status: " . ($actual == $expected ? "✅ PASS" : "❌ FAIL") . "\n\n";

// Test 2: Scalar Divide
echo "[2] Testing scalarDivide()...\n";
$t2 = ZTensor::full([100], 20.0);
$t2->scalarDivide(2.0);
$expected = 100 * 10.0; // (20.0 / 2.0) * 100
$actual = $t2->sumtotal();
echo "  Result: sum = $actual, expected = $expected\n";
echo "  Status: " . ($actual == $expected ? "✅ PASS" : "❌ FAIL") . "\n\n";

// Test 3: Addition with scalar via add method
echo "[3] Testing addition via add()...\n";
$t3_a = ZTensor::full([100], 5.0);
$t3_b = ZTensor::full([100], 3.0);
$t3_a->add($t3_b);
$expected = 100 * 8.0; // (5.0 + 3.0) * 100
$actual = $t3_a->sumtotal();
echo "  Result: sum = $actual, expected = $expected\n";
echo "  Status: " . ($actual == $expected ? "✅ PASS" : "❌ FAIL") . "\n\n";

// Test 4: Subtraction via sub method
echo "[4] Testing subtraction via sub()...\n";
$t4_a = ZTensor::full([100], 10.0);
$t4_b = ZTensor::full([100], 2.0);
$t4_a->sub($t4_b);
$expected = 100 * 8.0; // (10.0 - 2.0) * 100
$actual = $t4_a->sumtotal();
echo "  Result: sum = $actual, expected = $expected\n";
echo "  Status: " . ($actual == $expected ? "✅ PASS" : "❌ FAIL") . "\n\n";

// Test 5: Element-wise Divide
echo "[5] Testing divide() (element-wise)...\n";
$t5_a = ZTensor::full([100], 20.0);
$t5_b = ZTensor::full([100], 2.0);
$t5_a->divide($t5_b);
$expected = 100 * 10.0; // (20.0 / 2.0) * 100
$actual = $t5_a->sumtotal();
echo "  Result: sum = $actual, expected = $expected\n";
echo "  Status: " . ($actual == $expected ? "✅ PASS" : "❌ FAIL") . "\n\n";

// Test 6: Large array with scalar operations (stress test)
echo "[6] Testing with large array (10M elements)...\n";
$t6 = ZTensor::full([10000000], 1.0);
$t6->scalarMultiply(2.0);
$actual = $t6->sumtotal();
$expected = 20000000.0;
$error_pct = abs($actual - $expected) / $expected * 100;
echo "  Result: sum = $actual, expected = $expected\n";
echo "  Error: $error_pct%\n";
echo "  Status: " . ($error_pct < 0.01 ? "✅ PASS" : "❌ FAIL") . "\n\n";

echo "════════════════════════════════════════════════════════════════\n";
echo "✅ DIA 6 SCALAR OPERATIONS TEST COMPLETE\n";
echo "════════════════════════════════════════════════════════════════\n";
