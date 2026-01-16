<?php
use ZMatrix\ZTensor;

// Test sum() function with known values

// Test 1: Simple 2D array [1,2,3; 4,5,6] = 21
$tensor = ZTensor::arr([
    [1, 2, 3],
    [4, 5, 6]
]);
$actual = $tensor->sumtotal();
echo "Test 1: ZTensor::arr([[1,2,3], [4,5,6]])\n";
echo "  Expected: 21.0\n";
echo "  Actual: $actual\n";
echo "  Status: " . ($actual == 21.0 ? "✅ PASS" : "❌ FAIL") . "\n\n";

// Test 2: ZTensor::full([100], 2.5) = 250
$tensor2 = ZTensor::full([100], 2.5);
$actual2 = $tensor2->sumtotal();
echo "Test 2: ZTensor::full([100], 2.5)\n";
echo "  Expected: 250.0\n";
echo "  Actual: $actual2\n";
echo "  Status: " . ($actual2 == 250.0 ? "✅ PASS" : "❌ FAIL") . "\n";
echo "  Ratio: " . ($actual2 / 250.0) . "\n\n";

// Test 3: ZTensor::full([1000], 0.5) = 500
$tensor3 = ZTensor::full([1000], 0.5);
$actual3 = $tensor3->sumtotal();
echo "Test 3: ZTensor::full([1000], 0.5)\n";
echo "  Expected: 500.0\n";
echo "  Actual: $actual3\n";
echo "  Status: " . ($actual3 == 500.0 ? "✅ PASS" : "❌ FAIL") . "\n";
echo "  Ratio: " . ($actual3 / 500.0) . "\n\n";

// Test 4: ZTensor::full([1024], 1.0) = 1024
$tensor4 = ZTensor::full([1024], 1.0);
$actual4 = $tensor4->sumtotal();
echo "Test 4: ZTensor::full([1024], 1.0)\n";
echo "  Expected: 1024.0\n";
echo "  Actual: $actual4\n";
echo "  Status: " . ($actual4 == 1024.0 ? "✅ PASS" : "❌ FAIL") . "\n";
echo "  Ratio: " . ($actual4 / 1024.0) . "\n\n";

// Test 5: ZTensor::full([7], 1.0) = 7
$tensor5 = ZTensor::full([7], 1.0);
$actual5 = $tensor5->sumtotal();
echo "Test 5: ZTensor::full([7], 1.0)\n";
echo "  Expected: 7.0\n";
echo "  Actual: $actual5\n";
echo "  Status: " . ($actual5 == 7.0 ? "✅ PASS" : "❌ FAIL") . "\n\n";

// Test 6: ZTensor::full([16], 1.0) = 16
$tensor6 = ZTensor::full([16], 1.0);
$actual6 = $tensor6->sumtotal();
echo "Test 6: ZTensor::full([16], 1.0)\n";
echo "  Expected: 16.0\n";
echo "  Actual: $actual6\n";
echo "  Status: " . ($actual6 == 16.0 ? "✅ PASS" : "❌ FAIL") . "\n";
