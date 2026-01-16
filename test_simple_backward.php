<?php
/**
 * Debug test - trace backward execution with more detail
 */

namespace ZMatrix\Tests;

use ZMatrix\ZTensor;

echo "TEST: Simple backward propagation\n";
echo "==================================\n\n";

$x = ZTensor::arr([2.0])->requiresGrad(true);
$y = ZTensor::arr([3.0])->requiresGrad(true);

echo "Initial state:\n";
echo "  x requires_grad: " . ($x->isRequiresGrad() ? "YES" : "NO") . "\n";
echo "  y requires_grad: " . ($y->isRequiresGrad() ? "YES" : "NO") . "\n";
echo "  x grad: " . ($x->getGrad() === null ? "NULL" : json_encode($x->getGrad()->toArray())) . "\n";
echo "  y grad: " . ($y->getGrad() === null ? "NULL" : json_encode($y->getGrad()->toArray())) . "\n\n";

echo "Creating: z = x + y\n";
$z = ZTensor::addAutograd($x, $y);
echo "  z value: " . $z->toArray()[0] . "\n";
echo "  z requires_grad: " . ($z->isRequiresGrad() ? "YES" : "NO") . "\n\n";

echo "Calling z->backward()...\n";
$z->backward();

echo "\nAfter backward():\n";
echo "  x grad: " . ($x->getGrad() === null ? "NULL" : json_encode($x->getGrad()->toArray())) . "\n";
echo "  y grad: " . ($y->getGrad() === null ? "NULL" : json_encode($y->getGrad()->toArray())) . "\n";
echo "  z grad: " . ($z->getGrad() === null ? "NULL" : json_encode($z->getGrad()->toArray())) . "\n\n";

if ($x->getGrad() !== null && $y->getGrad() !== null) {
    echo "✅ SUCCESS: Gradients propagated!\n";
} else {
    echo "❌ FAILED: Gradients not propagated\n";
}
