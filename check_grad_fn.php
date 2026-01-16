<?php
// Check if grad_fn is preserved

use ZMatrix\ZTensor;

$x = ZTensor::arr([1.0])->requiresGrad(true);
$y = ZTensor::arr([2.0])->requiresGrad(true);

$z = ZTensor::addAutograd($x, $y);

// Try to check if grad_fn is set
echo "z = " . $z->toArray()[0] . "\n";
echo "z requires_grad = " . ($z->isRequiresGrad() ? "YES" : "NO") . "\n";

// Call backward
echo "Calling backward...\n";
$z->backward();

echo "After backward:\n";
echo "x->getGrad() = " . ($x->getGrad() === null ? "null" : json_encode($x->getGrad()->toArray())) . "\n";
echo "y->getGrad() = " . ($y->getGrad() === null ? "null" : json_encode($y->getGrad()->toArray())) . "\n";
