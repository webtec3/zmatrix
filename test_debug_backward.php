<?php
/**
 * Debug test - trace what's happening with backward()
 */

namespace ZMatrix\Tests;

use ZMatrix\ZTensor;

echo "Creating tensors...\n";
$a = ZTensor::arr([2.0])->requiresGrad(true);
$b = ZTensor::arr([3.0])->requiresGrad(true);

echo "a requires_grad: " . ($a->isRequiresGrad() ? "true" : "false") . "\n";
echo "b requires_grad: " . ($b->isRequiresGrad() ? "true" : "false") . "\n";

echo "\nPerforming addition...\n";
$y = ZTensor::addAutograd($a, $b);
echo "y = " . $y->toArray()[0] . "\n";
echo "y requires_grad: " . ($y->isRequiresGrad() ? "true" : "false") . "\n";

echo "\nBefore backward:\n";
echo "a->getGrad(): " . ($a->getGrad() === null ? "NULL" : json_encode($a->getGrad()->toArray())) . "\n";
echo "b->getGrad(): " . ($b->getGrad() === null ? "NULL" : json_encode($b->getGrad()->toArray())) . "\n";
echo "y->getGrad(): " . ($y->getGrad() === null ? "NULL" : json_encode($y->getGrad()->toArray())) . "\n";

echo "\nCalling backward()...\n";
$y->backward();

echo "\nAfter backward:\n";
echo "a->getGrad(): " . ($a->getGrad() === null ? "NULL" : json_encode($a->getGrad()->toArray())) . "\n";
echo "b->getGrad(): " . ($b->getGrad() === null ? "NULL" : json_encode($b->getGrad()->toArray())) . "\n";
echo "y->getGrad(): " . ($y->getGrad() === null ? "NULL" : json_encode($y->getGrad()->toArray())) . "\n";
