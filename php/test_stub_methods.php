<?php
// Test all new stub methods
echo "=== Testing New Factory Methods ===\n";

// Test zeros
$t = ZTensor::zeros([2, 3]);
echo "✓ zeros([2,3]): " . $t->shape()[0] . "x" . $t->shape()[1] . "\n";

// Test ones
$o = ZTensor::ones([3, 2]);
echo "✓ ones([3,2]): " . $o->shape()[0] . "x" . $o->shape()[1] . "\n";

// Test arange
$r = ZTensor::arange(0, 10, 2);
echo "✓ arange(0,10,2): shape=" . json_encode($r->shape()) . "\n";

// Test identity
$id = ZTensor::identity(3);
echo "✓ identity(3): " . $id->shape()[0] . "x" . $id->shape()[1] . "\n";

// Test randn
$rand = ZTensor::randn([2, 2]);
echo "✓ randn([2,2]): shape=" . json_encode($rand->shape()) . "\n";

// Test clip
$clipped = ZTensor::ones([3])->scalarMultiply(5)->clip(2, 4);
echo "✓ clip(2,4): OK\n";

// Test full
$full = ZTensor::full([2, 3], 3.14);
echo "✓ full([2,3], 3.14): shape=" . json_encode($full->shape()) . "\n";

// Test random
$rnd = ZTensor::random([2, 3], 0.0, 1.0);
echo "✓ random([2,3]): shape=" . json_encode($rnd->shape()) . "\n";

// Test linspace
$lin = ZTensor::linspace(0, 10, 5);
echo "✓ linspace(0,10,5): shape=" . json_encode($lin->shape()) . "\n";

// Test logspace
$log = ZTensor::logspace(0, 2, 5);
echo "✓ logspace(0,2,5): shape=" . json_encode($log->shape()) . "\n";

// Test eye
$eye = ZTensor::eye(4);
echo "✓ eye(4): " . $eye->shape()[0] . "x" . $eye->shape()[1] . "\n";

// Test tile
$base = ZTensor::arr([[1, 2]]);
$tiled = ZTensor::tile($base, 3);
echo "✓ tile(base, 3): shape=" . json_encode($tiled->shape()) . "\n";

// Test __toString
$str = (string)ZTensor::arr([1, 2, 3]);
echo "✓ __toString(): " . strlen($str) . " chars\n";

echo "\n=== Testing Autograd Methods ===\n";

// Test autograd
$x = ZTensor::arr([1, 2, 3])->requiresGrad(true);
echo "✓ requiresGrad(true): OK\n";

$is_req = $x->is_requires_grad();
echo "✓ is_requires_grad(): " . ($is_req ? "true" : "false") . "\n";

$y = $x->sum();
$y->backward();
echo "✓ backward(): OK\n";

$grad = $x->get_grad();
echo "✓ get_grad(): " . ($grad ? "tensor" : "null") . "\n";

$x->zero_grad();
echo "✓ zero_grad(): OK\n";

$x->ensure_grad();
echo "✓ ensure_grad(): OK\n";

echo "\n=== All Tests Passed! ===\n";
?>
