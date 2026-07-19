<?php
use ZMatrix\ZTensor;

if (!extension_loaded('zmatrix')) {
    throw new RuntimeException('zmatrix extension is not loaded');
}

function sameTree($actual, $expected, string $label): void {
    if (is_array($expected)) {
        if (!is_array($actual) || count($actual) !== count($expected)) {
            throw new RuntimeException($label . ' shape mismatch');
        }
        foreach ($expected as $index => $value) {
            sameTree($actual[$index], $value, $label . '[' . $index . ']');
        }
        return;
    }
    if (!is_numeric($actual) || abs((float) $actual - (float) $expected) > 1.0e-6) {
        throw new RuntimeException($label . ' mismatch');
    }
}

$source = ZTensor::arr([[1, 2, 3], [4, 5, 6]]);
$view = $source->transpose();
sameTree($view->toArray(), [[1, 4], [2, 5], [3, 6]], 'transpose view');
sameTree($source->toArray(), [[1, 2, 3], [4, 5, 6]], 'source');

$copy = $view->copy();
$copy->add(ZTensor::ones([3, 2]));
sameTree($copy->toArray(), [[2, 5], [3, 6], [4, 7]], 'materialized add');
sameTree($view->toArray(), [[1, 4], [2, 5], [3, 6]], 'view after copy mutation');

$reshaped = $view->reshape([2, 3]);
sameTree($reshaped->toArray(), [[1, 4, 2], [5, 3, 6]], 'reshape non-contiguous view');

$product = $view->matmul([[1], [1]]);
sameTree($product->toArray(), [[5], [7], [9]], 'matmul view');

$roundTrip = $view->transpose();
sameTree($roundTrip->toArray(), [[1, 2, 3], [4, 5, 6]], 'double transpose');

if ($view->isOnGpu() || $product->isOnGpu()) {
    throw new RuntimeException('CPU view unexpectedly selected GPU');
}

$empty = ZTensor::zeros([0, 3])->transpose();
sameTree($empty->toArray(), [[], [], []], 'empty transpose');

echo "PASS strided transpose views\n";
