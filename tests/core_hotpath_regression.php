<?php
use ZMatrix\ZTensor;
if (!extension_loaded('zmatrix')) {
    throw new RuntimeException('zmatrix extension is not loaded');
}

function assertTree($actual, $expected, string $label): void {
    if (is_array($expected)) {
        if (!is_array($actual) || count($actual) !== count($expected)) {
            throw new RuntimeException($label . ' shape mismatch');
        }
        foreach ($expected as $index => $value) {
            assertTree($actual[$index], $value, $label . '[' . $index . ']');
        }
        return;
    }

    if (is_numeric($actual) && is_numeric($expected)) {
        if (abs((float) $actual - (float) $expected) <= 1.0e-6) {
            return;
        }
    } elseif ($actual === $expected) {
        return;
    }

    throw new RuntimeException(
        $label . ' mismatch: ' . json_encode($actual) . ' !== ' . json_encode($expected)
    );
}

$ones = ZTensor::ones([2, 3]);
assertTree($ones->toArray(), [[1, 1, 1], [1, 1, 1]], 'ones');
$full = ZTensor::full([2, 2], -3.5);
assertTree($full->toArray(), [[-3.5, -3.5], [-3.5, -3.5]], 'full');

$input = ZTensor::arr([[-5.0, -1.0, 0.0, 2.0, 7.0]]);
$clipped = ZTensor::clip($input, -1.0, 3.0);
assertTree($input->toArray(), [[-5, -1, 0, 2, 7]], 'clip input remains independent');
assertTree($clipped->toArray(), [[-1, -1, 0, 2, 3]], 'clip result');

$special = ZTensor::clip(ZTensor::arr([NAN, -INF, INF, -2.0, 2.0]), -1.0, 1.0)->toArray();
if (!is_nan($special[0]) || $special[1] !== -1.0 || $special[2] !== 1.0 || $special[3] !== -1.0 || $special[4] !== 1.0) {
    throw new RuntimeException('clip special values mismatch: ' . json_encode($special));
}

$arg = ZTensor::arr([
    [[1.0, 5.0, 5.0], [9.0, 9.0, 1.0]],
    [[-2.0, -2.0, -3.0], [4.0, 1.0, 4.0]],
]);
assertTree($arg->argmax(2)->toArray(), [[1, 0], [0, 0]], 'argmax axis 2 first tie');
assertTree($arg->argmin(2)->toArray(), [[0, 2], [2, 1]], 'argmin axis 2 first tie');
assertTree($arg->argmax(1)->toArray(), [[1, 1, 0], [1, 1, 1]], 'argmax axis 1');
assertTree($arg->argmin(1)->toArray(), [[0, 0, 1], [0, 0, 0]], 'argmin axis 1');

$matmul = ZTensor::arr([[1, 2, 3], [4, 5, 6]])->matmul([[7, 8], [9, 10], [11, 12]]);
assertTree($matmul->toArray(), [[58, 64], [139, 154]], 'matmul cpu');
if ($matmul->isOnGpu()) {
    throw new RuntimeException('CPU matmul unexpectedly produced a GPU-resident result');
}

echo "PASS core hotpath regression\n";
