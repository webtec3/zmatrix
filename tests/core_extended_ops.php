<?php

use ZMatrix\ZTensor;

function same(mixed $actual, mixed $expected, string $name): void {
    if (is_array($expected)) {
        if (!is_array($actual) || count($actual) !== count($expected)) throw new RuntimeException("$name shape mismatch");
        foreach ($expected as $index => $value) same($actual[$index], $value, "{$name}[{$index}]");
        return;
    }
    if (is_nan((float) $expected)) {
        if (!is_nan((float) $actual)) throw new RuntimeException("$name expected NaN");
        return;
    }
    if (abs((float) $actual - (float) $expected) > 1.0e-4) throw new RuntimeException("$name mismatch");
}

same(ZTensor::arr([NAN, -0.0, 1.0])->greater(0)->toArray(), [0, 0, 1], 'greater');
same(ZTensor::zeros([2, 3])->broadcast(ZTensor::arr([[1, 2, 3]]))->toArray(), [[1, 2, 3], [1, 2, 3]], 'broadcast');
same(ZTensor::tile(ZTensor::arr([[1, 2], [3, 4]]), 2)->toArray(), [[1, 2], [3, 4], [1, 2], [3, 4]], 'tile');
same(ZTensor::arr([[1, 2], [3, 4]])->cumsum(0)->toArray(), [[1, 2], [4, 6]], 'cumsum');
same(ZTensor::arr([1, 2, 3])->dot([4, 5, 6]), 32.0, 'dot');
same(ZTensor::arr([[1, 2], [3, 4]])->dot([5, 6])->toArray(), [17, 39], 'matvec');

try {
    ZTensor::ones([1])->toGpu();
    throw new RuntimeException('CPU-only build unexpectedly accepted toGpu');
} catch (Throwable $exception) {
    if ($exception->getMessage() === 'CPU-only build unexpectedly accepted toGpu') throw $exception;
}

echo "CPU-only extended operations passed.\n";
