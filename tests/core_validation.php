<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

if (!extension_loaded('zmatrix')) exit(77);

function mustThrow(callable $operation, string $label): void
{
    try {
        $operation();
    } catch (Throwable) {
        echo "PASS {$label}\n";
        return;
    }
    throw new RuntimeException("{$label}: expected exception");
}

mustThrow(static fn(): ZTensor => ZTensor::zeros([PHP_INT_MAX, 2]), 'shape multiplication overflow');
mustThrow(static fn(): ZTensor => ZTensor::ones([-1, 2]), 'negative shape rejected');
mustThrow(static fn(): ZTensor => ZTensor::arr([1.0, 2.0])->reshape([PHP_INT_MAX, 2]), 'reshape overflow');
mustThrow(static fn(): ZTensor => ZTensor::clip(ZTensor::arr([1.0]), 2.0, -2.0), 'invalid clip bounds');
mustThrow(static fn(): ZTensor => ZTensor::clip(ZTensor::arr([1.0]), NAN, 2.0), 'NaN clip bound rejected');

$cpu = ZTensor::arr([1.0, 4.0, 9.0])->sqrt();
if ($cpu->isOnGpu()) throw new RuntimeException('CPU default unexpectedly activated GPU');
if ($cpu->toArray() !== [1.0, 2.0, 3.0]) throw new RuntimeException('CPU sqrt regression');
echo "PASS CPU remains default\n";

for ($i = 0; $i < 1000; ++$i) {
    $tensor = ZTensor::ones([16, 16])->mul((float) $i)->copy();
}
unset($tensor);
gc_collect_cycles();
echo "PASS lifecycle stress\n";
echo "All core validation checks passed.\n";
