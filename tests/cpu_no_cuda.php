<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

$tensor = ZTensor::arr([1, 2, 3]);
try {
    $tensor->toGpu();
    throw new RuntimeException('CPU-only build accepted toGpu()');
} catch (Throwable $error) {
    if ($error->getMessage() === 'CPU-only build accepted toGpu()') throw $error;
    if ($tensor->isOnGpu()) throw new RuntimeException('CPU-only failure marked tensor as device-resident');
}

$tensor->add(1.0);
if ($tensor->toArray() !== [2.0, 3.0, 4.0]) throw new RuntimeException('CPU path failed after rejected toGpu()');
echo "CPU-only CUDA rejection passed.\n";
