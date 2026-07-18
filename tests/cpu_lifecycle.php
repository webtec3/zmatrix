<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

for ($i = 0; $i < 500; ++$i) {
    $a = ZTensor::arr([[1, 2, 3], [4, 5, 6]]);
    $b = $a->transpose()->matmul([[1], [1]]);
    if ($b->toArray() !== [[5.0], [7.0], [9.0]]) {
        throw new RuntimeException('CPU lifecycle result mismatch');
    }
}

echo "CPU lifecycle stress passed.\n";
