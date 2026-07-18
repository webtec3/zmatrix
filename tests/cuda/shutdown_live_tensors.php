<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

$live = [];
for ($i = 0; $i < 128; ++$i) {
    $live[] = ZTensor::ones([4096 + $i])->toGpu()->cumsum()->greater(0);
}

echo "shutdown with 128 live device tensors\n";
