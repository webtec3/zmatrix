<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

if (!extension_loaded('zmatrix')) exit(1);

$size = (int) ($argv[1] ?? 1024);
$iterations = (int) ($argv[2] ?? 10);
$a = ZTensor::arange(0.0, (float) ($size * $size))->reshape([$size, $size])->mul(1.0 / ($size * $size))->toGpu();
$b = ZTensor::ones([$size, $size])->toGpu();
for ($i = 0; $i < 3; ++$i) $a->matmul($b)->add(0.25)->relu()->sumtotal();
for ($i = 0; $i < $iterations; ++$i) {
    $result = $a->matmul($b)->add(0.25)->relu()->mul(0.75)->tanh();
    $checksum = $result->sumtotal();
    if (!is_finite($checksum)) throw new RuntimeException('invalid profile checksum');
}
echo "profile target validated\n";

