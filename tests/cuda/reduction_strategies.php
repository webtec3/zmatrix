<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

if (!extension_loaded('zmatrix')) exit(77);

function sameScalar(float|int $actual, float|int $expected, string $label, float $atol = 1.0e-5, float $rtol = 1.0e-5): void
{
    if (is_nan((float) $expected)) {
        if (!is_nan((float) $actual)) throw new RuntimeException("{$label}: expected NaN, got {$actual}");
        return;
    }
    if (abs((float) $actual - (float) $expected) > $atol + $rtol * abs((float) $expected)) {
        throw new RuntimeException("{$label}: actual {$actual}, expected {$expected}");
    }
}

function sameTree(array $actual, array $expected, string $label): void
{
    if (count($actual) !== count($expected)) throw new RuntimeException("{$label}: size mismatch");
    foreach ($expected as $index => $value) {
        if (is_array($value)) sameTree($actual[$index], $value, "{$label}[{$index}]");
        else sameScalar($actual[$index], $value, "{$label}[{$index}]", 1.0e-4, 1.0e-4);
    }
}

$globalCases = [
    'finite-ties' => [3.0, 1.0, 1.0, 5.0, 5.0, -2.0],
    'nan-first' => [NAN, 2.0, 1.0, 4.0],
    'nan-later' => [2.0, NAN, 1.0, 4.0],
    'infinities' => [INF, 2.0, -INF, 4.0],
];

foreach (['serial', 'hierarchical', 'cub'] as $strategy) {
    putenv("ZMATRIX_REDUCTION_IMPL={$strategy}");
    foreach ($globalCases as $case => $values) {
        $cpu = ZTensor::arr($values);
        $gpu = ZTensor::arr($values)->toGpu();
        foreach (['min', 'max', 'argmin', 'argmax'] as $method) {
            sameScalar($gpu->{$method}(), $cpu->{$method}(), "{$strategy}/{$case}/{$method}", 0.0, 0.0);
        }
        if ($case === 'finite-ties') sameScalar($gpu->sumtotal(), $cpu->sumtotal(), "{$strategy}/{$case}/sum", 1.0e-5, 1.0e-5);
    }

    $matrix = [[4.0, 1.0, 1.0, 8.0], [2.0, 2.0, -3.0, -3.0], [9.0, 0.0, 7.0, 7.0]];
    foreach ([0, 1] as $axis) {
        foreach (['sum', 'argmin', 'argmax'] as $method) {
            $cpuResult = ZTensor::arr($matrix)->{$method}($axis);
            $gpuResult = ZTensor::arr($matrix)->toGpu()->{$method}($axis);
            if (!$gpuResult->isOnGpu()) throw new RuntimeException("{$strategy}/{$method}/axis{$axis}: lost residency");
            sameTree($gpuResult->toArray(), $cpuResult->toArray(), "{$strategy}/{$method}/axis{$axis}");
        }
    }
    echo "PASS reduction strategy {$strategy}\n";
}

putenv('ZMATRIX_REDUCTION_IMPL');
echo "All reduction strategy checks passed.\n";

