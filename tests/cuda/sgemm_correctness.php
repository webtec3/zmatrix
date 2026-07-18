<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

if (!extension_loaded('zmatrix')) {
    fwrite(STDERR, "SKIP: zmatrix extension is not loaded\n");
    exit(77);
}

/** @return list<list<float>> */
function matrix(int $rows, int $cols, callable $value): array
{
    $result = [];
    for ($i = 0; $i < $rows; ++$i) {
        $row = [];
        for ($j = 0; $j < $cols; ++$j) {
            $row[] = (float) $value($i, $j);
        }
        $result[] = $row;
    }
    return $result;
}

function assertClose(mixed $actual, mixed $expected, string $path, float $atol = 1.0e-5, float $rtol = 1.0e-5): void
{
    if (is_array($expected)) {
        if (!is_array($actual) || count($actual) !== count($expected)) {
            throw new RuntimeException("{$path}: array shape mismatch");
        }
        foreach ($expected as $index => $value) {
            assertClose($actual[$index], $value, "{$path}[{$index}]", $atol, $rtol);
        }
        return;
    }

    $difference = abs((float) $actual - (float) $expected);
    $limit = $atol + $rtol * abs((float) $expected);
    if (is_nan($difference) || $difference > $limit) {
        throw new RuntimeException(sprintf(
            '%s: actual %.9g, expected %.9g, difference %.9g > %.9g',
            $path,
            $actual,
            $expected,
            $difference,
            $limit
        ));
    }
}

function compareCpuAndGpu(string $name, array $a, array $b, float $atol = 1.0e-5, float $rtol = 1.0e-5): void
{
    $cpuA = ZTensor::arr($a);
    $cpuB = ZTensor::arr($b);
    $cpu = $cpuA->matmul($cpuB);
    if ($cpu->isOnGpu()) {
        throw new RuntimeException("{$name}: CPU inputs triggered implicit GPU dispatch");
    }

    $gpuA = ZTensor::arr($a)->toGpu();
    $gpuB = ZTensor::arr($b)->toGpu();
    if (!$gpuA->isOnGpu() || !$gpuB->isOnGpu()) {
        throw new RuntimeException("{$name}: toGpu() did not establish device residency");
    }

    $gpu = $gpuA->matmul($gpuB);
    if (!$gpu->isOnGpu()) {
        throw new RuntimeException("{$name}: SGEMM result did not remain device-resident");
    }

    $expectedShape = $cpu->shape();
    if ($gpu->shape() !== $expectedShape) {
        throw new RuntimeException("{$name}: result shape differs between CPU and GPU");
    }

    assertClose($gpu->toArray(), $cpu->toArray(), $name, $atol, $rtol);
    if (!$gpu->isOnGpu()) {
        throw new RuntimeException("{$name}: toArray() unexpectedly discarded device residency");
    }

    $gpu->toCpu();
    if ($gpu->isOnGpu()) {
        throw new RuntimeException("{$name}: toCpu() did not clear device residency");
    }
    assertClose($gpu->toArray(), $cpu->toArray(), "{$name}/after-toCpu", $atol, $rtol);
}

$cases = [
    'canonical-2x3-3x2' => [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8], [9, 10], [11, 12]],
    ],
    'square' => [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ],
    'rectangular' => [
        [[1, -2], [3, 4], [-5, 6]],
        [[7, 8, 9, 10], [-1, 0.5, 2, -3]],
    ],
    'one-by-n' => [
        [[1, 2, 3, 4]],
        [[2], [3], [4], [5]],
    ],
    'n-by-one' => [
        [[1], [-2], [3], [0.5]],
        [[2, -4, 8]],
    ],
    'identity' => [
        [[1.25, -2.5, 4], [0, 3, -7]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
    'zeros' => [
        [[1, 2, 3], [4, 5, 6]],
        [[0, 0], [0, 0], [0, 0]],
    ],
    'fractional-negative' => [
        [[0.25, -1.5], [2.75, -0.125]],
        [[-4.5, 0.5], [3.25, -2]],
    ],
    'large-and-small' => [
        [[1.0e10, 1.0e-10], [-1.0e-10, 1.0e10]],
        [[1.0e-10, -1.0e-10], [1.0e-10, 1.0e-10]],
    ],
];

mt_srand(20260718);
$cases['seeded-random-7x5-5x9'] = [
    matrix(7, 5, static fn(): float => (mt_rand(-10000, 10000) / 1000.0)),
    matrix(5, 9, static fn(): float => (mt_rand(-10000, 10000) / 1000.0)),
];

foreach ($cases as $name => [$a, $b]) {
    compareCpuAndGpu($name, $a, $b, 2.0e-5, 3.0e-5);
    echo "PASS {$name}\n";
}

try {
    ZTensor::arr([[1, 2]])->toGpu()->matmul(ZTensor::arr([[1, 2]])->toGpu());
    throw new RuntimeException('incompatible-shapes: expected an exception');
} catch (Throwable $error) {
    if ($error instanceof RuntimeException && $error->getMessage() === 'incompatible-shapes: expected an exception') {
        throw $error;
    }
    echo "PASS incompatible-shapes\n";
}

echo "All CUDA SGEMM correctness checks passed.\n";
