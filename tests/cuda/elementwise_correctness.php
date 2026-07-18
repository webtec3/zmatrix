<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

if (!extension_loaded('zmatrix')) {
    fwrite(STDERR, "SKIP: zmatrix extension is not loaded\n");
    exit(77);
}

function sameFloat(float $actual, float $expected, float $atol = 1.0e-6, float $rtol = 1.0e-5): bool
{
    if (is_nan($expected)) return is_nan($actual);
    if (is_infinite($expected)) return $actual === $expected;
    return abs($actual - $expected) <= $atol + $rtol * abs($expected);
}

function assertTree(mixed $actual, mixed $expected, string $path): void
{
    if (is_array($expected)) {
        if (!is_array($actual) || count($actual) !== count($expected)) {
            throw new RuntimeException("{$path}: shape mismatch");
        }
        foreach ($expected as $key => $value) assertTree($actual[$key], $value, "{$path}[{$key}]");
        return;
    }
    if (!sameFloat((float) $actual, (float) $expected)) {
        throw new RuntimeException("{$path}: actual {$actual}, expected {$expected}");
    }
}

function compare(string $name, array $input, callable $operation): void
{
    $cpu = ZTensor::arr($input);
    $gpu = ZTensor::arr($input)->toGpu();
    $operation($cpu);
    $operation($gpu);
    if (!$gpu->isOnGpu()) throw new RuntimeException("{$name}: lost device residency");
    assertTree($gpu->toArray(), $cpu->toArray(), $name);
    echo "PASS {$name}\n";
}

$values = [[-INF, -3.5, -0.0, 0.0], [0.25, 2.0, INF, NAN]];
$finite = [[-3.5, -1.0, 0.0, 0.25], [1.0, 2.0, 5.5, 10.0]];
$positive = [[0.125, 0.5, 1.0, 2.0], [3.5, 10.0, 100.0, 1.0e-5]];
$other = [[2.0, -4.0, 0.5, 8.0], [-1.0, 3.0, 2.0, -0.25]];

compare('add tensor', $finite, static fn(ZTensor $t) => $t->add($other));
compare('sub tensor', $finite, static fn(ZTensor $t) => $t->sub($other));
compare('mul tensor', $finite, static fn(ZTensor $t) => $t->mul($other));
compare('scalar add', $finite, static fn(ZTensor $t) => $t->add(1.25));
compare('scalar sub', $finite, static fn(ZTensor $t) => $t->sub(1.25));
compare('scalar mul', $finite, static fn(ZTensor $t) => $t->mul(-2.5));
compare('scalar div', $finite, static fn(ZTensor $t) => $t->divide(2.0));
compare('relu special', $values, static fn(ZTensor $t) => $t->relu());
compare('leakyRelu', $finite, static fn(ZTensor $t) => $t->leakyRelu(0.125));
compare('sigmoid', $finite, static fn(ZTensor $t) => $t->sigmoid());
compare('tanh', $finite, static fn(ZTensor $t) => $t->tanh());
compare('exp', $finite, static fn(ZTensor $t) => $t->exp());
compare('abs special', $values, static fn(ZTensor $t) => $t->abs());
compare('divide tensor', $positive, static fn(ZTensor $t) => $t->divide($other));
compare('pow', $positive, static fn(ZTensor $t) => $t->pow(1.75));
compare('log', $positive, static fn(ZTensor $t) => $t->log());
compare('fill', $finite, static fn(ZTensor $t) => $t->fill(-3.25));
foreach ([
    'zeros explicit device generation' => [ZTensor::zeros([3, 5]), array_fill(0, 3, array_fill(0, 5, 0.0))],
    'ones explicit device generation' => [ZTensor::ones([3, 5]), array_fill(0, 3, array_fill(0, 5, 1.0))],
] as $name => [$uniform, $expectedUniform]) {
    $uniform->toGpu();
    if (!$uniform->isOnGpu()) throw new RuntimeException("{$name}: no device residency");
    assertTree($uniform->toArray(), $expectedUniform, $name);
    echo "PASS {$name}\n";
}

foreach ([
    'transpose square' => [[1, 2], [3, 4]],
    'transpose rectangular' => [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
    'transpose 1xN' => [[1, 2, 3, 4, 5, 6, 7]],
    'transpose non-tile' => matrixForTranspose(35, 33),
] as $name => $input) {
    $cpuTranspose = ZTensor::arr($input)->transpose();
    $gpuTranspose = ZTensor::arr($input)->toGpu()->transpose();
    if (!$gpuTranspose->isOnGpu()) throw new RuntimeException("{$name}: result lost device residency");
    assertTree($gpuTranspose->toArray(), $cpuTranspose->toArray(), $name);
    echo "PASS {$name}\n";
}

$reductionInput = [[3.5, -2.0, 7.0], [1.0, 9.0, -4.0], [8.0, 0.5, 2.0]];
$cpuReduction = ZTensor::arr($reductionInput);
$gpuReduction = ZTensor::arr($reductionInput)->toGpu();
foreach (['sumtotal', 'mean', 'min', 'max', 'argmin', 'argmax'] as $method) {
    $actual = $gpuReduction->{$method}();
    $expected = $cpuReduction->{$method}();
    if (is_float($expected)) {
        if (!sameFloat((float) $actual, $expected, 1.0e-5, 1.0e-5)) {
            throw new RuntimeException("reduction {$method}: actual {$actual}, expected {$expected}");
        }
    } elseif ($actual !== $expected) {
        throw new RuntimeException("reduction {$method}: actual {$actual}, expected {$expected}");
    }
    if (!$gpuReduction->isOnGpu()) throw new RuntimeException("reduction {$method}: input lost residency");
    echo "PASS reduction {$method}\n";
}
foreach ([0, 1, -1] as $axis) {
    $actual = $gpuReduction->sum($axis);
    $expected = $cpuReduction->sum($axis);
    if (!$actual->isOnGpu()) throw new RuntimeException("sum axis {$axis}: result lost residency");
    assertTree($actual->toArray(), $expected->toArray(), "sum axis {$axis}");
    echo "PASS sum axis {$axis}\n";

    foreach (['argmin', 'argmax'] as $method) {
        $actualArg = $gpuReduction->{$method}($axis);
        $expectedArg = $cpuReduction->{$method}($axis);
        if (!$actualArg->isOnGpu()) throw new RuntimeException("{$method} axis {$axis}: result lost residency");
        assertTree($actualArg->toArray(), $expectedArg->toArray(), "{$method} axis {$axis}");
        echo "PASS {$method} axis {$axis}\n";
    }
}

$cpu = ZTensor::arr($finite);
$gpu = ZTensor::arr($finite)->toGpu();
$cpu->add($other)->relu()->mul(0.5)->sub(0.25)->tanh();
$gpu->add(ZTensor::arr($other)->toGpu())->relu()->mul(0.5)->sub(0.25)->tanh();
if (!$gpu->isOnGpu()) throw new RuntimeException('chain: lost device residency');
assertTree($gpu->toArray(), $cpu->toArray(), 'chain');
echo "PASS chained GPU operations\n";

$copy = ZTensor::arr($positive)->toGpu();
$copy->log(); // currently exercises a GPU-to-CPU coherence transition
assertTree($copy->toArray(), ZTensor::arr($positive)->log()->toArray(), 'device-to-CPU-operation');
echo "PASS device write/read coherence\n";

echo "All existing CUDA elementwise checks passed.\n";

function matrixForTranspose(int $rows, int $cols): array
{
    $matrix = [];
    for ($i = 0; $i < $rows; ++$i) {
        $row = [];
        for ($j = 0; $j < $cols; ++$j) $row[] = (float) ($i * $cols + $j - 200);
        $matrix[] = $row;
    }
    return $matrix;
}
