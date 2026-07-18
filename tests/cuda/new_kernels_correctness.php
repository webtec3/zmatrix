<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

if (!extension_loaded('zmatrix')) exit(77);

function sameNumber(float $actual, float $expected, string $label, float $atol = 2.0e-5, float $rtol = 2.0e-5): void
{
    if (is_nan($expected)) {
        if (!is_nan($actual)) throw new RuntimeException("{$label}: expected NaN, got {$actual}");
        return;
    }
    if (is_infinite($expected)) {
        if ($actual !== $expected) throw new RuntimeException("{$label}: expected {$expected}, got {$actual}");
        return;
    }
    $error = abs($actual - $expected);
    if ($error > $atol + $rtol * abs($expected)) {
        throw new RuntimeException("{$label}: actual {$actual}, expected {$expected}, error {$error}");
    }
}

function sameTree(mixed $actual, mixed $expected, string $label, float $atol = 2.0e-5, float $rtol = 2.0e-5): void
{
    if (is_array($expected)) {
        if (!is_array($actual) || count($actual) !== count($expected)) throw new RuntimeException("{$label}: shape mismatch");
        foreach ($expected as $index => $value) sameTree($actual[$index], $value, "{$label}[{$index}]", $atol, $rtol);
        return;
    }
    sameNumber((float) $actual, (float) $expected, $label, $atol, $rtol);
}

function compareUnary(string $name, array $input, callable $operation, float $atol = 2.0e-5, float $rtol = 2.0e-5): void
{
    $cpu = ZTensor::arr($input);
    $gpu = ZTensor::arr($input)->toGpu();
    $operation($cpu);
    $operation($gpu);
    if (!$gpu->isOnGpu()) throw new RuntimeException("{$name}: GPU result lost residency");
    if ($cpu->isOnGpu()) throw new RuntimeException("{$name}: CPU path activated GPU automatically");
    sameTree($gpu->toArray(), $cpu->toArray(), $name, $atol, $rtol);
    echo "PASS {$name}\n";
}

compareUnary('sqrt special values', [0.0, 1.0, 4.0, 0.25, NAN, INF], static fn(ZTensor $t): ZTensor => $t->sqrt());
$largeCpu = ZTensor::linspace(0.0, 100.0, 262144);
$largeGpu = ZTensor::arr($largeCpu)->toGpu();
$largeCpu->sqrt();
$largeGpu->sqrt();
if (!$largeGpu->isOnGpu()) throw new RuntimeException('sqrt large lost residency');
sameTree($largeGpu->toArray(), $largeCpu->toArray(), 'sqrt large', 3.0e-5, 3.0e-5);
echo "PASS sqrt large\n";

foreach ([false, true] as $gpuPath) {
    $negative = ZTensor::arr([4.0, -1.0, 9.0]);
    if ($gpuPath) $negative->toGpu();
    try {
        $negative->sqrt();
        throw new RuntimeException('sqrt negative did not throw');
    } catch (Throwable $error) {
        if ($error->getMessage() === 'sqrt negative did not throw') throw $error;
    }
    sameTree($negative->toArray(), [4.0, -1.0, 9.0], 'sqrt failure atomicity');
}
echo "PASS sqrt negative atomic failure\n";

$empty = ZTensor::zeros([0])->toGpu()->sqrt();
if (!$empty->isOnGpu() || $empty->toArray() !== []) throw new RuntimeException('sqrt empty semantics failed');
echo "PASS sqrt empty\n";

$clipInput = [-INF, -4.0, -1.0, 0.0, 1.0, 4.0, INF, NAN];
$clipCpu = ZTensor::clip(ZTensor::arr($clipInput), -1.5, 2.5);
$clipGpu = ZTensor::clip(ZTensor::arr($clipInput)->toGpu(), -1.5, 2.5);
if (!$clipGpu->isOnGpu()) throw new RuntimeException('clip result lost residency');
sameTree($clipGpu->toArray(), $clipCpu->toArray(), 'clip special values');
echo "PASS clip special values\n";

foreach ([ZTensor::arr([1.0]), ZTensor::arr([1.0])->toGpu()] as $clipInvalid) {
    try {
        ZTensor::clip($clipInvalid, 2.0, -2.0);
        throw new RuntimeException('clip invalid bounds did not throw');
    } catch (Throwable $error) {
        if ($error->getMessage() === 'clip invalid bounds did not throw') throw $error;
    }
}
echo "PASS clip invalid bounds\n";

$emptyClip = ZTensor::clip(ZTensor::zeros([0])->toGpu(), -1.0, 1.0);
if (!$emptyClip->isOnGpu() || $emptyClip->toArray() !== []) throw new RuntimeException('clip empty semantics failed');
echo "PASS clip empty\n";

$softmaxCases = [
    'softmax vector' => [1.0, 2.0, 3.0, -4.0],
    'softmax large values' => [10000.0, 10001.0, 9999.0],
    'softmax equal values' => [7.0, 7.0, 7.0, 7.0],
    'softmax singleton' => [42.0],
    'softmax matrix' => [[1.0, 2.0, 3.0], [-1000.0, -1001.0, -999.0]],
    'softmax matrix NaN/Inf' => [[NAN, 1.0, 2.0], [INF, 1.0, -INF]],
];
foreach ($softmaxCases as $name => $input) compareUnary($name, $input, static fn(ZTensor $t): ZTensor => $t->softmax(), 5.0e-5, 5.0e-5);

$emptySoftmax = ZTensor::zeros([0])->toGpu()->softmax();
if (!$emptySoftmax->isOnGpu() || $emptySoftmax->toArray() !== []) throw new RuntimeException('softmax empty semantics failed');
echo "PASS softmax empty\n";

$largeSoftmaxCpu = ZTensor::linspace(-8.0, 8.0, 262144)->reshape([1024, 256]);
$largeSoftmaxGpu = ZTensor::arr($largeSoftmaxCpu)->toGpu();
$largeSoftmaxCpu->softmax();
$largeSoftmaxGpu->softmax();
if (!$largeSoftmaxGpu->isOnGpu()) throw new RuntimeException('softmax large matrix lost residency');
sameTree($largeSoftmaxGpu->toArray(), $largeSoftmaxCpu->toArray(), 'softmax large matrix', 5.0e-5, 5.0e-5);
echo "PASS softmax large matrix synchronization\n";

$probabilities = [[0.1, 0.2, 0.7], [0.0, 1.0, NAN]];
compareUnary('softmax derivative', $probabilities, static fn(ZTensor $t): ZTensor => $t->softmaxDerivative());

$chained = ZTensor::arr([[1.0, 4.0], [9.0, 16.0]])->toGpu()->sqrt()->softmax()->softmaxDerivative();
if (!$chained->isOnGpu()) throw new RuntimeException('new kernel chain lost residency');
$beforeFree = $chained->toArray();
$again = ZTensor::arr([[1.0, 4.0], [9.0, 16.0]])->toGpu()->sqrt()->softmax()->softmaxDerivative();
$again->freeDevice();
if ($again->isOnGpu()) throw new RuntimeException('freeDevice left device residency active');
sameTree($again->toArray(), $beforeFree, 'freeDevice preserves device-only value');
echo "PASS chained residency and freeDevice coherence\n";

echo "All new CUDA kernel checks passed.\n";
