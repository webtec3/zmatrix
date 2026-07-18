<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

if (!extension_loaded('zmatrix')) exit(77);

function assertSameTree(mixed $actual, mixed $expected, string $name, float $tol = 1.0e-5): void
{
    if (is_array($expected)) {
        if (!is_array($actual) || count($actual) !== count($expected)) throw new RuntimeException("{$name}: shape mismatch");
        foreach ($expected as $key => $value) assertSameTree($actual[$key], $value, "{$name}[{$key}]", $tol);
        return;
    }
    if (abs((float) $actual - (float) $expected) > $tol) {
        throw new RuntimeException("{$name}: actual {$actual}, expected {$expected}");
    }
}

// CPU write -> H2D -> CUDA write -> D2H.
$tensor = ZTensor::arr([[1, 4], [9, 16]]);
$tensor->sqrt()->toGpu()->mul(2.0);
if (!$tensor->isOnGpu()) throw new RuntimeException('basic transition lost device residency');
assertSameTree($tensor->toArray(), [[2, 4], [6, 8]], 'basic transition');
echo "PASS CPU/H2D/CUDA/D2H transition\n";

// CUDA write -> CPU-only write -> later CUDA operation. sigmoidDerivative
// intentionally remains CPU-only; softmax now has a CUDA implementation.
$tensor->add(1.0)->sigmoidDerivative();
if ($tensor->isOnGpu()) throw new RuntimeException('CPU-only sigmoidDerivative left stale device state valid');
$cpuAfterCpuOperation = $tensor->toArray();
$tensor->toGpu()->mul(3.0);
$expected = ZTensor::arr($cpuAfterCpuOperation)->mul(3.0)->toArray();
assertSameTree($tensor->toArray(), $expected, 'host modification re-upload');
echo "PASS host modification invalidates and re-uploads\n";

// Device-resident deep copy must copy the current device version, not stale host bytes.
$source = ZTensor::arr([[1, 2], [3, 4]])->toGpu()->add(10.0);
$copy = ZTensor::arr($source);
if (!$copy->isOnGpu()) throw new RuntimeException('device copy lost residency');
$source->add(100.0);
assertSameTree($copy->toArray(), [[11, 12], [13, 14]], 'device deep copy');
echo "PASS device-resident deep copy\n";

// Operations called out by the audit must synchronize host reads correctly.
$device = ZTensor::arr([[1, 5], [3, -2]])->toGpu()->add(2.0);
assertSameTree(ZTensor::clip($device, 0.0, 5.0)->toArray(), [[3, 5], [5, 0]], 'clip');
assertSameTree(ZTensor::minimum($device, 4.0)->toArray(), [[3, 4], [4, 0]], 'minimum');
assertSameTree(ZTensor::maximum($device, 4.0)->toArray(), [[4, 7], [5, 4]], 'maximum');
assertSameTree(ZTensor::tile($device, 2)->toArray(), [[3, 7], [5, 0], [3, 7], [5, 0]], 'tile');
assertSameTree(ZTensor::arr([[0, 0], [0, 0]])->toGpu()->broadcast(ZTensor::arr([10, 20])->toGpu())->toArray(), [[10, 20], [10, 20]], 'broadcast');
$dotA = ZTensor::arr([1, 2, 3])->toGpu()->add(1.0);
$dotB = ZTensor::arr([4, 5, 6])->toGpu()->mul(2.0);
if (abs($dotA->dot($dotB) - 94.0) > 1.0e-5) throw new RuntimeException('dot read stale host data');
echo "PASS audited CPU readers synchronize device data\n";

$features = ZTensor::arr([[1, 10], [3, 20], [2, 30], [5, 40]])->toGpu()->add(1.0);
assertSameTree($features->findIndicesWhere(0, 3.0)->toArray(), [0, 2], 'findIndicesWhere');
$labels = ZTensor::arr([0, 1, 0, 1])->toGpu()->add(0.0);
$gpuGini = $features->calculateSplitGini(0, 3.0, $labels);
$cpuGini = ZTensor::arr([[2, 11], [4, 21], [3, 31], [6, 41]])->calculateSplitGini(0, 3.0, ZTensor::arr([0, 1, 0, 1]));
if (abs($gpuGini - $cpuGini) > 1.0e-6) throw new RuntimeException('calculateSplitGini read stale data');
echo "PASS model helper host coherence\n";

// Validation failures happen before mutation and may not corrupt validity flags.
$before = [[2, 4], [6, 8]];
$failed = ZTensor::arr($before)->toGpu();
try {
    $failed->divide(ZTensor::arr([[1, 0], [1, 1]])->toGpu());
    throw new RuntimeException('divide by zero did not fail');
} catch (Throwable $error) {
    if ($error->getMessage() === 'divide by zero did not fail') throw $error;
}
if (!$failed->isOnGpu()) throw new RuntimeException('divide failure invalidated valid input');
assertSameTree($failed->toArray(), $before, 'divide failure atomicity');

$failedLog = ZTensor::arr([[1, 0.5], [0, 4]])->toGpu();
try {
    $failedLog->log();
    throw new RuntimeException('invalid log did not fail');
} catch (Throwable $error) {
    if ($error->getMessage() === 'invalid log did not fail') throw $error;
}
if (!$failedLog->isOnGpu()) throw new RuntimeException('log failure invalidated valid input');
assertSameTree($failedLog->toArray(), [[1, 0.5], [0, 4]], 'log failure atomicity');
echo "PASS failure paths preserve state\n";

for ($i = 0; $i < 300; ++$i) {
    $temporary = ZTensor::arr([[1, 2], [3, 4]])->toGpu()->add((float) $i)->transpose();
    if (($i % 50) === 0) assertSameTree($temporary->toArray(), [[1 + $i, 3 + $i], [2 + $i, 4 + $i]], 'lifecycle stress');
}
unset($temporary);
gc_collect_cycles();
echo "PASS device lifecycle stress\n";

echo "All residency/coherence checks passed.\n";
