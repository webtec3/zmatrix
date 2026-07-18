<?php

use ZMatrix\ZTensor;

function fail(string $message): never { throw new RuntimeException($message); }
function assertResident(ZTensor $tensor, string $name): void {
    if (!$tensor->isOnGpu()) fail("$name did not remain device-resident");
}
function assertTree(mixed $actual, mixed $expected, string $name, float $atol = 0.0, float $rtol = 0.0): void {
    if (is_array($expected)) {
        if (!is_array($actual) || count($actual) !== count($expected)) fail("$name shape mismatch");
        foreach ($expected as $i => $value) assertTree($actual[$i], $value, "{$name}[{$i}]", $atol, $rtol);
        return;
    }
    $a = (float) $actual;
    $e = (float) $expected;
    if (is_nan($e)) { if (!is_nan($a)) fail("$name expected NaN, got $a"); return; }
    if (is_infinite($e)) { if ($a !== $e) fail("$name infinity mismatch"); return; }
    $error = abs($a - $e);
    if ($error > $atol + $rtol * abs($e)) fail("$name mismatch: $a vs $e (error $error)");
}
function expectException(callable $callable, string $name): void {
    try { $callable(); } catch (Throwable) { return; }
    fail("$name did not throw");
}
function passed(string $name): void { echo "PASS $name\n"; }

$special = [NAN, INF, -INF, 0.0, -0.0, 2.0, -2.0];
$cpu = ZTensor::arr($special)->greater(0.0)->toArray();
$gpu = ZTensor::arr($special)->toGpu()->greater(0.0);
assertResident($gpu, 'greater scalar');
assertTree($gpu->toArray(), $cpu, 'greater scalar');

$cpu = ZTensor::arr($special)->greater($special)->toArray();
$gpu = ZTensor::arr($special)->toGpu()->greater(ZTensor::arr($special)->toGpu());
assertResident($gpu, 'greater tensor');
assertTree($gpu->toArray(), $cpu, 'greater tensor');

$cpu = ZTensor::arr([[1, 5, -1], [4, 0, 9]])->greater([2, 0, 8])->toArray();
$gpu = ZTensor::arr([[1, 5, -1], [4, 0, 9]])->toGpu()->greater(ZTensor::arr([2, 0, 8]));
assertResident($gpu, 'greater broadcast');
assertTree($gpu->toArray(), $cpu, 'greater broadcast');

$empty = ZTensor::zeros([0])->toGpu()->greater(0);
assertResident($empty, 'greater empty');
assertTree($empty->toArray(), [], 'greater empty');
passed('greater semantics and residency');

$broadcastCases = [
    [[2, 3], [10, 20, 30]],
    [[2, 3], [[10, 20, 30]]],
    [[2, 2, 3], [[10, 20, 30]]],
    [[2, 2, 3], [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]],
];
foreach ($broadcastCases as $caseIndex => [$shape, $source]) {
    $cpu = ZTensor::zeros($shape)->broadcast(ZTensor::arr($source))->toArray();
    $gpu = ZTensor::zeros($shape)->toGpu()->broadcast(ZTensor::arr($source));
    assertResident($gpu, "broadcast $caseIndex");
    assertTree($gpu->toArray(), $cpu, "broadcast $caseIndex");
}
expectException(fn() => ZTensor::zeros([2, 3])->broadcast(ZTensor::ones([2, 2])), 'broadcast incompatible');
$empty = ZTensor::zeros([0, 3])->toGpu()->broadcast(ZTensor::arr([1, 2, 3]));
assertResident($empty, 'broadcast empty');
assertTree($empty->toArray(), [], 'broadcast empty');
passed('broadcast shapes failures and residency');

$tileCases = [
    [[1, 2, 3], 3],
    [[[1, 2], [3, 4]], 2],
    [[[1, 2], [3, 4]], 1],
];
foreach ($tileCases as $caseIndex => [$values, $times]) {
    $cpu = ZTensor::tile(ZTensor::arr($values), $times)->toArray();
    $gpu = ZTensor::tile(ZTensor::arr($values)->toGpu(), $times);
    assertResident($gpu, "tile $caseIndex");
    assertTree($gpu->toArray(), $cpu, "tile $caseIndex");
}
expectException(fn() => ZTensor::tile(ZTensor::arr([1]), 0), 'tile zero');
expectException(fn() => ZTensor::tile(ZTensor::arr([1]), -1), 'tile negative');
$empty = ZTensor::tile(ZTensor::zeros([0, 2])->toGpu(), 2);
assertResident($empty, 'tile empty');
assertTree($empty->toArray(), [], 'tile empty');
passed('tile mappings failures and residency');

$cumsumCases = [
    [[1, 2, 3, 4], null],
    [[1.0e8, 1, -1.0e8, 3, -2], null],
    [[[1, 2, 3], [4, 5, 6]], 0],
    [[[1, 2, 3], [4, 5, 6]], 1],
    [[[NAN, 1, 2], [INF, -INF, 1]], 1],
];
foreach ($cumsumCases as $caseIndex => [$values, $axis]) {
    $cpuTensor = ZTensor::arr($values);
    $gpuTensor = ZTensor::arr($values)->toGpu();
    $cpu = $axis === null ? $cpuTensor->cumsum()->toArray() : $cpuTensor->cumsum($axis)->toArray();
    $gpu = $axis === null ? $gpuTensor->cumsum() : $gpuTensor->cumsum($axis);
    assertResident($gpu, "cumsum $caseIndex");
    assertTree($gpu->toArray(), $cpu, "cumsum $caseIndex", 2.0, 2.0e-6);
}
expectException(fn() => ZTensor::ones([2, 2])->cumsum(2), 'cumsum invalid axis');
$empty = ZTensor::zeros([0])->toGpu()->cumsum();
assertResident($empty, 'cumsum empty');
assertTree($empty->toArray(), [], 'cumsum empty');
passed('cumsum axes numerical behavior and residency');

$longValues = array_map(static fn(int $i): float => (($i % 17) - 8) / 17.0, range(0, 65536));
$longCpu = ZTensor::arr($longValues)->cumsum()->toArray();
$longGpu = ZTensor::arr($longValues)->toGpu()->cumsum();
assertResident($longGpu, 'cumsum non-multiple');
assertTree($longGpu->toArray(), $longCpu, 'cumsum non-multiple', 0.05, 5.0e-5);
passed('cumsum long non-multiple scan');

$dotCases = [
    [[1, 2, 3], [4, 5, 6]],
    [[-2, 3, -4], [5, -6, 7]],
    [[0, 1, 0, 1], [1, 0, 1, 0]],
    [[NAN, 1], [2, 3]],
];
foreach ($dotCases as $caseIndex => [$left, $right]) {
    $cpu = ZTensor::arr($left)->dot($right);
    $gpu = ZTensor::arr($left)->toGpu()->dot(ZTensor::arr($right)->toGpu());
    assertTree($gpu, $cpu, "dot $caseIndex", 1.0e-4, 1.0e-5);
}
assertTree(ZTensor::zeros([0])->toGpu()->dot(ZTensor::zeros([0])->toGpu()), 0.0, 'dot empty');
expectException(fn() => ZTensor::ones([2])->toGpu()->dot(ZTensor::ones([3])->toGpu()), 'dot incompatible');
passed('dot semantics and CUDA scalar result');

$largeLeft = array_map(static fn(int $i): float => (($i % 31) - 15) / 31.0, range(0, 65536));
$largeRight = array_map(static fn(int $i): float => (($i % 29) - 14) / 29.0, range(0, 65536));
$largeCpuDot = ZTensor::arr($largeLeft)->dot($largeRight);
$largeGpuDot = ZTensor::arr($largeLeft)->toGpu()->dot(ZTensor::arr($largeRight)->toGpu());
assertTree($largeGpuDot, $largeCpuDot, 'dot large', 0.01, 1.0e-4);
passed('dot large');

$matvecCases = [
    [[[1, 2], [3, 4]], [5, 6]],
    [[[1, 2, 3], [4, 5, 6]], [1, -1, 2]],
    [[[1, 2], [3, 4], [5, 6]], [-2, 3]],
    [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [7, 8, 9]],
];
foreach ($matvecCases as $caseIndex => [$matrix, $vector]) {
    $cpu = ZTensor::arr($matrix)->dot($vector)->toArray();
    $gpu = ZTensor::arr($matrix)->toGpu()->dot(ZTensor::arr($vector)->toGpu());
    assertResident($gpu, "matvec $caseIndex");
    assertTree($gpu->toArray(), $cpu, "matvec $caseIndex", 1.0e-4, 1.0e-5);
}
expectException(fn() => ZTensor::ones([2, 3])->toGpu()->dot(ZTensor::ones([2])->toGpu()), 'matvec incompatible');
passed('matvec shapes and residency');

$chain = ZTensor::zeros([2, 3])->toGpu()
    ->broadcast(ZTensor::arr([[1, 4, 9]]))
    ->sqrt()
    ->softmax();
assertResident($chain, 'broadcast chain');
assertTree($chain->toArray(), ZTensor::arr([[1, 4, 9], [1, 4, 9]])->sqrt()->softmax()->toArray(), 'broadcast chain', 1.0e-5, 1.0e-5);

$chain = ZTensor::arr([-2, 1, 3, -1])->toGpu()->greater(0)->cumsum();
assertResident($chain, 'comparison cumsum chain');
assertTree($chain->toArray(), [0, 1, 2, 2], 'comparison cumsum chain');

$chain = ZTensor::arr([[1, 2], [3, 4]])->toGpu()->dot(ZTensor::arr([1, 1]))->softmax();
assertResident($chain, 'matvec softmax chain');
assertTree($chain->toArray(), ZTensor::arr([3, 7])->softmax()->toArray(), 'matvec softmax chain', 1.0e-5, 1.0e-5);
passed('resident chains');

echo "All extended CUDA operation checks passed.\n";
