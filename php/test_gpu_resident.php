<?php
// Benchmark with GPU-resident tensors (no host roundtrip per op)

use ZMatrix\ZTensor;

$rows = 1024;
$cols = 1024;
$n = $rows * $cols;
$iters = 10;

echo "N = {$n} iters={$iters}\n";

$a = ZTensor::random([$rows, $cols], -1.0, 1.0);
$b = ZTensor::random([$rows, $cols], -1.0, 1.0);

$a->toGpu();
$b->toGpu();
echo "on_gpu: a=" . ($a->isOnGpu() ? "1" : "0") . " b=" . ($b->isOnGpu() ? "1" : "0") . "\n";

function bench($label, $fn, $iters) {
    $times = [];
    for ($i = 0; $i < $iters; $i++) {
        $t0 = microtime(true);
        $fn();
        $t1 = microtime(true);
        $times[] = ($t1 - $t0) * 1000.0;
    }
    $avg = array_sum($times) / $iters;
    $min = min($times);
    $max = max($times);
    echo sprintf("%-18s avg %8.2f ms | min %8.2f | max %8.2f\n", $label, $avg, $min, $max);
}

bench("add", function() use ($a, $b) { $a->add($b); }, $iters);
bench("mul", function() use ($a, $b) { $a->mul($b); }, $iters);
bench("relu", function() use ($a) { $a->relu(); }, $iters);
bench("sigmoid", function() use ($a) { $a->sigmoid(); }, $iters);
bench("exp", function() use ($a) { $a->exp(); }, $iters);

// Bring back to CPU if you need host data
$a->toCpu();
echo "done\n";
