<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

[$script, $operationName, $sizeText, $scenario] = $argv + [null, null, null, null];
if (!in_array($operationName, ['dot', 'matvec'], true) || !ctype_digit((string) $sizeText)
    || !in_array($scenario, ['cpu_cpu', 'gpu_gpu', 'gpu_cpu', 'cpu_gpu'], true)) {
    fwrite(STDERR, "usage: isolated_linalg_case.php dot|matvec size cpu_cpu|gpu_gpu|gpu_cpu|cpu_gpu\n");
    exit(2);
}
$size = (int) $sizeText;

function isolatedStats(array $samples): array {
    $sorted = $samples; sort($sorted, SORT_NUMERIC);
    $at = static function (float $p) use ($sorted): float {
        $position = $p * (count($sorted) - 1); $lo = (int) floor($position); $hi = (int) ceil($position);
        return $sorted[$lo] + ($sorted[$hi] - $sorted[$lo]) * ($position - $lo);
    };
    $median = $at(0.5); $deviations = array_map(static fn($v) => abs($v - $median), $samples);
    sort($deviations, SORT_NUMERIC);
    return ['min_ms' => min($samples), 'p25_ms' => $at(0.25), 'median_ms' => $median,
        'p75_ms' => $at(0.75), 'max_ms' => max($samples),
        'mad_ms' => $deviations[intdiv(count($deviations), 2)], 'samples_ms' => $samples];
}

$makeLeft = $operationName === 'dot'
    ? static fn() => ZTensor::ones([$size])
    : static fn() => ZTensor::ones([$size, $size]);
$makeRight = static fn() => ZTensor::ones([$size]);
$left = $makeLeft(); $right = $makeRight();
if ($scenario === 'gpu_gpu' || $scenario === 'gpu_cpu') $left->toGpu();
if ($scenario === 'gpu_gpu' || $scenario === 'cpu_gpu') $right->toGpu();

$invoke = static function () use ($operationName, $scenario, $makeLeft, $makeRight, $left, $right) {
    $a = $scenario === 'cpu_gpu' ? $makeLeft() : $left;
    $b = $scenario === 'gpu_cpu' ? $makeRight() : $right;
    return $a->dot($b);
};

$start = hrtime(true); $cold = $invoke(); $coldMs = (hrtime(true) - $start) / 1.0e6; unset($cold);
for ($i = 0; $i < 3; ++$i) { $warm = $invoke(); unset($warm); }
$validationResult = $invoke();
if ($operationName === 'dot') {
    if (abs((float) $validationResult - $size) > max(0.01, $size * 1.0e-4)) throw new RuntimeException('dot validation failed');
} else {
    if ($validationResult->isOnGpu()) $validationResult->toCpu();
    $sum = (float) $validationResult->sum()->toArray()[0];
    if (abs($sum - ($size * $size)) > max(0.1, $size * $size * 1.0e-4)) throw new RuntimeException('matvec validation failed');
}
unset($validationResult);
$operationSamples = $materializationSamples = $destructionSamples = $e2eSamples = [];
for ($i = 0; $i < 15; ++$i) {
    $e2eStart = hrtime(true);
    $start = hrtime(true); $result = $invoke(); $operationSamples[] = (hrtime(true) - $start) / 1.0e6;
    $start = hrtime(true);
    if ($result instanceof ZTensor && $result->isOnGpu()) $result->toCpu();
    $materializationSamples[] = (hrtime(true) - $start) / 1.0e6;
    $start = hrtime(true); unset($result); $destructionSamples[] = (hrtime(true) - $start) / 1.0e6;
    $e2eSamples[] = (hrtime(true) - $e2eStart) / 1.0e6;
}

echo json_encode([
    'operation' => $operationName, 'size' => $size, 'scenario' => $scenario,
    'allocator' => getenv('ZMATRIX_CUDA_ALLOCATOR') ?: 'auto', 'cold_ms' => $coldMs,
    'operation_stats' => isolatedStats($operationSamples),
    'materialization_stats' => isolatedStats($materializationSamples),
    'destruction_stats' => isolatedStats($destructionSamples),
    'e2e_stats' => isolatedStats($e2eSamples), 'validated' => true,
], JSON_THROW_ON_ERROR) . "\n";
