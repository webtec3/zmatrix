<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

const MATRIX_WARMUPS = 3;
const MATRIX_REPETITIONS = 7;

function matrixMilliseconds(callable $operation): array {
    $start = hrtime(true);
    $result = $operation();
    return [(hrtime(true) - $start) / 1.0e6, $result];
}

function matrixPercentile(array $samples, float $percentile): float {
    sort($samples, SORT_NUMERIC);
    $position = ($percentile / 100.0) * (count($samples) - 1);
    $lower = (int) floor($position);
    $upper = (int) ceil($position);
    return $samples[$lower] + ($samples[$upper] - $samples[$lower]) * ($position - $lower);
}

function matrixScalar(ZTensor $tensor): float {
    return (float) $tensor->toArray()[0];
}

function matrixClose(float $actual, float $expected, float $rtol = 2.0e-4): bool {
    return abs($actual - $expected) <= 0.1 + $rtol * abs($expected);
}

function matrixMeasure(string $operation, string $size, callable $callable,
                       callable $validate, int $outputBytes): array {
    [$coldMs, $coldResult] = matrixMilliseconds($callable);
    if (!$coldResult instanceof ZTensor || !$coldResult->isOnGpu() || !$validate($coldResult)) {
        throw new RuntimeException("$operation $size failed validation or residency");
    }
    unset($coldResult);

    for ($warmup = 0; $warmup < MATRIX_WARMUPS; ++$warmup) {
        $result = $callable();
        unset($result);
    }
    $samples = [];
    $last = null;
    for ($iteration = 0; $iteration < MATRIX_REPETITIONS; ++$iteration) {
        [$elapsed, $result] = matrixMilliseconds($callable);
        $samples[] = $elapsed;
        unset($last);
        $last = $result;
    }
    if (!$last instanceof ZTensor || !$last->isOnGpu() || !$validate($last)) {
        throw new RuntimeException("$operation $size failed steady-state validation or residency");
    }
    unset($last);
    $median = matrixPercentile($samples, 50);
    return [
        'operation' => $operation,
        'size' => $size,
        'cold_ms' => $coldMs,
        'median_ms' => $median,
        'p25_ms' => matrixPercentile($samples, 25),
        'p75_ms' => matrixPercentile($samples, 75),
        'samples_ms' => $samples,
        'effective_gbps' => $median > 0.0 ? ($outputBytes / 1.0e9) / ($median / 1000.0) : null,
        'validated' => true,
    ];
}

$results = [];
foreach ([1024, 65536, 1048576, 16777216] as $count) {
    $input = ZTensor::linspace(-1, 1, $count)->toGpu();
    $expected = matrixScalar(ZTensor::linspace(-1, 1, $count)->greater(0)->sum());
    $results[] = matrixMeasure('greater_scalar', "$count elements", fn() => $input->greater(0),
        fn(ZTensor $result) => matrixClose(matrixScalar($result->sum()), $expected, 0.0), $count * 4);
    unset($input);
}

foreach ([256, 1024, 2048] as $side) {
    $inputCount = $side * $side;
    $input = ZTensor::ones([$side, $side])->toGpu();
    foreach ([2, 4, 8, 16] as $factor) {
        $outputCount = $inputCount * $factor;
        $results[] = matrixMeasure('tile', "{$side}x{$side} x$factor", fn() => ZTensor::tile($input, $factor),
            fn(ZTensor $result) => $result->shape() === [$side * $factor, $side]
                && matrixClose(matrixScalar($result->sum()), (float) $outputCount), $outputCount * 4);
    }
    unset($input);
}

foreach ([1024, 65536, 1048576, 8388608] as $count) {
    $input = ZTensor::ones([$count])->toGpu();
    $expected = ((float) $count * ($count + 1.0)) / 2.0;
    $results[] = matrixMeasure('cumsum_1d', "$count elements", fn() => $input->cumsum(),
        fn(ZTensor $result) => matrixClose(matrixScalar($result->sum()), $expected, 5.0e-4), $count * 4);
    unset($input);
}

foreach ([256, 1024, 2048] as $side) {
    $count = $side * $side;
    $input = ZTensor::ones([$side, $side])->toGpu();
    $expected = (float) $side * $side * ($side + 1.0) / 2.0;
    foreach ([0, 1] as $axis) {
        $results[] = matrixMeasure("cumsum_axis_$axis", "{$side}x{$side}", fn() => $input->cumsum($axis),
            fn(ZTensor $result) => matrixClose(matrixScalar($result->sum()), $expected, 5.0e-4), $count * 4);
    }
    unset($input);
}

$matrix = ZTensor::ones([2048, 2048]);
$vector = ZTensor::ones([2048]);
foreach (['both_resident', 'matrix_resident', 'vector_resident'] as $residency) {
    if ($residency === 'both_resident') {
        $a = ZTensor::arr($matrix)->toGpu();
        $x = ZTensor::arr($vector)->toGpu();
        $operation = fn() => $a->dot($x);
    } elseif ($residency === 'matrix_resident') {
        $a = ZTensor::arr($matrix)->toGpu();
        $operation = fn() => $a->dot(ZTensor::arr($vector));
    } else {
        $x = ZTensor::arr($vector)->toGpu();
        $operation = fn() => ZTensor::arr($matrix)->dot($x);
    }
    $results[] = matrixMeasure('matvec_' . $residency, '2048x2048 @ 2048', $operation,
        fn(ZTensor $result) => matrixClose(matrixScalar($result->sum()), 2048.0 * 2048.0), 2048 * 4);
    unset($a, $x, $operation);
}

$document = [
    'environment' => [
        'timestamp' => date(DATE_ATOM),
        'warmups' => MATRIX_WARMUPS,
        'repetitions' => MATRIX_REPETITIONS,
        'php' => PHP_VERSION,
        'zmatrix' => phpversion('zmatrix'),
        'commit' => trim((string) shell_exec('git rev-parse HEAD')),
        'binary_sha256' => hash_file('sha256', __DIR__ . '/../../modules/zmatrix.so'),
    ],
    'results' => $results,
];
$suffix = date('Ymd_His');
$jsonPath = __DIR__ . "/results/hotpath_matrix_$suffix.json";
$csvPath = __DIR__ . "/results/hotpath_matrix_$suffix.csv";
file_put_contents($jsonPath, json_encode($document, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . "\n");
$csv = fopen($csvPath, 'wb');
fputcsv($csv, ['operation', 'size', 'cold_ms', 'median_ms', 'p25_ms', 'p75_ms', 'effective_gbps', 'validated']);
foreach ($results as $result) {
    fputcsv($csv, [$result['operation'], $result['size'], $result['cold_ms'], $result['median_ms'],
        $result['p25_ms'], $result['p75_ms'], $result['effective_gbps'], 'true']);
}
fclose($csv);
echo "$jsonPath\n$csvPath\n";
