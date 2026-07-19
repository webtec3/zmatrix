<?php
use ZMatrix\ZTensor;

function stats(array $samples): array {
    sort($samples);
    $count = count($samples);
    return [
        'median_ms' => $samples[intdiv($count, 2)],
        'mean_ms' => array_sum($samples) / $count,
        'min_ms' => $samples[0],
        'max_ms' => $samples[$count - 1],
    ];
}

function measure(callable $operation, int $warmups = 5, int $iterations = 31): array {
    for ($i = 0; $i < $warmups; ++$i) {
        $operation();
    }
    $samples = [];
    for ($i = 0; $i < $iterations; ++$i) {
        $start = hrtime(true);
        $operation();
        $samples[] = (hrtime(true) - $start) / 1e6;
    }
    return stats($samples);
}

$results = [];
foreach ([128, 512, 1024] as $size) {
    $tensor = ZTensor::ones([$size, $size]);
    $results[(string) $size] = [
        'transpose_view' => measure(fn() => $tensor->transpose()),
        'transpose_then_materialize' => measure(
            fn() => ZTensor::clip($tensor->transpose(), -INF, INF)
        ),
    ];
}

echo json_encode([
    'version' => phpversion('zmatrix'),
    'php' => PHP_VERSION,
    'results' => $results,
], JSON_PRETTY_PRINT), PHP_EOL;
