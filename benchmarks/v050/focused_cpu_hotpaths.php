<?php
use ZMatrix\ZTensor;
if (!extension_loaded('zmatrix')) {
    throw new RuntimeException('zmatrix extension is not loaded');
}

function elapsed_ms(callable $fn): float {
    $start = hrtime(true);
    $fn();
    return (hrtime(true) - $start) / 1e6;
}

function stats(array $values): array {
    sort($values);
    $n = count($values);
    return [
        'min_ms' => $values[0],
        'median_ms' => $values[intdiv($n, 2)],
        'mean_ms' => array_sum($values) / $n,
        'p95_ms' => $values[(int) floor(($n - 1) * 0.95)],
        'samples' => $values,
    ];
}

function bench(string $name, callable $fn, int $repetitions = 9, int $warmups = 3): array {
    for ($i = 0; $i < $warmups; ++$i) $fn();
    $times = [];
    for ($i = 0; $i < $repetitions; ++$i) $times[] = elapsed_ms($fn);
    return ['operation' => $name] + stats($times);
}

$argBase = ZTensor::random([512, 512, 4], -1.0, 1.0);
$clipBase = ZTensor::linspace(-10.0, 10.0, 1048576);
$a = ZTensor::ones([512, 512]);
$b = ZTensor::ones([512, 512]);

$results = [
    bench('ones_1024x1024', fn() => ZTensor::ones([1024, 1024])),
    bench('full_1024x1024', fn() => ZTensor::full([1024, 1024], 3.5)),
    bench('clip_1048576', fn() => ZTensor::clip($clipBase, -2.5, 3.5)),
    bench('argmax_axis2_512x512x4', fn() => $argBase->argmax(2), 7),
    bench('argmin_axis2_512x512x4', fn() => $argBase->argmin(2), 7),
    bench('matmul_512_cpu', fn() => $a->matmul($b), 5),
];

$payload = [
    'version' => phpversion('zmatrix'),
    'php' => PHP_VERSION,
    'timestamp' => gmdate('c'),
    'results' => $results,
];

echo json_encode($payload, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES) . PHP_EOL;
