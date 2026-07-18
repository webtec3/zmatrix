<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

if (!extension_loaded('zmatrix')) exit(1);

$label = $argv[1] ?? 'run';
$outputDirectory = $argv[2] ?? (__DIR__ . '/results');
$repetitions = (int) ($argv[3] ?? 9);
if ($repetitions < 3) throw new InvalidArgumentException('at least three repetitions are required');
if (!is_dir($outputDirectory) && !mkdir($outputDirectory, 0775, true) && !is_dir($outputDirectory)) throw new RuntimeException('cannot create output directory');

function ms(callable $operation): float
{
    $start = hrtime(true);
    $operation();
    return (hrtime(true) - $start) / 1_000_000.0;
}

function timingStats(array $values): array
{
    sort($values, SORT_NUMERIC);
    $mean = array_sum($values) / count($values);
    $variance = array_sum(array_map(static fn(float $v): float => ($v - $mean) ** 2, $values)) / count($values);
    return [
        'samples' => count($values), 'min_ms' => $values[0],
        'median_ms' => $values[intdiv(count($values), 2)], 'mean_ms' => $mean,
        'stddev_ms' => sqrt($variance), 'p95_ms' => $values[(int) floor((count($values) - 1) * 0.95)],
        'max_ms' => $values[array_key_last($values)], 'raw_ms' => $values,
    ];
}

function validate(ZTensor $actual, ZTensor $expected, float $atol, float $rtol): array
{
    $difference = ZTensor::arr($actual)->toCpu()->sub(ZTensor::arr($expected)->toCpu())->abs()->max();
    $scale = ZTensor::arr($expected)->abs()->max();
    return ['valid' => $difference <= $atol + $rtol * $scale, 'max_abs_error' => $difference,
        'max_relative_error' => $difference / max($scale, 1.17549435e-38),
        'actual_max' => ZTensor::arr($actual)->max(), 'expected_max' => $scale,
        'actual_sum' => ZTensor::arr($actual)->sumtotal(), 'expected_sum' => ZTensor::arr($expected)->sumtotal(),
        'actual_sample' => array_slice($actual->row(0)->toArray(), 0, 8),
        'expected_sample' => array_slice($expected->row(0)->toArray(), 0, 8),
        'atol' => $atol, 'rtol' => $rtol];
}

function benchmark(string $name, ZTensor $base, callable $operation, int $repetitions, float $atol, float $rtol): array
{
    for ($i = 0; $i < 3; ++$i) $operation(ZTensor::arr($base));
    $cpu = [];
    for ($i = 0; $i < $repetitions; ++$i) {
        $candidate = ZTensor::arr($base);
        $cpu[] = ms(static function () use ($operation, $candidate, &$cpuResult): void { $cpuResult = $operation($candidate); });
    }
    $h2d = [];
    for ($i = 0; $i < $repetitions; ++$i) {
        $candidate = ZTensor::arr($base);
        $h2d[] = ms(static function () use ($candidate): void { $candidate->toGpu(); });
    }
    $kernel = [];
    $d2h = [];
    $resident = true;
    for ($i = 0; $i < $repetitions; ++$i) {
        $candidate = ZTensor::arr($base)->toGpu();
        $kernel[] = ms(static function () use ($operation, $candidate, &$gpuResult): void { $gpuResult = $operation($candidate); });
        $resident = $resident && $gpuResult->isOnGpu();
        $d2h[] = ms(static function () use ($gpuResult): void { $gpuResult->toCpu(); });
    }
    $end = [];
    for ($i = 0; $i < $repetitions; ++$i) {
        $candidate = ZTensor::arr($base);
        $end[] = ms(static function () use ($operation, $candidate, &$endResult): void {
            $endResult = $operation($candidate->toGpu())->toCpu();
        });
    }
    $validation = validate($gpuResult, $cpuResult, $atol, $rtol);
    if (!$validation['valid']) throw new RuntimeException("{$name}: CPU/GPU validation failed: " . json_encode($validation, JSON_THROW_ON_ERROR));
    return ['operation' => $name, 'shape' => $base->shape(), 'gpu_result_resident_before_d2h' => $resident,
        'validation' => $validation, 'timings' => ['cpu' => timingStats($cpu), 'h2d' => timingStats($h2d),
            'gpu_operation' => timingStats($kernel), 'd2h' => timingStats($d2h), 'gpu_end_to_end' => timingStats($end)]];
}

$sqrtBase = ZTensor::linspace(0.0, 100.0, 1048576)->reshape([1024, 1024]);
$clipBase = ZTensor::linspace(-10.0, 10.0, 1048576)->reshape([1024, 1024]);
$softmaxBase = ZTensor::linspace(-8.0, 8.0, 1048576)->reshape([4096, 256]);
$records = [
    benchmark('sqrt', $sqrtBase, static fn(ZTensor $t): ZTensor => $t->sqrt(), $repetitions, 3.0e-5, 3.0e-5),
    benchmark('clip', $clipBase, static fn(ZTensor $t): ZTensor => ZTensor::clip($t, -2.5, 3.5), $repetitions, 0.0, 0.0),
    benchmark('softmax', $softmaxBase, static fn(ZTensor $t): ZTensor => $t->softmax(), $repetitions, 5.0e-5, 5.0e-5),
];
$report = ['schema_version' => 1, 'label' => $label, 'created_utc' => gmdate(DATE_ATOM),
    'environment' => ['php' => PHP_VERSION, 'zmatrix' => phpversion('zmatrix'),
        'gpu' => trim((string) shell_exec('nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null'))],
    'methodology' => ['warmups' => 3, 'repetitions' => $repetitions, 'validation' => 'CPU reference required'],
    'records' => $records];
$json = "{$outputDirectory}/new_kernels_{$label}.json";
$csv = "{$outputDirectory}/new_kernels_{$label}.csv";
file_put_contents($json, json_encode($report, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR) . PHP_EOL);
$handle = fopen($csv, 'wb');
fputcsv($handle, ['operation', 'scenario', 'resident', 'median_ms', 'mean_ms', 'stddev_ms', 'p95_ms', 'valid', 'max_abs_error', 'max_relative_error'], ',', '"', '');
foreach ($records as $record) foreach ($record['timings'] as $scenario => $timing) {
    fputcsv($handle, [$record['operation'], $scenario, $record['gpu_result_resident_before_d2h'] ? 1 : 0,
        $timing['median_ms'], $timing['mean_ms'], $timing['stddev_ms'], $timing['p95_ms'],
        $record['validation']['valid'] ? 1 : 0, $record['validation']['max_abs_error'],
        $record['validation']['max_relative_error']], ',', '"', '');
}
fclose($handle);
echo json_encode(['valid' => true, 'json' => $json, 'csv' => $csv], JSON_PRETTY_PRINT), PHP_EOL;
