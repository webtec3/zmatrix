<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

const LIFECYCLE_WARMUPS = 3;
const LIFECYCLE_REPETITIONS = 15;

function lifeMs(callable $callable): array {
    $start = hrtime(true);
    $value = $callable();
    return [(hrtime(true) - $start) / 1.0e6, $value];
}

function lifeStats(array $values): array {
    $sorted = $values;
    sort($sorted, SORT_NUMERIC);
    $percentile = static function (float $p) use ($sorted): float {
        $position = $p * (count($sorted) - 1);
        $lo = (int) floor($position);
        $hi = (int) ceil($position);
        return $sorted[$lo] + ($sorted[$hi] - $sorted[$lo]) * ($position - $lo);
    };
    $median = $percentile(0.5);
    $deviations = array_map(static fn(float $v): float => abs($v - $median), $values);
    sort($deviations, SORT_NUMERIC);
    return [
        'min_ms' => min($values), 'p25_ms' => $percentile(0.25),
        'median_ms' => $median, 'p75_ms' => $percentile(0.75),
        'max_ms' => max($values), 'mad_ms' => $deviations[intdiv(count($deviations), 2)],
        'samples_ms' => $values,
    ];
}

function lifeRss(): int {
    $status = file_get_contents('/proc/self/status') ?: '';
    return preg_match('/VmRSS:\s+(\d+)\s+kB/', $status, $matches) ? (int) $matches[1] * 1024 : 0;
}

$greater = ZTensor::linspace(-1, 1, 1048576)->toGpu();
$tile = ZTensor::ones([1024, 1024])->toGpu();
$scan = ZTensor::ones([1048576])->toGpu();
$matrix = ZTensor::ones([2048, 2048])->toGpu();
$vector = ZTensor::ones([2048])->toGpu();
$dotA = ZTensor::ones([1048576])->toGpu();
$dotB = ZTensor::ones([1048576])->toGpu();

$cases = [
    'greater_1m' => fn() => $greater->greater(0),
    'tile_1024sq_x2' => fn() => ZTensor::tile($tile, 2),
    'cumsum_1m' => fn() => $scan->cumsum(),
    'dot_1m' => fn() => $dotA->dot($dotB),
    'matvec_2048sq' => fn() => $matrix->dot($vector),
];

$results = [];
foreach ($cases as $name => $operation) {
    [$cold, $coldResult] = lifeMs($operation);
    unset($coldResult);
    gc_collect_cycles();
    for ($i = 0; $i < LIFECYCLE_WARMUPS; ++$i) {
        $result = $operation();
        unset($result);
    }
    $operationSamples = $destructionSamples = $gcSamples = [];
    for ($i = 0; $i < LIFECYCLE_REPETITIONS; ++$i) {
        [$operationMs, $result] = lifeMs($operation);
        $destroyStart = hrtime(true);
        unset($result);
        $destructionMs = (hrtime(true) - $destroyStart) / 1.0e6;
        $gcStart = hrtime(true);
        gc_collect_cycles();
        $gcMs = (hrtime(true) - $gcStart) / 1.0e6;
        $operationSamples[] = $operationMs;
        $destructionSamples[] = $destructionMs;
        $gcSamples[] = $gcMs;
    }
    $results[$name] = [
        'cold_ms' => $cold,
        'operation' => lifeStats($operationSamples),
        'destruction' => lifeStats($destructionSamples),
        'gc' => lifeStats($gcSamples),
    ];
}

$root = dirname(__DIR__, 2);
$document = [
    'environment' => [
        'timestamp' => date(DATE_ATOM), 'warmups' => LIFECYCLE_WARMUPS,
        'repetitions' => LIFECYCLE_REPETITIONS, 'allocator' => getenv('ZMATRIX_CUDA_ALLOCATOR') ?: 'auto',
        'commit' => trim((string) shell_exec('git -C ' . escapeshellarg($root) . ' rev-parse HEAD')),
        'binary_sha256' => hash_file('sha256', $root . '/modules/zmatrix.so'),
        'php_memory_bytes' => memory_get_usage(true), 'php_peak_bytes' => memory_get_peak_usage(true),
        'rss_bytes' => lifeRss(),
    ],
    'results' => $results,
];
$suffix = date('Ymd_His');
$json = __DIR__ . "/results/wrapper_lifecycle_$suffix.json";
$csv = __DIR__ . "/results/wrapper_lifecycle_$suffix.csv";
file_put_contents($json, json_encode($document, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . "\n");
$handle = fopen($csv, 'wb');
fputcsv($handle, ['operation', 'phase', 'cold_ms', 'min_ms', 'p25_ms', 'median_ms', 'p75_ms', 'max_ms', 'mad_ms'], ',', '"', '');
foreach ($results as $operation => $result) {
    foreach (['operation', 'destruction', 'gc'] as $phase) {
        $s = $result[$phase];
        fputcsv($handle, [$operation, $phase, $result['cold_ms'], $s['min_ms'], $s['p25_ms'],
            $s['median_ms'], $s['p75_ms'], $s['max_ms'], $s['mad_ms']], ',', '"', '');
    }
}
fclose($handle);
echo "$json\n$csv\n";
