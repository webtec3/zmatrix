<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

const ALLOCATION_REPETITIONS = 15;

function allocationStats(array $samples): array {
    sort($samples, SORT_NUMERIC);
    $at = static function (float $p) use ($samples): float {
        $position = $p * (count($samples) - 1);
        $lo = (int) floor($position); $hi = (int) ceil($position);
        return $samples[$lo] + ($samples[$hi] - $samples[$lo]) * ($position - $lo);
    };
    return ['min_ms' => min($samples), 'p25_ms' => $at(0.25), 'median_ms' => $at(0.5),
        'p75_ms' => $at(0.75), 'max_ms' => max($samples), 'samples_ms' => $samples];
}

$sizes = [1024 => 256, 1048576 => 262144, 4194304 => 1048576,
    16777216 => 4194304, 67108864 => 16777216];
$results = [];
foreach ($sizes as $bytes => $elements) {
    $source = ZTensor::ones([$elements])->toGpu();
    $coldStart = hrtime(true); $cold = $source->greater(0); $coldMs = (hrtime(true) - $coldStart) / 1.0e6;
    unset($cold);
    for ($i = 0; $i < 3; ++$i) { $warm = $source->greater(0); unset($warm); }
    $operation = $destruction = [];
    for ($i = 0; $i < ALLOCATION_REPETITIONS; ++$i) {
        $start = hrtime(true); $result = $source->greater(0); $operation[] = (hrtime(true) - $start) / 1.0e6;
        $start = hrtime(true); unset($result); $destruction[] = (hrtime(true) - $start) / 1.0e6;
    }
    $results[] = ['bytes' => $bytes, 'elements' => $elements, 'cold_ms' => $coldMs,
        'operation' => allocationStats($operation), 'destruction' => allocationStats($destruction)];
    unset($source);
}

$root = dirname(__DIR__, 2);
$document = ['environment' => [
    'allocator' => getenv('ZMATRIX_CUDA_ALLOCATOR') ?: 'auto', 'repetitions' => ALLOCATION_REPETITIONS,
    'commit' => trim((string) shell_exec('git -C ' . escapeshellarg($root) . ' rev-parse HEAD')),
    'binary_sha256' => hash_file('sha256', $root . '/modules/zmatrix.so'),
], 'results' => $results];
$suffix = date('Ymd_His');
$path = __DIR__ . "/results/allocation_lifecycle_$suffix.json";
file_put_contents($path, json_encode($document, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . "\n");
echo "$path\n";
