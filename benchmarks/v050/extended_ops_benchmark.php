<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

const WARMUPS = 2;
const REPETITIONS = 7;

function milliseconds(callable $callable): array {
    $start = hrtime(true);
    $value = $callable();
    return [(hrtime(true) - $start) / 1.0e6, $value];
}
function percentile(array $values, float $p): float {
    sort($values, SORT_NUMERIC);
    $position = ($p / 100.0) * (count($values) - 1);
    $lower = (int) floor($position);
    $upper = (int) ceil($position);
    return $values[$lower] + ($values[$upper] - $values[$lower]) * ($position - $lower);
}
function summarize(array $values): array {
    return ['median_ms' => percentile($values, 50), 'p25_ms' => percentile($values, 25),
        'p75_ms' => percentile($values, 75), 'samples_ms' => $values];
}
function sameTree(mixed $actual, mixed $expected, float $atol, float $rtol): bool {
    if (is_array($expected)) {
        if (!is_array($actual) || count($actual) !== count($expected)) return false;
        foreach ($expected as $index => $value) if (!sameTree($actual[$index], $value, $atol, $rtol)) return false;
        return true;
    }
    $a = (float) $actual; $e = (float) $expected;
    if (is_nan($e)) return is_nan($a);
    if (is_infinite($e)) return $a === $e;
    return abs($a - $e) <= $atol + $rtol * abs($e);
}
function residentCopy(ZTensor $tensor): ZTensor { return ZTensor::arr($tensor)->toGpu(); }
function rssBytes(): int {
    $status = file_get_contents('/proc/self/status') ?: '';
    return preg_match('/VmRSS:\s+(\d+)\s+kB/', $status, $matches) ? (int) $matches[1] * 1024 : 0;
}
function command(string $command): string { return trim((string) shell_exec($command . ' 2>/dev/null')); }

$cases = [
    'greater' => [
        'size' => '1048576 elements',
        'factory' => fn() => [ZTensor::linspace(-2, 2, 1048576)],
        'op' => fn(array $x) => $x[0]->greater(0.125),
        'validate' => fn($gpu, $cpu) => sameTree($gpu->toArray(), $cpu->toArray(), 0, 0),
    ],
    'broadcast' => [
        'size' => '1024x1024 <- 1x1024',
        'factory' => fn() => [ZTensor::zeros([1024, 1024]), ZTensor::linspace(-1, 1, 1024)->reshape([1, 1024])],
        'op' => fn(array $x) => $x[0]->broadcast($x[1]),
        'validate' => fn($gpu, $cpu) => sameTree($gpu->toArray(), $cpu->toArray(), 0, 0),
    ],
    'tile' => [
        'size' => '1024x1024 x2',
        'factory' => fn() => [ZTensor::linspace(-1, 1, 1048576)->reshape([1024, 1024])],
        'op' => fn(array $x) => ZTensor::tile($x[0], 2),
        'validate' => fn($gpu, $cpu) => sameTree($gpu->toArray(), $cpu->toArray(), 0, 0),
    ],
    'cumsum' => [
        'size' => '1048576 elements',
        'factory' => fn() => [ZTensor::linspace(-0.001, 0.001, 1048576)],
        'op' => fn(array $x) => $x[0]->cumsum(),
        'validate' => fn($gpu, $cpu) => sameTree($gpu->toArray(), $cpu->toArray(), 0.05, 5.0e-5),
    ],
    'dot' => [
        'size' => '1048576 elements',
        'factory' => fn() => [ZTensor::linspace(-1, 1, 1048576), ZTensor::linspace(1, -1, 1048576)],
        'op' => fn(array $x) => $x[0]->dot($x[1]),
        // One million float products are accumulated in a different order by
        // cuBLAS than by the scalar CPU loop. The observed relative delta is
        // about 3.7e-4, so this bound records rather than hides that behavior.
        'validate' => fn($gpu, $cpu) => sameTree($gpu, $cpu, 0.1, 5.0e-4),
    ],
    'matvec' => [
        'size' => '2048x2048 @ 2048',
        'factory' => fn() => [ZTensor::ones([2048, 2048]), ZTensor::linspace(-1, 1, 2048)],
        'op' => fn(array $x) => $x[0]->dot($x[1]),
        'validate' => fn($gpu, $cpu) => sameTree($gpu->toArray(), $cpu->toArray(), 0.01, 1.0e-4),
    ],
    'chain_greater_cumsum_clip' => [
        'size' => '1048576 elements',
        'factory' => fn() => [ZTensor::linspace(-2, 2, 1048576)],
        'op' => fn(array $x) => ZTensor::clip($x[0]->greater(0)->cumsum(), 0, 1048576),
        'validate' => fn($gpu, $cpu) => sameTree($gpu->toArray(), $cpu->toArray(), 0.05, 5.0e-5),
    ],
    'chain_broadcast_greater_cumsum' => [
        'size' => '1024x1024 <- 1x1024',
        'factory' => fn() => [ZTensor::zeros([1024, 1024]), ZTensor::linspace(-1, 1, 1024)->reshape([1, 1024])],
        'op' => fn(array $x) => $x[0]->broadcast($x[1])->greater(0)->cumsum(1),
        'validate' => fn($gpu, $cpu) => sameTree($gpu->toArray(), $cpu->toArray(), 0.05, 5.0e-5),
    ],
    'chain_matvec_clip_softmax' => [
        'size' => '2048x2048 @ 2048',
        'factory' => fn() => [ZTensor::ones([2048, 2048]), ZTensor::linspace(-1, 1, 2048)],
        'op' => fn(array $x) => ZTensor::clip($x[0]->dot($x[1]), -10, 10)->softmax(),
        'validate' => fn($gpu, $cpu) => sameTree($gpu->toArray(), $cpu->toArray(), 0.01, 1.0e-4),
    ],
    'chain_tile_sqrt_sum' => [
        'size' => '1024x1024 x2',
        'factory' => fn() => [ZTensor::linspace(0, 4, 1048576)->reshape([1024, 1024])],
        'op' => fn(array $x) => ZTensor::tile($x[0], 2)->sqrt()->sum(),
        'validate' => fn($gpu, $cpu) => sameTree($gpu->toArray(), $cpu->toArray(), 2.0, 5.0e-5),
    ],
];

$results = [];
foreach ($cases as $name => $case) {
    $base = $case['factory']();
    $cpuReference = $case['op']($base);
    $resident = array_map('residentCopy', $base);
    $gpuReference = $case['op']($resident);
    if (!$case['validate']($gpuReference, $cpuReference)) {
        $detail = is_scalar($gpuReference) && is_scalar($cpuReference)
            ? " (GPU $gpuReference, CPU $cpuReference, abs " . abs((float) $gpuReference - (float) $cpuReference) . ')'
            : '';
        throw new RuntimeException("$name validation failed$detail");
    }

    $samples = ['cpu' => [], 'h2d' => [], 'gpu' => [], 'd2h' => [], 'end_to_end' => []];
    for ($iteration = -WARMUPS; $iteration < REPETITIONS; ++$iteration) {
        [$cpuTime] = milliseconds(fn() => $case['op']($base));
        $uploadArguments = array_map(fn(ZTensor $x) => ZTensor::arr($x), $base);
        [$h2dTime, $uploaded] = milliseconds(function () use ($uploadArguments) {
            foreach ($uploadArguments as $argument) $argument->toGpu();
            return $uploadArguments;
        });
        [$gpuTime, $gpuResult] = milliseconds(fn() => $case['op']($resident));
        if ($gpuResult instanceof ZTensor) {
            [$d2hTime] = milliseconds(fn() => $gpuResult->toCpu());
        } else {
            $d2hTime = 0.0; // dot() returns a PHP scalar; its mandatory scalar D2H is included in gpuTime.
        }
        $endToEndArguments = array_map(fn(ZTensor $x) => ZTensor::arr($x), $base);
        [$endToEndTime] = milliseconds(function () use ($case, $endToEndArguments) {
            $arguments = $endToEndArguments;
            foreach ($arguments as $argument) $argument->toGpu();
            $result = $case['op']($arguments);
            if ($result instanceof ZTensor) $result->toCpu();
            return $result;
        });
        unset($uploaded, $gpuResult);
        if ($iteration >= 0) {
            $samples['cpu'][] = $cpuTime; $samples['h2d'][] = $h2dTime;
            $samples['gpu'][] = $gpuTime; $samples['d2h'][] = $d2hTime;
            $samples['end_to_end'][] = $endToEndTime;
        }
    }
    $stats = array_map('summarize', $samples);
    $stats['speedup_resident'] = $stats['cpu']['median_ms'] / $stats['gpu']['median_ms'];
    $stats['speedup_end_to_end'] = $stats['cpu']['median_ms'] / $stats['end_to_end']['median_ms'];
    $results[$name] = ['size' => $case['size'], 'validated' => true, 'stats' => $stats];
    unset($base, $resident, $cpuReference, $gpuReference);
    gc_collect_cycles();
    echo "completed $name\n";
}

$commit = command('git -C ' . escapeshellarg(__DIR__ . '/../..') . ' rev-parse HEAD');
$binary = __DIR__ . '/../../modules/zmatrix.so';
$document = [
    'environment' => [
        'timestamp' => date(DATE_ATOM), 'warmups' => WARMUPS, 'repetitions' => REPETITIONS,
        'php' => PHP_VERSION, 'zmatrix' => phpversion('zmatrix'), 'cuda' => command('nvcc --version | tail -n 1'),
        'driver' => command('nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1'),
        'gpu' => command('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1'),
        'commit' => $commit, 'binary_sha256' => is_file($binary) ? hash_file('sha256', $binary) : null,
        'php_memory_bytes' => memory_get_usage(true), 'php_peak_bytes' => memory_get_peak_usage(true), 'rss_bytes' => rssBytes(),
    ],
    'results' => $results,
];
$suffix = date('Ymd_His');
$jsonPath = __DIR__ . "/results/extended_ops_$suffix.json";
$csvPath = __DIR__ . "/results/extended_ops_$suffix.csv";
file_put_contents($jsonPath, json_encode($document, JSON_PRETTY_PRINT | JSON_PRESERVE_ZERO_FRACTION | JSON_THROW_ON_ERROR) . "\n");
$csv = fopen($csvPath, 'wb');
fputcsv($csv, ['operation', 'size', 'phase', 'median_ms', 'p25_ms', 'p75_ms', 'validated']);
foreach ($results as $operation => $result) {
    foreach ($result['stats'] as $phase => $stats) {
        if (!is_array($stats) || !isset($stats['median_ms'])) continue;
        fputcsv($csv, [$operation, $result['size'], $phase, $stats['median_ms'], $stats['p25_ms'], $stats['p75_ms'], 'true']);
    }
}
fclose($csv);
echo "$jsonPath\n$csvPath\n";
