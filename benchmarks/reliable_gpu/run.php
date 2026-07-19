<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

const BENCHMARK_MIN_WARMUPS = 10;
const BENCHMARK_MIN_ITERATIONS = 50;
const BENCHMARK_MAD_Z_LIMIT = 3.5;

if (!extension_loaded('zmatrix') || !class_exists(ZTensor::class, false)) {
    fwrite(STDERR, "The zmatrix extension must be loaded.\n");
    exit(1);
}

$options = getopt('', [
    'sizes::',
    'warmups::',
    'iterations::',
    'json::',
    'csv::',
]);

$warmups = parseMinimumInt($options['warmups'] ?? BENCHMARK_MIN_WARMUPS, BENCHMARK_MIN_WARMUPS, 'warmups');
$iterations = parseMinimumInt($options['iterations'] ?? BENCHMARK_MIN_ITERATIONS, BENCHMARK_MIN_ITERATIONS, 'iterations');
$sizes = parseSizes($options['sizes'] ?? '256,512,1024,2048,4096');
$timestamp = gmdate('Ymd_His');
$resultsDirectory = __DIR__ . '/results';
$jsonPath = (string) ($options['json'] ?? "{$resultsDirectory}/benchmark_{$timestamp}.json");
$csvPath = (string) ($options['csv'] ?? "{$resultsDirectory}/benchmark_{$timestamp}.csv");

foreach ([
    'OMP_NUM_THREADS' => '1',
    'OMP_DYNAMIC' => 'FALSE',
    'OMP_WAIT_POLICY' => 'PASSIVE',
    'OPENBLAS_NUM_THREADS' => '1',
] as $name => $value) {
    putenv("{$name}={$value}");
}

if (gc_enabled()) {
    gc_collect_cycles();
    gc_disable();
}

$operations = [
    ['name' => 'MatMul', 'binary' => true, 'gpu' => true, 'comparable' => true],
    ['name' => 'Add', 'binary' => true, 'gpu' => true, 'comparable' => true],
    ['name' => 'Mul', 'binary' => true, 'gpu' => true, 'comparable' => true],
    ['name' => 'ReLU', 'binary' => false, 'gpu' => true, 'comparable' => true],
    ['name' => 'Exp', 'binary' => false, 'gpu' => true, 'comparable' => true],
    ['name' => 'Sqrt', 'binary' => false, 'gpu' => true, 'comparable' => true],
    ['name' => 'Abs', 'binary' => false, 'gpu' => true, 'comparable' => true],
    ['name' => 'Sum', 'binary' => false, 'gpu' => true, 'comparable' => true],
    [
        'name' => 'TransposeView',
        'binary' => false,
        'gpu' => false,
        'comparable' => false,
        'note' => 'O(1) CPU view creation; result is consumed after timing',
    ],
    [
        'name' => 'TransposePhysicalProxy',
        'binary' => false,
        'gpu' => true,
        'comparable' => false,
        'note' => 'CPU: view + contiguous materialization + no-op clip; GPU: physical transpose kernel',
    ],
];

$report = [
    'schema_version' => 1,
    'generated_at_utc' => gmdate(DATE_ATOM),
    'extension_version' => phpversion('zmatrix'),
    'php_version' => PHP_VERSION,
    'timer' => 'hrtime(true) monotonic nanoseconds',
    'warmups' => $warmups,
    'iterations' => $iterations,
    'sizes' => $sizes,
    'outlier_filter' => [
        'primary' => 'MAD',
        'modified_z_limit' => BENCHMARK_MAD_Z_LIMIT,
        'fallback' => 'IQR 1.5x when MAD is zero',
    ],
    'thread_environment' => [
        'OMP_NUM_THREADS' => getenv('OMP_NUM_THREADS'),
        'OMP_DYNAMIC' => getenv('OMP_DYNAMIC'),
        'OMP_WAIT_POLICY' => getenv('OMP_WAIT_POLICY'),
        'OPENBLAS_NUM_THREADS' => getenv('OPENBLAS_NUM_THREADS'),
    ],
    'gpu_synchronization' => [
        'cuda_kernels' => 'CUDA_KERNEL_CHECK calls cudaPeekAtLastError and cudaDeviceSynchronize',
        'cublas_sgemm' => 'gpu_matmul_device calls cudaDeviceSynchronize after cublasSgemm',
        'api_contract' => 'ZMatrix PHP CUDA operations are synchronous',
    ],
    'scenario_contracts' => [
        'cpu_operation' => 'Input cloning excluded; operation and output allocation included.',
        'gpu_resident' => 'Inputs already device-resident; D2D preparation excluded. Kernel synchronization included. Output allocation is included for result-producing operations.',
        'gpu_end_to_end' => 'CPU inputs are prepared outside timing. H2D, device allocation, operation, synchronization and result D2H are included.',
        'sum_exception' => 'Sum necessarily includes reduction workspace and scalar D2H.',
        'context' => 'CUDA context creation and first-use compilation/handle initialization are excluded by setup and 10 warm-ups.',
    ],
    'results' => [],
];

printf(
    "ZMatrix reliable CPU/GPU benchmark: %d warm-ups, %d measured samples, monotonic timer, MAD/IQR filtering\n",
    $warmups,
    $iterations,
);
printf(
    "%-24s %6s %-15s %10s %10s %10s %10s %10s %8s\n",
    'operation',
    'size',
    'scenario',
    'median',
    'mean',
    'stddev',
    'min',
    'max',
    'outliers',
);

foreach ($sizes as $size) {
    $baseCpuA = ZTensor::full([$size, $size], 0.5);
    $baseCpuB = ZTensor::full([$size, $size], 0.25);

    $baseGpuA = $baseCpuA->copy();
    $baseGpuB = $baseCpuB->copy();
    $baseGpuA->toGpu();
    $baseGpuB->toGpu();

    foreach ($operations as $definition) {
        $name = $definition['name'];

        $cpuPrepare = static fn (): callable => prepareCpuOperation($name, $baseCpuA, $baseCpuB);
        $cpuValidationOperation = $cpuPrepare();
        $cpuValidationResult = $cpuValidationOperation();
        $cpuSignature = validationSignature($cpuValidationResult);

        $validation = [
            'cpu_checksum' => $cpuSignature['checksum'],
            'gpu_checksum' => null,
            'checksum_absolute_error' => null,
            'checksum_relative_error' => null,
            'sample_count' => count($cpuSignature['sample']),
            'sample_max_absolute_error' => null,
            'sample_max_relative_error' => null,
            'valid' => true,
        ];

        if ($definition['gpu']) {
            $gpuPrepare = static fn (): callable => prepareGpuResidentOperation($name, $baseGpuA, $baseGpuB);
            $gpuValidationOperation = $gpuPrepare();
            $gpuValidationResult = $gpuValidationOperation();
            $gpuSignature = validationSignature($gpuValidationResult);
            $validation = validateSignatures($cpuSignature, $gpuSignature, $name, $size);
        } else {
            $gpuPrepare = null;
        }

        $cpuMeasurement = measureScenario($cpuPrepare, $warmups, $iterations);
        appendResult(
            $report,
            $name,
            $size,
            'cpu_operation',
            $cpuMeasurement,
            $validation,
            $definition['note'] ?? null,
        );

        if ($definition['gpu']) {
            $gpuMeasurement = measureScenario($gpuPrepare, $warmups, $iterations);
            appendResult(
                $report,
                $name,
                $size,
                'gpu_resident',
                $gpuMeasurement,
                $validation,
                $definition['note'] ?? null,
            );
            $gpuResultIndex = array_key_last($report['results']);
            if ($definition['comparable']) {
                $cpuMedian = $cpuMeasurement['statistics']['median_ms'];
                $gpuMedian = $gpuMeasurement['statistics']['median_ms'];
                $report['results'][$gpuResultIndex]['resident_speedup_vs_cpu'] =
                    $gpuMedian > 0.0 ? $cpuMedian / $gpuMedian : null;
            }

            $endToEndPrepare = static fn (): callable => prepareGpuEndToEndOperation(
                $name,
                $baseCpuA,
                $baseCpuB,
                (bool) $definition['binary'],
            );
            $endToEndMeasurement = measureScenario($endToEndPrepare, $warmups, $iterations);
            appendResult(
                $report,
                $name,
                $size,
                'gpu_end_to_end',
                $endToEndMeasurement,
                $validation,
                $definition['note'] ?? null,
            );

        }

        unset(
            $cpuValidationResult,
            $gpuValidationResult,
            $cpuValidationOperation,
            $gpuValidationOperation,
        );
    }

    unset($baseCpuA, $baseCpuB, $baseGpuA, $baseGpuB);
    gc_collect_cycles();
}

if (!is_dir($resultsDirectory) && !mkdir($resultsDirectory, 0775, true) && !is_dir($resultsDirectory)) {
    throw new RuntimeException("Unable to create results directory: {$resultsDirectory}");
}
writeJson($jsonPath, $report);
writeCsv($csvPath, $report['results']);

printf("\nJSON: %s\nCSV:  %s\n", $jsonPath, $csvPath);

function parseMinimumInt(mixed $value, int $minimum, string $name): int
{
    $parsed = filter_var($value, FILTER_VALIDATE_INT);
    if ($parsed === false || $parsed < $minimum) {
        throw new InvalidArgumentException("{$name} must be an integer >= {$minimum}");
    }
    return $parsed;
}

function parseSizes(mixed $value): array
{
    $parts = array_filter(array_map('trim', explode(',', (string) $value)), 'strlen');
    if ($parts === []) {
        throw new InvalidArgumentException('sizes must contain at least one positive integer');
    }

    $sizes = [];
    foreach ($parts as $part) {
        $size = filter_var($part, FILTER_VALIDATE_INT);
        if ($size === false || $size <= 0) {
            throw new InvalidArgumentException("invalid matrix size: {$part}");
        }
        $sizes[] = $size;
    }
    return array_values(array_unique($sizes));
}

function executeOperation(string $name, ZTensor $a, ?ZTensor $b, bool $gpu): mixed
{
    return match ($name) {
        'MatMul' => $a->matmul($b),
        'Add' => $a->add($b),
        'Mul' => $a->mul($b),
        'ReLU' => $a->relu(),
        'Exp' => $a->exp(),
        'Sqrt' => $a->sqrt(),
        'Abs' => $a->abs(),
        'Sum' => $a->sumtotal(),
        'TransposeView' => $a->transpose(),
        'TransposePhysicalProxy' => $gpu
            ? $a->transpose()
            : ZTensor::clip($a->transpose(), -INF, INF),
        default => throw new InvalidArgumentException("Unknown operation: {$name}"),
    };
}

function prepareCpuOperation(string $name, ZTensor $baseA, ZTensor $baseB): callable
{
    $a = $baseA->copy();
    $b = $baseB->copy();
    return static fn (): mixed => executeOperation($name, $a, $b, false);
}

function prepareGpuResidentOperation(string $name, ZTensor $baseA, ZTensor $baseB): callable
{
    // Device-to-device copies and their allocations happen before the timer.
    $a = $baseA->copy();
    $b = $baseB->copy();
    return static fn (): mixed => executeOperation($name, $a, $b, true);
}

function prepareGpuEndToEndOperation(
    string $name,
    ZTensor $baseCpuA,
    ZTensor $baseCpuB,
    bool $binary,
): callable {
    // Host cloning is setup. Device allocation and H2D begin inside the timer.
    $a = $baseCpuA->copy();
    $b = $baseCpuB->copy();

    return static function () use ($name, $a, $b, $binary): mixed {
        $a->toGpu();
        if ($binary) {
            $b->toGpu();
        }

        $result = executeOperation($name, $a, $b, true);
        if ($result instanceof ZTensor) {
            $result->toCpu();
        }
        return $result;
    };
}

function measureScenario(callable $prepare, int $warmups, int $iterations): array
{
    for ($i = 0; $i < $warmups; ++$i) {
        $operation = $prepare();
        $result = $operation();
        consumeResult($result);
        unset($operation, $result);
    }

    $samples = [];
    for ($i = 0; $i < $iterations; ++$i) {
        $operation = $prepare();

        $start = hrtime(true);
        $result = $operation();
        $end = hrtime(true);

        if (!is_int($start) || !is_int($end) || $end < $start) {
            throw new RuntimeException('Monotonic timer invariant violated');
        }

        $samples[] = ($end - $start) / 1_000_000.0;
        consumeResult($result);
        unset($operation, $result);
    }

    [$filtered, $filter] = filterOutliers($samples);
    return [
        'raw_samples_ms' => $samples,
        'filtered_samples_ms' => $filtered,
        'outlier_filter' => $filter,
        'statistics' => calculateStatistics($filtered),
    ];
}

function consumeResult(mixed $result): void
{
    static $blackhole = 0;
    $value = checksumResult($result);
    $blackhole ^= crc32(sprintf('%.9g', $value));
}

function checksumResult(mixed $result): float
{
    if ($result instanceof ZTensor) {
        return (float) $result->sumtotal();
    }
    if (is_int($result) || is_float($result)) {
        return (float) $result;
    }
    throw new RuntimeException('Benchmark operation returned an unsupported result type');
}

function validationSignature(mixed $result): array
{
    $checksum = checksumResult($result);

    if (!$result instanceof ZTensor) {
        return ['checksum' => $checksum, 'sample' => [(float) $result]];
    }

    $sample = $result;
    $shape = $sample->shape();
    foreach ($shape as $axis => $dimension) {
        if ($dimension === 0) {
            return ['checksum' => $checksum, 'sample' => []];
        }
        $limit = $axis === 0 ? min(2, $dimension) : min(8, $dimension);
        if ($limit < $dimension) {
            $sample = $sample->slice($axis, 0, $limit);
        }
    }

    $flat = [];
    flattenNumericTree($sample->toArray(), $flat);
    return ['checksum' => $checksum, 'sample' => $flat];
}

function flattenNumericTree(mixed $value, array &$flat): void
{
    if (is_array($value)) {
        foreach ($value as $child) {
            flattenNumericTree($child, $flat);
        }
        return;
    }
    if (!is_int($value) && !is_float($value)) {
        throw new RuntimeException('Validation sample contains a non-numeric value');
    }
    $flat[] = (float) $value;
}

function validateSignatures(array $cpu, array $gpu, string $operation, int $size): array
{
    if (count($cpu['sample']) !== count($gpu['sample'])) {
        throw new RuntimeException("{$operation} {$size}x{$size} validation sample shape mismatch");
    }

    $sampleMaxAbsolute = 0.0;
    $sampleMaxRelative = 0.0;
    foreach ($cpu['sample'] as $index => $cpuValue) {
        $gpuValue = $gpu['sample'][$index];
        $absolute = abs($cpuValue - $gpuValue);
        $relative = $absolute / max(abs($cpuValue), abs($gpuValue), 1.0);
        $sampleMaxAbsolute = max($sampleMaxAbsolute, $absolute);
        $sampleMaxRelative = max($sampleMaxRelative, $relative);

        if (!is_finite($cpuValue)
            || !is_finite($gpuValue)
            || ($absolute > 2.0e-5 && $relative > 2.0e-5)) {
            throw new RuntimeException(sprintf(
                '%s %dx%d sample[%d] validation failed: CPU %.9g, GPU %.9g, abs %.9g, rel %.9g',
                $operation,
                $size,
                $size,
                $index,
                $cpuValue,
                $gpuValue,
                $absolute,
                $relative,
            ));
        }
    }

    $checksumAbsolute = abs($cpu['checksum'] - $gpu['checksum']);
    $checksumRelative = $checksumAbsolute
        / max(abs($cpu['checksum']), abs($gpu['checksum']), 1.0);

    return [
        'cpu_checksum' => $cpu['checksum'],
        'gpu_checksum' => $gpu['checksum'],
        'checksum_absolute_error' => $checksumAbsolute,
        'checksum_relative_error' => $checksumRelative,
        'sample_count' => count($cpu['sample']),
        'sample_max_absolute_error' => $sampleMaxAbsolute,
        'sample_max_relative_error' => $sampleMaxRelative,
        'valid' => true,
        'note' => 'Element sample is the correctness gate; aggregate checksum error is diagnostic because float reduction order differs.',
    ];
}

function filterOutliers(array $samples): array
{
    $median = median($samples);
    $deviations = array_map(static fn (float $value): float => abs($value - $median), $samples);
    $mad = median($deviations);

    if ($mad > 0.0) {
        $scale = 1.4826 * $mad;
        $filtered = array_values(array_filter(
            $samples,
            static fn (float $value): bool => abs($value - $median) / $scale <= BENCHMARK_MAD_Z_LIMIT,
        ));
        $method = 'MAD';
        $lower = $median - BENCHMARK_MAD_Z_LIMIT * $scale;
        $upper = $median + BENCHMARK_MAD_Z_LIMIT * $scale;
    } else {
        $sorted = $samples;
        sort($sorted, SORT_NUMERIC);
        $q1 = percentileSorted($sorted, 0.25);
        $q3 = percentileSorted($sorted, 0.75);
        $iqr = $q3 - $q1;
        if ($iqr > 0.0) {
            $lower = $q1 - 1.5 * $iqr;
            $upper = $q3 + 1.5 * $iqr;
            $filtered = array_values(array_filter(
                $samples,
                static fn (float $value): bool => $value >= $lower && $value <= $upper,
            ));
            $method = 'IQR';
        } else {
            $filtered = $samples;
            $method = 'none';
            $lower = min($samples);
            $upper = max($samples);
        }
    }

    return [
        $filtered,
        [
            'method' => $method,
            'raw_count' => count($samples),
            'filtered_count' => count($filtered),
            'discarded_count' => count($samples) - count($filtered),
            'median_ms' => $median,
            'mad_ms' => $mad,
            'lower_bound_ms' => $lower,
            'upper_bound_ms' => $upper,
        ],
    ];
}

function calculateStatistics(array $samples): array
{
    if ($samples === []) {
        throw new RuntimeException('No benchmark samples remained after filtering');
    }

    $count = count($samples);
    $mean = array_sum($samples) / $count;
    $variance = 0.0;
    foreach ($samples as $sample) {
        $variance += ($sample - $mean) ** 2;
    }
    $variance /= max(1, $count - 1);

    return [
        'count' => $count,
        'mean_ms' => $mean,
        'median_ms' => median($samples),
        'stddev_ms' => sqrt($variance),
        'min_ms' => min($samples),
        'max_ms' => max($samples),
    ];
}

function median(array $values): float
{
    sort($values, SORT_NUMERIC);
    $count = count($values);
    $middle = intdiv($count, 2);
    return $count % 2 === 0
        ? ($values[$middle - 1] + $values[$middle]) / 2.0
        : $values[$middle];
}

function percentileSorted(array $sorted, float $fraction): float
{
    $position = (count($sorted) - 1) * $fraction;
    $lower = (int) floor($position);
    $upper = (int) ceil($position);
    if ($lower === $upper) {
        return $sorted[$lower];
    }
    $weight = $position - $lower;
    return $sorted[$lower] * (1.0 - $weight) + $sorted[$upper] * $weight;
}

function appendResult(
    array &$report,
    string $operation,
    int $size,
    string $scenario,
    array $measurement,
    array $validation,
    ?string $note,
): void {
    $entry = [
        'operation' => $operation,
        'shape' => [$size, $size],
        'scenario' => $scenario,
        'statistics' => $measurement['statistics'],
        'outlier_filter' => $measurement['outlier_filter'],
        'validation' => $validation,
        'raw_samples_ms' => $measurement['raw_samples_ms'],
        'filtered_samples_ms' => $measurement['filtered_samples_ms'],
    ];
    if ($note !== null) {
        $entry['note'] = $note;
    }
    $report['results'][] = $entry;

    $stats = $measurement['statistics'];
    printf(
        "%-24s %6d %-15s %10.4f %10.4f %10.4f %10.4f %10.4f %8d\n",
        $operation,
        $size,
        $scenario,
        $stats['median_ms'],
        $stats['mean_ms'],
        $stats['stddev_ms'],
        $stats['min_ms'],
        $stats['max_ms'],
        $measurement['outlier_filter']['discarded_count'],
    );
}

function writeJson(string $path, array $report): void
{
    $directory = dirname($path);
    if (!is_dir($directory) && !mkdir($directory, 0775, true) && !is_dir($directory)) {
        throw new RuntimeException("Unable to create JSON output directory: {$directory}");
    }
    $encoded = json_encode($report, JSON_PRETTY_PRINT | JSON_PRESERVE_ZERO_FRACTION | JSON_THROW_ON_ERROR);
    if (file_put_contents($path, $encoded . PHP_EOL) === false) {
        throw new RuntimeException("Unable to write JSON output: {$path}");
    }
}

function writeCsv(string $path, array $results): void
{
    $directory = dirname($path);
    if (!is_dir($directory) && !mkdir($directory, 0775, true) && !is_dir($directory)) {
        throw new RuntimeException("Unable to create CSV output directory: {$directory}");
    }
    $handle = fopen($path, 'wb');
    if ($handle === false) {
        throw new RuntimeException("Unable to write CSV output: {$path}");
    }

    fputcsv($handle, [
        'operation',
        'rows',
        'columns',
        'scenario',
        'count',
        'mean_ms',
        'median_ms',
        'stddev_ms',
        'min_ms',
        'max_ms',
        'outliers',
        'cpu_checksum',
        'gpu_checksum',
        'absolute_error',
        'relative_error',
        'valid',
        'note',
    ], ',', '"', '', "\n");

    foreach ($results as $result) {
        $stats = $result['statistics'];
        $validation = $result['validation'];
        fputcsv($handle, [
            $result['operation'],
            $result['shape'][0],
            $result['shape'][1],
            $result['scenario'],
            $stats['count'],
            $stats['mean_ms'],
            $stats['median_ms'],
            $stats['stddev_ms'],
            $stats['min_ms'],
            $stats['max_ms'],
            $result['outlier_filter']['discarded_count'],
            $validation['cpu_checksum'],
            $validation['gpu_checksum'],
            $validation['checksum_absolute_error'] ?? null,
            $validation['checksum_relative_error'] ?? null,
            $validation['valid'] ? 1 : 0,
            $result['note'] ?? '',
        ], ',', '"', '', "\n");
    }

    fclose($handle);
}
