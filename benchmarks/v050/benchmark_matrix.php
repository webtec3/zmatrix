<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

if (!extension_loaded('zmatrix')) {
    fwrite(STDERR, "ZMatrix extension is not loaded.\n");
    exit(1);
}

function option(string $name, string $default): string
{
    foreach (array_slice($GLOBALS['argv'], 1) as $argument) {
        if (str_starts_with($argument, "--{$name}=")) return substr($argument, strlen($name) + 3);
    }
    return $default;
}

function deterministicTensor(array $shape, int $variant = 0): ZTensor
{
    $count = array_product($shape);
    $tensor = ZTensor::arange(0.0, (float) $count, 1.0)->reshape($shape);
    $tensor->mul(1.0 / max(1, $count));
    return $variant === 0 ? $tensor->sub(0.5) : $tensor->mul(-0.75)->add(0.125);
}

function elapsed(callable $operation): float
{
    $start = hrtime(true);
    $operation();
    return (hrtime(true) - $start) / 1_000_000.0;
}

function percentile(array $sorted, float $fraction): float
{
    $position = (count($sorted) - 1) * $fraction;
    $lower = (int) floor($position);
    $upper = (int) ceil($position);
    if ($lower === $upper) return $sorted[$lower];
    return $sorted[$lower] + ($sorted[$upper] - $sorted[$lower]) * ($position - $lower);
}

function statistics(array $values): array
{
    sort($values, SORT_NUMERIC);
    $mean = array_sum($values) / count($values);
    $variance = array_sum(array_map(static fn(float $v): float => ($v - $mean) ** 2, $values)) / count($values);
    return [
        'samples' => count($values),
        'min_ms' => $values[0],
        'p05_ms' => percentile($values, 0.05),
        'p25_ms' => percentile($values, 0.25),
        'median_ms' => percentile($values, 0.50),
        'p75_ms' => percentile($values, 0.75),
        'p95_ms' => percentile($values, 0.95),
        'p99_ms' => percentile($values, 0.99),
        'mean_ms' => $mean,
        'stddev_ms' => sqrt($variance),
        'max_ms' => $values[array_key_last($values)],
        'raw_ms' => $values,
    ];
}

function float32UlpDistance(float $a, float $b): ?int
{
    if (!is_finite($a) || !is_finite($b)) return null;
    $aBits = (int) unpack('V', pack('g', $a))[1];
    $bBits = (int) unpack('V', pack('g', $b))[1];
    $aOrdered = ($aBits & 0x80000000) ? 0x80000000 - ($aBits & 0x7fffffff) : 0x80000000 + $aBits;
    $bOrdered = ($bBits & 0x80000000) ? 0x80000000 - ($bBits & 0x7fffffff) : 0x80000000 + $bBits;
    return abs($aOrdered - $bOrdered);
}

function validateTensor(ZTensor $actual, ZTensor $expected, float $atol, float $rtol): array
{
    $actualHost = ZTensor::arr($actual)->toCpu();
    $expectedHost = ZTensor::arr($expected)->toCpu();
    $absolute = ZTensor::arr($actualHost)->sub($expectedHost)->abs()->max();
    $scale = ZTensor::arr($expectedHost)->abs()->max();
    $relative = $absolute / max($scale, 1.17549435e-38);
    return [
        'valid' => $absolute <= $atol + $rtol * $scale,
        'max_abs_error' => $absolute,
        'max_relative_error' => $relative,
        'max_ulp_error' => null,
        'atol' => $atol,
        'rtol' => $rtol,
    ];
}

function validateScalar(float|int $actual, float|int $expected, float $atol, float $rtol, bool $exact = false): array
{
    $absolute = abs((float) $actual - (float) $expected);
    $relative = $absolute / max(abs((float) $expected), 1.17549435e-38);
    return [
        'valid' => $exact ? $actual === $expected : $absolute <= $atol + $rtol * abs((float) $expected),
        'max_abs_error' => $absolute,
        'max_relative_error' => $relative,
        'max_ulp_error' => float32UlpDistance((float) $actual, (float) $expected),
        'atol' => $atol,
        'rtol' => $rtol,
    ];
}

function processRssMiB(): float
{
    $status = file_get_contents('/proc/self/status') ?: '';
    preg_match('/^VmRSS:\s+(\d+)\s+kB$/m', $status, $matches);
    return ((int) ($matches[1] ?? 0)) / 1024.0;
}

function gpuMemoryMiB(): ?float
{
    $pid = getmypid();
    $lines = [];
    exec('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null', $lines);
    foreach ($lines as $line) {
        [$candidate, $memory] = array_pad(array_map('trim', explode(',', $line)), 2, '');
        if ((int) $candidate === $pid && is_numeric($memory)) return (float) $memory;
    }
    return null;
}

function record(string $category, string $operation, string $case, array $shape, ?int $axis, int $chainLength,
                string $scenario, array $times, array $validation): array
{
    return [
        'category' => $category,
        'operation' => $operation,
        'case' => $case,
        'shape' => $shape,
        'axis' => $axis,
        'chain_length' => $chainLength,
        'scenario' => $scenario,
        'timing' => statistics($times),
        'validation' => $validation,
        'memory' => [
            'php_mib' => memory_get_usage(true) / 1048576.0,
            'php_peak_mib' => memory_get_peak_usage(true) / 1048576.0,
            'rss_mib' => processRssMiB(),
            'gpu_process_mib' => gpuMemoryMiB(),
        ],
    ];
}

function warmupCuda(int $count): void
{
    for ($i = 0; $i < $count; ++$i) {
        $a = ZTensor::ones([32, 32])->toGpu();
        $b = ZTensor::ones([32, 32])->toGpu();
        $a->matmul($b)->add(1.0)->sumtotal();
    }
}

function benchmarkSgemm(int $m, int $k, int $n, int $repetitions): array
{
    $case = "{$m}x{$k}_{$k}x{$n}";
    $a = deterministicTensor([$m, $k], 0);
    $b = deterministicTensor([$k, $n], 1);
    $cpuTimes = [];
    for ($i = 0; $i < $repetitions; ++$i) {
        $cpuTimes[] = elapsed(static function () use ($a, $b, &$cpuResult): void { $cpuResult = $a->matmul($b); });
    }

    $h2dTimes = [];
    for ($i = 0; $i < $repetitions; ++$i) {
        $ga = ZTensor::arr($a);
        $gb = ZTensor::arr($b);
        $h2dTimes[] = elapsed(static function () use ($ga, $gb): void { $ga->toGpu(); $gb->toGpu(); });
    }

    $ga = ZTensor::arr($a)->toGpu();
    $gb = ZTensor::arr($b)->toGpu();
    $kernelTimes = [];
    $d2hTimes = [];
    for ($i = 0; $i < $repetitions; ++$i) {
        $kernelTimes[] = elapsed(static function () use ($ga, $gb, &$gpuResult): void { $gpuResult = $ga->matmul($gb); });
        if (!$gpuResult->isOnGpu()) throw new RuntimeException("{$case}: SGEMM result lost device residency");
        $d2hTimes[] = elapsed(static function () use ($gpuResult): void { $gpuResult->toCpu(); });
    }

    $endTimes = [];
    for ($i = 0; $i < $repetitions; ++$i) {
        $ea = ZTensor::arr($a);
        $eb = ZTensor::arr($b);
        $endTimes[] = elapsed(static function () use ($ea, $eb, &$endResult): void {
            $endResult = $ea->toGpu()->matmul($eb->toGpu())->toCpu();
        });
    }

    $gpuValidation = $ga->matmul($gb);
    $unitRoundoff = 5.960464477539063e-8;
    $gamma = ($k * $unitRoundoff < 0.5) ? ($k * $unitRoundoff) / (1.0 - $k * $unitRoundoff) : 1.0e-3;
    $validation = validateTensor($gpuValidation, $cpuResult, 2.0e-5 * sqrt($k), max(2.0e-4, 2.0 * $gamma));
    if (!$validation['valid']) throw new RuntimeException("{$case}: SGEMM CPU/GPU validation failed");

    $shape = [$m, $k, $n];
    return [
        record('sgemm', 'matmul', $case, $shape, null, 1, 'cpu', $cpuTimes, $validation),
        record('sgemm', 'matmul', $case, $shape, null, 1, 'h2d', $h2dTimes, $validation),
        record('sgemm', 'matmul', $case, $shape, null, 1, 'gpu_resident', $kernelTimes, $validation),
        record('sgemm', 'matmul', $case, $shape, null, 1, 'd2h', $d2hTimes, $validation),
        record('sgemm', 'matmul', $case, $shape, null, 1, 'gpu_end_to_end', $endTimes, $validation),
    ];
}

function applyChain(ZTensor $tensor, int $length): ZTensor
{
    $operations = [
        static fn(ZTensor $t): ZTensor => $t->add(0.125),
        static fn(ZTensor $t): ZTensor => $t->relu(),
        static fn(ZTensor $t): ZTensor => $t->mul(0.75),
        static fn(ZTensor $t): ZTensor => $t->sub(0.05),
        static fn(ZTensor $t): ZTensor => $t->tanh(),
        static fn(ZTensor $t): ZTensor => $t->add(0.5),
        static fn(ZTensor $t): ZTensor => $t->sigmoid(),
        static fn(ZTensor $t): ZTensor => $t->mul(1.25),
        static fn(ZTensor $t): ZTensor => $t->abs(),
        static fn(ZTensor $t): ZTensor => $t->sub(0.01),
    ];
    for ($i = 0; $i < $length; ++$i) $operations[$i % count($operations)]($tensor);
    return $tensor;
}

function benchmarkChain(int $size, int $length, int $repetitions): array
{
    $base = deterministicTensor([$size, $size]);
    $cpuTimes = [];
    for ($i = 0; $i < $repetitions; ++$i) {
        $candidate = ZTensor::arr($base);
        $cpuTimes[] = elapsed(static function () use ($candidate, $length, &$cpuResult): void {
            $cpuResult = applyChain($candidate, $length);
        });
    }
    $residentTimes = [];
    for ($i = 0; $i < $repetitions; ++$i) {
        $candidate = ZTensor::arr($base)->toGpu();
        $residentTimes[] = elapsed(static function () use ($candidate, $length, &$gpuResult): void {
            $gpuResult = applyChain($candidate, $length);
        });
        if (!$gpuResult->isOnGpu()) throw new RuntimeException('elementwise chain lost device residency');
    }
    $endTimes = [];
    for ($i = 0; $i < $repetitions; ++$i) {
        $candidate = ZTensor::arr($base);
        $endTimes[] = elapsed(static function () use ($candidate, $length, &$endResult): void {
            $endResult = applyChain($candidate->toGpu(), $length)->toCpu();
        });
    }
    $validation = validateTensor($gpuResult, $cpuResult, 2.0e-5, 2.0e-5);
    if (!$validation['valid']) throw new RuntimeException("{$size}^2 chain {$length}: CPU/GPU validation failed");
    $case = "{$size}x{$size}_chain_{$length}";
    return [
        record('elementwise_chain', 'mixed', $case, [$size, $size], null, $length, 'cpu', $cpuTimes, $validation),
        record('elementwise_chain', 'mixed', $case, [$size, $size], null, $length, 'gpu_resident', $residentTimes, $validation),
        record('elementwise_chain', 'mixed', $case, [$size, $size], null, $length, 'gpu_end_to_end', $endTimes, $validation),
    ];
}

function invokeReduction(ZTensor $tensor, string $operation, ?int $axis): float|int|ZTensor
{
    return match ($operation) {
        'sum' => $axis === null ? $tensor->sumtotal() : $tensor->sum($axis),
        'min' => $axis === null ? $tensor->min() : throw new InvalidArgumentException('axis min is not public'),
        'max' => $axis === null ? $tensor->max() : throw new InvalidArgumentException('axis max is not public'),
        'argmin' => $tensor->argmin($axis),
        'argmax' => $tensor->argmax($axis),
        default => throw new InvalidArgumentException("unknown reduction {$operation}"),
    };
}

function benchmarkReduction(array $shape, string $operation, ?int $axis, int $repetitions): array
{
    // Positive, non-symmetric data avoids making the primary performance
    // corpus an ill-conditioned cancellation test. Cancellation is covered
    // separately in the reduction accuracy study.
    $base = deterministicTensor($shape)->add(0.51);
    $cpuTimes = [];
    for ($i = 0; $i < $repetitions; ++$i) {
        $candidate = ZTensor::arr($base);
        $cpuTimes[] = elapsed(static function () use ($candidate, $operation, $axis, &$cpuResult): void {
            $cpuResult = invokeReduction($candidate, $operation, $axis);
        });
    }
    $gpuTimes = [];
    for ($i = 0; $i < $repetitions; ++$i) {
        $candidate = ZTensor::arr($base)->toGpu();
        $gpuTimes[] = elapsed(static function () use ($candidate, $operation, $axis, &$gpuResult): void {
            $gpuResult = invokeReduction($candidate, $operation, $axis);
        });
    }
    $isScalar = !($gpuResult instanceof ZTensor);
    $exact = str_starts_with($operation, 'arg');
    $validation = $isScalar
        ? validateScalar($gpuResult, $cpuResult, 5.0e-5, 1.0e-3, $exact)
        : validateTensor($gpuResult, $cpuResult, $exact ? 0.0 : 5.0e-5, $exact ? 0.0 : 1.0e-3);
    // Experimental variants must preserve failed comparisons as evidence
    // instead of hiding them by widening tolerances. The production/default
    // hybrid remains fail-fast.
    if (!$validation['valid'] && getenv('ZMATRIX_REDUCTION_IMPL') === false) {
        throw new RuntimeException("{$operation}: reduction CPU/GPU validation failed: " . json_encode([
            'actual' => $isScalar ? $gpuResult : 'tensor',
            'expected' => $isScalar ? $cpuResult : 'tensor',
            'validation' => $validation,
        ], JSON_THROW_ON_ERROR));
    }
    $case = implode('x', $shape) . '_' . $operation . '_axis_' . ($axis ?? 'all');
    return [
        record('reduction', $operation, $case, $shape, $axis, 1, 'cpu', $cpuTimes, $validation),
        record('reduction', $operation, $case, $shape, $axis, 1,
            'gpu_' . (getenv('ZMATRIX_REDUCTION_IMPL') ?: 'hybrid'), $gpuTimes, $validation),
    ];
}

function writeCsv(string $path, array $records): void
{
    $handle = fopen($path, 'wb');
    if (!$handle) throw new RuntimeException("cannot create {$path}");
    $header = ['category', 'operation', 'case', 'shape', 'axis', 'chain_length', 'scenario', 'samples', 'min_ms',
        'p05_ms', 'p25_ms', 'median_ms', 'p75_ms', 'p95_ms', 'p99_ms', 'mean_ms', 'stddev_ms', 'max_ms',
        'valid', 'max_abs_error', 'max_relative_error', 'max_ulp_error', 'atol', 'rtol', 'php_mib', 'php_peak_mib',
        'rss_mib', 'gpu_process_mib', 'raw_ms'];
    fputcsv($handle, $header, ',', '"', '');
    foreach ($records as $record) {
        $row = [$record['category'], $record['operation'], $record['case'], implode('x', $record['shape']),
            $record['axis'], $record['chain_length'], $record['scenario']];
        foreach (['samples', 'min_ms', 'p05_ms', 'p25_ms', 'median_ms', 'p75_ms', 'p95_ms', 'p99_ms', 'mean_ms', 'stddev_ms', 'max_ms'] as $key) $row[] = $record['timing'][$key];
        foreach (['valid', 'max_abs_error', 'max_relative_error', 'max_ulp_error', 'atol', 'rtol'] as $key) $row[] = $record['validation'][$key];
        foreach (['php_mib', 'php_peak_mib', 'rss_mib', 'gpu_process_mib'] as $key) $row[] = $record['memory'][$key];
        $row[] = json_encode($record['timing']['raw_ms'], JSON_THROW_ON_ERROR);
        fputcsv($handle, $row, ',', '"', '');
    }
    fclose($handle);
}

$suite = option('suite', 'quick');
$categories = array_filter(explode(',', option('categories', 'sgemm,chain,reduction')));
$repetitions = (int) option('repetitions', $suite === 'full' ? '9' : '5');
$warmups = (int) option('warmups', '3');
$outputDirectory = option('output', __DIR__ . '/results');
if (!in_array($suite, ['quick', 'full'], true) || $repetitions < 3 || $warmups < 1
    || array_diff($categories, ['sgemm', 'chain', 'reduction'])) {
    throw new InvalidArgumentException('suite must be quick|full, repetitions >= 3 and warmups >= 1');
}
if (!is_dir($outputDirectory) && !mkdir($outputDirectory, 0775, true) && !is_dir($outputDirectory)) {
    throw new RuntimeException("cannot create {$outputDirectory}");
}

$sgemmCases = $suite === 'full'
    ? [[512,512,512], [1024,1024,1024], [2048,2048,2048], [4096,4096,4096],
       [256,4096,128], [4096,256,4096], [4096,128,256], [128,4096,256],
       [1,4096,256], [4096,1,1], [1,4096,1]]
    : [[256,256,256], [512,512,512], [256,1024,128], [1024,128,512]];
$chainSizes = $suite === 'full' ? [256, 512, 1024, 2048] : [256, 512];
$chainLengths = [1, 2, 5, 10];
$reductionCases = $suite === 'full'
    ? [[[1024,1024], null], [[2048,2048], null], [[4096,256], 0], [[256,4096], 1]]
    : [[[512,512], null], [[256,1024], 1]];

warmupCuda($warmups);
$records = [];
if (in_array('sgemm', $categories, true)) {
    foreach ($sgemmCases as [$m, $k, $n]) array_push($records, ...benchmarkSgemm($m, $k, $n, $repetitions));
}
if (in_array('chain', $categories, true)) {
    foreach ($chainSizes as $size) foreach ($chainLengths as $length) array_push($records, ...benchmarkChain($size, $length, $repetitions));
}
if (in_array('reduction', $categories, true)) {
    foreach ($reductionCases as [$shape, $axis]) {
        $operations = $axis === null ? ['sum', 'min', 'max', 'argmin', 'argmax'] : ['sum', 'argmin', 'argmax'];
        foreach ($operations as $operation) array_push($records, ...benchmarkReduction($shape, $operation, $axis, $repetitions));
    }
}

$stamp = gmdate('Ymd_His');
$invalidRecords = count(array_filter($records, static fn(array $record): bool => !$record['validation']['valid']));
$metadata = [
    'schema_version' => 1,
    'created_utc' => gmdate(DATE_ATOM),
    'suite' => $suite,
    'categories' => array_values($categories),
    'all_records_valid' => $invalidRecords === 0,
    'invalid_records' => $invalidRecords,
    'reduction_implementation' => getenv('ZMATRIX_REDUCTION_IMPL') ?: 'hybrid',
    'methodology' => [
        'warmups' => $warmups,
        'repetitions' => $repetitions,
        'primary_statistic' => 'median_ms',
        'validation' => 'Every timed case is compared with the CPU result; SGEMM tolerance derives from float32 gamma_k.',
        'gpu_contract' => 'GPU execution occurs only after explicit toGpu().',
    ],
    'environment' => [
        'php' => PHP_VERSION,
        'zmatrix' => phpversion('zmatrix'),
        'os' => php_uname(),
        'cpu' => trim((string) shell_exec("lscpu | sed -n 's/^Model name:[[:space:]]*//p'")),
        'gpu' => trim((string) shell_exec('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null')),
        'nvcc' => trim((string) shell_exec('nvcc --version 2>/dev/null | tail -1')),
        'nsys' => trim((string) shell_exec('nsys --version 2>/dev/null')),
        'ncu' => trim((string) shell_exec('ncu --version 2>/dev/null | tail -1')),
    ],
    'records' => $records,
];
$strategy = getenv('ZMATRIX_REDUCTION_IMPL') ?: 'hybrid';
$jsonPath = "{$outputDirectory}/baseline_{$suite}_{$strategy}_{$stamp}.json";
$csvPath = "{$outputDirectory}/baseline_{$suite}_{$strategy}_{$stamp}.csv";
file_put_contents($jsonPath, json_encode($metadata, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR) . PHP_EOL);
writeCsv($csvPath, $records);
echo json_encode(['validated' => $invalidRecords === 0, 'invalid_records' => $invalidRecords,
    'records' => count($records), 'json' => $jsonPath, 'csv' => $csvPath], JSON_PRETTY_PRINT), PHP_EOL;
