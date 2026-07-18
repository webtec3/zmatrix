<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

if (!extension_loaded('zmatrix')) {
    fwrite(STDERR, "ZMatrix extension is not loaded.\n");
    exit(1);
}

$size = (int) ($argv[1] ?? 512);
$repetitions = (int) ($argv[2] ?? 15);
$warmups = 3;
if ($size < 1 || $repetitions < 3) throw new InvalidArgumentException('size >= 1 and repetitions >= 3 are required');

function deterministicMatrix(int $rows, int $cols, int $salt): array
{
    $matrix = [];
    for ($i = 0; $i < $rows; ++$i) {
        $row = [];
        for ($j = 0; $j < $cols; ++$j) {
            $row[] = (float) (((($i * 131 + $j * 17 + $salt) % 257) - 128) / 64.0);
        }
        $matrix[] = $row;
    }
    return $matrix;
}

function timed(callable $operation): float
{
    $start = hrtime(true);
    $operation();
    return (hrtime(true) - $start) / 1_000_000.0;
}

function stats(array $values): array
{
    sort($values);
    $count = count($values);
    $mean = array_sum($values) / $count;
    $variance = 0.0;
    foreach ($values as $value) $variance += ($value - $mean) ** 2;
    return [
        'min_ms' => $values[0],
        'median_ms' => $values[intdiv($count, 2)],
        'mean_ms' => $mean,
        'stddev_ms' => sqrt($variance / $count),
        'max_ms' => $values[$count - 1],
    ];
}

function assertClose(array $actual, array $expected, float $atol = 2.0e-4, float $rtol = 2.0e-4): void
{
    foreach ($expected as $i => $row) {
        foreach ($row as $j => $value) {
            $difference = abs($actual[$i][$j] - $value);
            if ($difference > $atol + $rtol * abs($value)) {
                throw new RuntimeException("validation failed at [{$i},{$j}]");
            }
        }
    }
}

function residentGpuMemoryMiB(): ?float
{
    $pid = getmypid();
    $lines = [];
    exec('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null', $lines);
    foreach ($lines as $line) {
        $parts = array_map('trim', explode(',', $line));
        if ((int) ($parts[0] ?? 0) === $pid) return (float) ($parts[1] ?? 0.0);
    }
    return null;
}

function rssMiB(): float
{
    $status = file_get_contents('/proc/self/status') ?: '';
    preg_match('/^VmRSS:\s+(\d+)\s+kB$/m', $status, $matches);
    return ((int) ($matches[1] ?? 0)) / 1024.0;
}

$aData = deterministicMatrix($size, $size, 11);
$bData = deterministicMatrix($size, $size, 97);
$cpuA = ZTensor::arr($aData);
$cpuB = ZTensor::arr($bData);

// Context/cuBLAS/kernel warm-up is intentionally excluded from stable timings.
for ($i = 0; $i < $warmups; ++$i) {
    $warmA = ZTensor::arr([[1, 2], [3, 4]])->toGpu();
    $warmB = ZTensor::arr([[5, 6], [7, 8]])->toGpu();
    $warmA->matmul($warmB)->toCpu();
}

$cpuTimes = [];
for ($i = 0; $i < $repetitions; ++$i) {
    $cpuTimes[] = timed(static function () use ($cpuA, $cpuB, &$cpuResult): void {
        $cpuResult = $cpuA->matmul($cpuB);
    });
}

$h2dTimes = [];
for ($i = 0; $i < $repetitions; ++$i) {
    $uploadA = ZTensor::arr($aData);
    $uploadB = ZTensor::arr($bData);
    $h2dTimes[] = timed(static function () use ($uploadA, $uploadB): void {
        $uploadA->toGpu();
        $uploadB->toGpu();
    });
}

$gpuA = ZTensor::arr($aData)->toGpu();
$gpuB = ZTensor::arr($bData)->toGpu();
if (!$gpuA->isOnGpu() || !$gpuB->isOnGpu()) throw new RuntimeException('toGpu() did not establish residency');

$kernelTimes = [];
$d2hTimes = [];
for ($i = 0; $i < $repetitions; ++$i) {
    $kernelTimes[] = timed(static function () use ($gpuA, $gpuB, &$gpuResult): void {
        $gpuResult = $gpuA->matmul($gpuB);
    });
    if (!$gpuResult->isOnGpu()) throw new RuntimeException('SGEMM result is not device-resident');
    $d2hTimes[] = timed(static function () use ($gpuResult): void {
        $gpuResult->toCpu();
    });
}

$endToEndTimes = [];
for ($i = 0; $i < $repetitions; ++$i) {
    $endA = ZTensor::arr($aData);
    $endB = ZTensor::arr($bData);
    $endToEndTimes[] = timed(static function () use ($endA, $endB, &$endResult): void {
        $endA->toGpu();
        $endB->toGpu();
        $endResult = $endA->matmul($endB);
        $endResult->toCpu();
    });
}

$validationGpu = $gpuA->matmul($gpuB);
assertClose($validationGpu->toArray(), $cpuResult->toArray());

$chainTimes = [];
for ($i = 0; $i < $repetitions; ++$i) {
    $chain = ZTensor::arr($aData)->toGpu();
    $chainOther = ZTensor::arr($bData)->toGpu();
    $chainTimes[] = timed(static function () use ($chain, $chainOther): void {
        $chain->add($chainOther)->relu()->mul(0.5)->sub(0.25)->tanh();
    });
    if (!$chain->isOnGpu()) throw new RuntimeException('GPU chain lost residency');
}

$report = [
    'methodology' => [
        'shape' => [$size, $size],
        'repetitions' => $repetitions,
        'warmups' => $warmups,
        'validation' => 'absolute + relative tolerance, atol=2e-4, rtol=2e-4',
        'kernel_contract' => 'backend synchronizes before returning, so kernel timing includes real completion',
    ],
    'software' => [
        'php' => PHP_VERSION,
        'extension' => phpversion('zmatrix'),
        'os' => php_uname(),
        'nvidia_smi' => trim((string) shell_exec('nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null')),
        'nvcc' => trim((string) shell_exec('nvcc --version 2>/dev/null | tail -1')),
    ],
    'timings' => [
        'cpu_sgemm' => stats($cpuTimes),
        'h2d_two_inputs' => stats($h2dTimes),
        'gpu_sgemm_resident' => stats($kernelTimes),
        'd2h_output' => stats($d2hTimes),
        'gpu_end_to_end' => stats($endToEndTimes),
        'gpu_resident_elementwise_chain' => stats($chainTimes),
    ],
    'memory' => [
        'php_usage_mib' => memory_get_usage(true) / 1048576.0,
        'php_peak_mib' => memory_get_peak_usage(true) / 1048576.0,
        'process_rss_mib' => rssMiB(),
        'gpu_process_mib' => residentGpuMemoryMiB(),
        'note' => 'PHP memory does not include every native/CUDA allocation; RSS and per-process GPU memory are reported separately.',
    ],
    'validated' => true,
];

echo json_encode($report, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES), PHP_EOL;
