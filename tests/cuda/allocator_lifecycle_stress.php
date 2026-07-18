<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

function stressRss(): int {
    $status = file_get_contents('/proc/self/status') ?: '';
    return preg_match('/VmRSS:\s+(\d+)\s+kB/', $status, $matches) ? (int) $matches[1] * 1024 : 0;
}

function stressClose(float $actual, float $expected, float $tolerance = 1.0e-4): void {
    if (abs($actual - $expected) > $tolerance) {
        throw new RuntimeException("expected $expected, got $actual");
    }
}

$base = ZTensor::linspace(-1, 1, 65536)->toGpu();
$ones = ZTensor::ones([65536])->toGpu();
$tile = ZTensor::ones([256, 256])->toGpu();
$matrix = ZTensor::ones([512, 512])->toGpu();
$vector = ZTensor::ones([512])->toGpu();

for ($warmup = 0; $warmup < 50; ++$warmup) {
    $temporary = $base->greater(0)->cumsum();
    unset($temporary);
}
gc_collect_cycles();
$rssAfterWarmup = stressRss();
$rssSamples = [];

for ($iteration = 0; $iteration < 1000; ++$iteration) {
    switch ($iteration % 5) {
        case 0:
            $result = $base->greater(0);
            break;
        case 1:
            $result = ZTensor::tile($tile, ($iteration % 4) + 1);
            break;
        case 2:
            $result = $ones->cumsum();
            break;
        case 3:
            $result = $matrix->dot($vector);
            break;
        default:
            $scalar = $base->dot($ones);
            stressClose($scalar, 0.0, 0.05);
            $result = null;
    }
    if ($result instanceof ZTensor && !$result->isOnGpu()) {
        throw new RuntimeException('temporary result lost device residency');
    }
    if (($iteration % 3) === 0) gc_collect_cycles();
    unset($result);
    if (($iteration % 100) === 0) $rssSamples[] = stressRss();
}
gc_collect_cycles();

$cloneSource = ZTensor::arr([[1, 2], [3, 4]])->toGpu()->greater(2);
$clone = clone $cloneSource;
if (!$clone->isOnGpu() || $clone->toArray() !== [[0.0, 0.0], [1.0, 1.0]]) {
    throw new RuntimeException('device-only PHP clone is not independent and valid');
}
unset($cloneSource, $clone);

$sameCountDifferentShape = ZTensor::ones([2, 8])->toGpu()->reshape([4, 4]);
if ($sameCountDifferentShape->shape() !== [4, 4]) throw new RuntimeException('same-count reshape failed');
$empty = ZTensor::zeros([0])->toGpu()->greater(0);
if (!$empty->isEmpty()) throw new RuntimeException('empty device result failed');

try {
    ZTensor::ones([2, 3])->toGpu()->dot(ZTensor::ones([4])->toGpu());
    throw new RuntimeException('expected shape exception');
} catch (Throwable $throwable) {
    if ($throwable->getMessage() === 'expected shape exception') throw $throwable;
}
$recovered = ZTensor::ones([2, 3])->toGpu()->dot(ZTensor::ones([3])->toGpu());
if ($recovered->toArray() !== [3.0, 3.0]) throw new RuntimeException('recovery after exception failed');

$rssFinal = stressRss();
$growth = $rssFinal - $rssAfterWarmup;
if ($growth > 128 * 1024 * 1024) {
    throw new RuntimeException("RSS did not stabilize after warm-up: growth=$growth");
}

echo json_encode([
    'allocator' => getenv('ZMATRIX_CUDA_ALLOCATOR') ?: 'auto',
    'iterations' => 1000,
    'rss_after_warmup' => $rssAfterWarmup,
    'rss_final' => $rssFinal,
    'rss_growth' => $growth,
    'rss_samples' => $rssSamples,
    'php_memory' => memory_get_usage(true),
    'php_peak' => memory_get_peak_usage(true),
    'status' => 'PASS',
], JSON_THROW_ON_ERROR) . "\n";
