<?php

declare(strict_types=1);

function captureStats(array $values): array {
    sort($values, SORT_NUMERIC);
    $at = static function (float $p) use ($values): float {
        $position = $p * (count($values) - 1); $lo = (int) floor($position); $hi = (int) ceil($position);
        return $values[$lo] + ($values[$hi] - $values[$lo]) * ($position - $lo);
    };
    return ['p25_ms' => $at(0.25), 'median_ms' => $at(0.5), 'p75_ms' => $at(0.75), 'samples_ms' => $values];
}

$root = dirname(__DIR__, 2);
$command = ['env', 'ZMATRIX_CUDA_PROFILE=1', 'ZMATRIX_CUDA_ALLOCATOR=' . (getenv('ZMATRIX_CUDA_ALLOCATOR') ?: 'auto'),
    PHP_BINARY, '-n', '-d', 'extension=' . $root . '/modules/zmatrix.so', __DIR__ . '/hotpath_profile.php'];
$pipes = [];
$process = proc_open($command, [['pipe', 'r'], ['pipe', 'w'], ['pipe', 'w']], $pipes, $root);
if (!is_resource($process)) throw new RuntimeException('unable to start profiling child');
fclose($pipes[0]); $stdout = stream_get_contents($pipes[1]); fclose($pipes[1]);
$stderr = stream_get_contents($pipes[2]); fclose($pipes[2]);
$status = proc_close($process);
if ($status !== 0) throw new RuntimeException("profiling child failed: $stderr");

$wrappers = $phpWrappers = $lifecycle = $device = [];
foreach (preg_split('/\R/', $stderr) as $line) {
    if (preg_match('/^\[zmatrix\]\[cuda-wrapper\] (\{.*\})$/', $line, $match)) {
        $record = json_decode($match[1], true, flags: JSON_THROW_ON_ERROR);
        $wrappers[$record['operation']][] = $record;
    } elseif (preg_match('/^\[zmatrix\]\[php-wrapper\] (\{.*\})$/', $line, $match)) {
        $record = json_decode($match[1], true, flags: JSON_THROW_ON_ERROR);
        $phpWrappers[$record['operation']][] = $record;
    } elseif (preg_match('/^\[zmatrix\]\[cuda-lifecycle\] (\{.*\})$/', $line, $match)) {
        $lifecycle[] = json_decode($match[1], true, flags: JSON_THROW_ON_ERROR);
    } elseif (preg_match('/^\[zmatrix\]\[cuda-profile\] op=(\S+) device_ms=([0-9.]+)/', $line, $match)) {
        $device[$match[1]][] = (float) $match[2];
    }
}

$wrapperSummary = [];
foreach ($wrappers as $operation => $records) {
    $steady = array_slice($records, -7);
    foreach (['result_creation_ms', 'device_allocation_ms', 'operand_transfer_ms', 'cuda_submit_ms', 'state_update_ms'] as $field) {
        $wrapperSummary[$operation][$field] = captureStats(array_column($steady, $field));
    }
}
$deviceSummary = [];
foreach ($device as $operation => $samples) $deviceSummary[$operation] = captureStats(array_slice($samples, -7));
$phpSummary = [];
foreach ($phpWrappers as $operation => $records) {
    $steady = array_slice($records, -7);
    foreach (['parse_ms', 'validation_and_residency_ms', 'return_construction_ms'] as $field) {
        $phpSummary[$operation][$field] = captureStats(array_column($steady, $field));
    }
}
$lifecycleGroups = [];
foreach ($lifecycle as $record) {
    if ($record['bytes'] === 0) continue;
    $key = $record['event'] . ':' . $record['mode'] . ':' . $record['bytes'];
    $lifecycleGroups[$key][] = (float) $record['host_ms'];
}
$lifecycleSummary = [];
foreach ($lifecycleGroups as $key => $samples) $lifecycleSummary[$key] = captureStats($samples);

$document = ['environment' => [
    'allocator' => getenv('ZMATRIX_CUDA_ALLOCATOR') ?: 'auto',
    'commit' => trim((string) shell_exec('git -C ' . escapeshellarg($root) . ' rev-parse HEAD')),
    'binary_sha256' => hash_file('sha256', $root . '/modules/zmatrix.so'),
], 'wrapper_summary' => $wrapperSummary, 'php_summary' => $phpSummary, 'device_summary' => $deviceSummary,
   'lifecycle_summary' => $lifecycleSummary, 'raw' => ['wrapper' => $wrappers, 'device' => $device, 'lifecycle' => $lifecycle],
   'child_stdout' => $stdout, 'raw_php_wrapper' => $phpWrappers];
$suffix = date('Ymd_His');
$path = __DIR__ . "/results/wrapper_decomposition_$suffix.json";
file_put_contents($path, json_encode($document, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . "\n");
echo "$path\n";
