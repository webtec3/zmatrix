<?php

declare(strict_types=1);

$root = dirname(__DIR__, 2);
$extension = $root . '/modules/zmatrix.so';
$caseScript = __DIR__ . '/isolated_linalg_case.php';
$allocator = getenv('ZMATRIX_CUDA_ALLOCATOR') ?: 'auto';
$cases = [];
foreach ([1024, 65536, 1048576, 16777216] as $size) {
    foreach (['cpu_cpu', 'gpu_gpu', 'gpu_cpu', 'cpu_gpu'] as $scenario) $cases[] = ['dot', $size, $scenario];
}
foreach ([512, 1024, 2048, 4096] as $size) {
    foreach (['cpu_cpu', 'gpu_gpu', 'gpu_cpu', 'cpu_gpu'] as $scenario) $cases[] = ['matvec', $size, $scenario];
}

$results = [];
foreach ($cases as [$operation, $size, $scenario]) {
    $command = [PHP_BINARY, '-n', '-d', "extension=$extension", $caseScript, $operation, (string) $size, $scenario];
    $pipes = [];
    $process = proc_open($command, [['pipe', 'r'], ['pipe', 'w'], ['pipe', 'w']], $pipes, $root,
        ['ZMATRIX_CUDA_ALLOCATOR' => $allocator]);
    if (!is_resource($process)) throw new RuntimeException('unable to launch isolated process');
    fclose($pipes[0]); $stdout = stream_get_contents($pipes[1]); fclose($pipes[1]);
    $stderr = stream_get_contents($pipes[2]); fclose($pipes[2]);
    $status = proc_close($process);
    if ($status !== 0) throw new RuntimeException("$operation/$size/$scenario failed: $stderr");
    $results[] = json_decode(trim($stdout), true, flags: JSON_THROW_ON_ERROR);
    echo "completed $operation $size $scenario\n";
}

$document = ['environment' => [
    'allocator' => $allocator, 'process_per_case' => true, 'warmups' => 3, 'repetitions' => 15,
    'commit' => trim((string) shell_exec('git -C ' . escapeshellarg($root) . ' rev-parse HEAD')),
    'binary_sha256' => hash_file('sha256', $extension),
], 'results' => $results];
$suffix = date('Ymd_His');
$path = __DIR__ . "/results/isolated_linalg_$suffix.json";
file_put_contents($path, json_encode($document, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . "\n");
echo "$path\n";
