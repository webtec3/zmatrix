<?php

declare(strict_types=1);

$cleanChild = getenv('ZMATRIX_BENCH_CLEAN_CHILD') === '1';

if (!$cleanChild && extension_loaded('xdebug')) {
    $extension = rtrim((string) ini_get('extension_dir'), DIRECTORY_SEPARATOR)
        . DIRECTORY_SEPARATOR
        . 'zmatrix.so';

    $arguments = array_map('escapeshellarg', array_slice($argv, 1));
    $command = sprintf(
        'ZMATRIX_BENCH_CLEAN_CHILD=1 OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 %s -n -d extension=%s %s %s',
        escapeshellarg(PHP_BINARY),
        escapeshellarg($extension),
        escapeshellarg(__FILE__),
        implode(' ', $arguments),
    );

    fwrite(STDERR, "Xdebug detected; restarting benchmark with a clean php -n process.\n");
    passthru($command, $status);
    exit($status);
}

require __DIR__ . '/benchmarks/reliable_gpu/run.php';
