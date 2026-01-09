#!/usr/bin/env php
<?php

declare(strict_types=1);

$stubFile = __DIR__ . '/ztensor.stub.php';
$targetDir = __DIR__;
$generatedArginfo = $targetDir . '/zmatrix_arginfo.h';

if (!file_exists($stubFile)) {
    echo "❌ Stub file not found: $stubFile\n";
    exit(1);
}

$cmd = sprintf(
    'php -dextension=zend_test.so -dzend_extension=opcache -dzend_test.generate_arginfo=1 %s > %s',
    escapeshellarg($stubFile),
    escapeshellarg($generatedArginfo)
);

echo "📦 Gerando arginfo de: zmatrix.stub.php\n";
exec($cmd, $output, $returnVar);

if ($returnVar === 0) {
    echo "✅ Arquivo gerado com sucesso: $generatedArginfo\n";
} else {
    echo "❌ Falha ao gerar arginfo.\n";
}
