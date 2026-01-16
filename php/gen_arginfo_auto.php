#!/usr/bin/env php
<?php

declare(strict_types=1);

$targetDir = __DIR__;
$zmatrixStub = $targetDir . '/zmatrix.stub.php';
$ztensorStub = $targetDir . '/ztensor.stub.php';
$zmatrixArginfo = $targetDir . '/zmatrix_arginfo.h';
$ztensorArginfo = $targetDir . '/ztensor_arginfo.h';

function generateArginfo($stubFile, $outputFile, $label) {
    if (!file_exists($stubFile)) {
        echo "❌ Stub não encontrado: $stubFile\n";
        return false;
    }

    // Tentar com zend_test
    $cmd = sprintf(
        'php -dextension=zend_test.so -dzend_extension=opcache -dzend_test.generate_arginfo=1 %s 2>/dev/null',
        escapeshellarg($stubFile)
    );
    
    echo "🔄 Gerando arginfo de: $label\n";
    exec($cmd, $output, $returnVar);

    if ($returnVar === 0 && !empty($output)) {
        $content = implode("\n", $output);
        file_put_contents($outputFile, $content);
        echo "✅ Gerado: $label → $outputFile\n";
        return true;
    } else {
        // Fallback: sem zend_test
        echo "⚠️  Tentando sem zend_test.so...\n";
        $cmd2 = sprintf(
            'php -dzend_extension=opcache %s 2>/dev/null | head -50',
            escapeshellarg($stubFile)
        );
        exec($cmd2, $output2, $returnVar2);
        
        if ($returnVar2 === 0 && !empty($output2)) {
            $content = implode("\n", $output2);
            file_put_contents($outputFile, $content);
            echo "✅ Gerado (sem zend_test): $label\n";
            return true;
        }
    }
    
    echo "❌ Falha ao gerar: $label\n";
    return false;
}

// Processar ambos os stubs
$success = true;
$success &= generateArginfo($zmatrixStub, $zmatrixArginfo, 'zmatrix.stub.php');
$success &= generateArginfo($ztensorStub, $ztensorArginfo, 'ztensor.stub.php');

if ($success) {
    echo "\n✅ Todos os arginfo foram regenerados!\n";
    exit(0);
} else {
    echo "\n⚠️  Alguns arginfo falharam. Verifique manualmente.\n";
    exit(1);
}
