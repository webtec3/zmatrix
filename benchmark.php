<?php
/**
 * Script aprimorado para comparar performance entre implementação nativa (zmatrix)
 * e implementação em PHP puro das operações com matrizes
 */

// Incluir as funções em PHP puro
require_once 'vanilha/matrix_operations.php';
require_once __DIR__ . '/php/Matrix.php';

/**
 * Benchmark para comparar desempenho
 *
 * @param callable $func Função a ser testada
 * @param array $args Argumentos para a função
 * @param int $iterations Número de iterações
 * @return array Resultados do benchmark [tempo (s), memória (bytes)]
 */
function benchmark(string $label, callable $fn): mixed  {
    $startMemory = memory_get_usage(true);
    $startTime = microtime(true);
    $result = $fn();
    $endTime = microtime(true);
    $endMemory = memory_get_usage(true);

    echo "$label:\n";
    echo "  Tempo: " . number_format($endTime - $startTime, 6) . " segundos\n";
    echo "  Memória: " . number_format(($endMemory - $startMemory) / 1024, 2) . " KB\n\n";

    return $result;
}

// Dados de exemplo
$size = 500;
$A = [];
$B = [];

for ($i = 0; $i < $size; $i++) {
    for ($j = 0; $j < $size; $j++) {
        $A[$i][$j] = rand(1, 10);
        $B[$i][$j] = rand(1, 10);
    }
}

benchmark('PHP: matrix_add', fn() => matrix_add($A, $B));
benchmark('PHP: matrix_subtract', fn() => matrix_subtract($A, $B));
benchmark('PHP: matrix_multiply', fn() => matrix_multiply($A, $B));
benchmark('PHP: matrix_scalar_multiply', fn() => matrix_scalar_multiply($A, 2));
benchmark('PHP: matrix_transpose', fn() => matrix_transpose($A));
benchmark('PHP: matrix_identity', fn() => matrix_identity($size));
benchmark('PHP: matrix_trace', fn() => matrix_trace($A));
benchmark('PHP: matrix_is_symmetric', fn() => matrix_is_symmetric($A));

// determinante pode demorar muito! (só faça com size <= 8)
if ($size <= 8) {
    benchmark('PHP: matrix_determinant', fn() => matrix_determinant($A));
}

// --------------------------
// Extensão Nativa ZMatrix
// --------------------------
use ZMatrix\Matrix;

$matA = new Matrix($A);
$matB = new Matrix($B);
$scalar = 2;

benchmark('ZMatrix: add', fn() => $matA->add($matB));
benchmark('ZMatrix: subtract', fn() => $matA->subtract($matB));
benchmark('ZMatrix: multiply', fn() => $matA->multiply($matB));
benchmark('ZMatrix: scalarMultiply', fn() => $matA->scalarMultiply($scalar));
benchmark('ZMatrix: transpose', fn() => $matA->transpose());
benchmark('ZMatrix: identity', fn() => Matrix::identity($size));
benchmark('ZMatrix: trace', fn() => $matA->trace());
benchmark('ZMatrix: isSymmetric', fn() => $matA->isSymmetric());

if ($size <= 8) {
    benchmark('ZMatrix: determinant', fn() => $matA->determinant());
}