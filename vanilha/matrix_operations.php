<?php
/**
 * Implementação de operações com matrizes em PHP puro
 */

/**
 * Adiciona duas matrizes
 * @param array $A Primeira matriz
 * @param array $B Segunda matriz
 * @return array Matriz resultante da soma
 */
function matrix_add(array $A, array $B): array {
    $rows = count($A);
    $cols = count($A[0]);
    $result = [];

    for ($i = 0; $i < $rows; $i++) {
        $result[$i] = [];
        for ($j = 0; $j < $cols; $j++) {
            $result[$i][$j] = $A[$i][$j] + $B[$i][$j];
        }
    }

    return $result;
}

/**
 * Subtrai uma matriz de outra
 * @param array $A Primeira matriz
 * @param array $B Segunda matriz
 * @return array Matriz resultante da subtração
 */
function matrix_subtract(array $A, array $B): array {
    $rows = count($A);
    $cols = count($A[0]);
    $result = [];

    for ($i = 0; $i < $rows; $i++) {
        $result[$i] = [];
        for ($j = 0; $j < $cols; $j++) {
            $result[$i][$j] = $A[$i][$j] - $B[$i][$j];
        }
    }

    return $result;
}

/**
 * Multiplica duas matrizes
 * @param array $A Primeira matriz
 * @param array $B Segunda matriz
 * @return array Matriz resultante da multiplicação
 */
function matrix_multiply(array $A, array $B): array {
    $rowsA = count($A);
    $colsA = count($A[0]);
    $colsB = count($B[0]);
    $result = [];

    for ($i = 0; $i < $rowsA; $i++) {
        $result[$i] = [];
        for ($j = 0; $j < $colsB; $j++) {
            $result[$i][$j] = 0;
            for ($k = 0; $k < $colsA; $k++) {
                $result[$i][$j] += $A[$i][$k] * $B[$k][$j];
            }
        }
    }

    return $result;
}

/**
 * Multiplica uma matriz por um escalar
 * @param array $A Matriz
 * @param float $scalar Valor escalar
 * @return array Matriz resultante
 */
function matrix_scalar_multiply(array $A, float $scalar): array {
    $rows = count($A);
    $cols = count($A[0]);
    $result = [];

    for ($i = 0; $i < $rows; $i++) {
        $result[$i] = [];
        for ($j = 0; $j < $cols; $j++) {
            $result[$i][$j] = $A[$i][$j] * $scalar;
        }
    }

    return $result;
}

/**
 * Calcula a transposta de uma matriz
 * @param array $A Matriz
 * @return array Matriz transposta
 */
function matrix_transpose(array $A): array {
    $rows = count($A);
    $cols = count($A[0]);
    $result = [];

    for ($j = 0; $j < $cols; $j++) {
        $result[$j] = [];
        for ($i = 0; $i < $rows; $i++) {
            $result[$j][$i] = $A[$i][$j];
        }
    }

    return $result;
}

/**
 * Cria uma matriz identidade
 * @param int $size Tamanho da matriz
 * @return array Matriz identidade
 */
function matrix_identity(int $size): array {
    $result = [];

    for ($i = 0; $i < $size; $i++) {
        $result[$i] = [];
        for ($j = 0; $j < $size; $j++) {
            $result[$i][$j] = ($i === $j) ? 1 : 0;
        }
    }

    return $result;
}

/**
 * Cria uma matriz nula
 * @param int $rows Número de linhas
 * @param int $cols Número de colunas
 * @return array Matriz nula
 */
function matrix_zero(int $rows, int $cols): array {
    $result = [];

    for ($i = 0; $i < $rows; $i++) {
        $result[$i] = array_fill(0, $cols, 0);
    }

    return $result;
}

/**
 * Calcula o traço de uma matriz
 * @param array $A Matriz
 * @return float Traço da matriz
 */
function matrix_trace(array $A): float {
    $size = count($A);
    $trace = 0;

    for ($i = 0; $i < $size; $i++) {
        $trace += $A[$i][$i];
    }

    return $trace;
}

/**
 * Verifica se uma matriz é quadrada
 * @param array $A Matriz
 * @return bool True se for quadrada, False caso contrário
 */
function matrix_is_square(array $A): bool {
    $rows = count($A);
    if ($rows === 0) return false;

    $cols = count($A[0]);
    return $rows === $cols;
}

/**
 * Verifica se uma matriz é simétrica
 * @param array $A Matriz
 * @return bool True se for simétrica, False caso contrário
 */
function matrix_is_symmetric(array $A): bool {
    if (!matrix_is_square($A)) {
        return false;
    }

    $size = count($A);

    for ($i = 0; $i < $size; $i++) {
        for ($j = 0; $j < $i; $j++) {
            if ($A[$i][$j] !== $A[$j][$i]) {
                return false;
            }
        }
    }

    return true;
}

/**
 * Calcula o determinante de uma matriz 2x2
 * @param array $A Matriz
 * @return float Determinante
 */
function matrix_determinant_2x2(array $A): float {
    return $A[0][0] * $A[1][1] - $A[0][1] * $A[1][0];
}

/**
 * Calcula o determinante de uma matriz NxN usando a expansão de Laplace
 * @param array $A Matriz
 * @return float Determinante
 */
function matrix_determinant(array $A): float {
    $size = count($A);

    // Caso base: matriz 1x1
    if ($size === 1) {
        return $A[0][0];
    }

    // Caso base: matriz 2x2
    if ($size === 2) {
        return matrix_determinant_2x2($A);
    }

    $det = 0;

    // Expansão de Laplace ao longo da primeira linha
    for ($j = 0; $j < $size; $j++) {
        // Criar submatriz
        $submatrix = [];
        for ($i = 1; $i < $size; $i++) {
            $row = [];
            for ($k = 0; $k < $size; $k++) {
                if ($k !== $j) {
                    $row[] = $A[$i][$k];
                }
            }
            $submatrix[] = $row;
        }

        // Usar o cofator para adicionar à soma
        $sign = ($j % 2 === 0) ? 1 : -1;
        $det += $sign * $A[0][$j] * matrix_determinant($submatrix);
    }

    return $det;
}

/**
 * Imprime uma matriz formatada
 * @param string $label Rótulo da matriz
 * @param array $matrix Matriz a ser impressa
 */
function printMatrix(string $label, array $matrix): void
{
    echo "$label\n";
    foreach ($matrix as $row) {
        echo implode("\t", $row) . "\n";
    }
    echo "\n";
}