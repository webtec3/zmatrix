<?php

namespace ZMatrix;

use InvalidArgumentException;

class Matrix
{
    /**
     * @var array
     */
    protected array $data;

    /**
     * @param array $data
     */
    public function __construct(array $data)
    {
        if (empty($data)) {
            $this->data = [];
            return;
        }
        $this->validateMatrix($data);
        $this->data = $data;
    }

    /**
     * @param array $matrix
     * @return void
     */
    protected function validateMatrix(array $matrix): void
    {

    }

    /**
     * @param int $start
     * @param int $end
     * @param int $step
     * @return Matrix
     */
    public static function arange(int $start, int $end, int $step): Matrix
    {
        return new self(\zmatrix_arange($start, $end, $step));
    }

    /**
     * @param int $start
     * @param int $end
     * @param int $num
     * @return Matrix
     */
    public static function linspace(int $start, int $end, int $num): Matrix
    {
        return new self([\zmatrix_linspace($start, $end, $num)]);
    }

    /**
     * @param int $startExp
     * @param int $endExp
     * @param int $num
     * @return Matrix
     */
    public static function logspace(int $startExp, int $endExp, int $num): Matrix
    {
        return new self([\zmatrix_logspace($startExp, $endExp, $num)]);
    }

    /**
     * @param int $rows
     * @param int $cols
     * @return Matrix
     */
    public static function rand(int $rows, int $cols): Matrix
    {
        return new self([\zmatrix_rand($rows, $cols)]);
    }

    /**
     * @param int $rows
     * @param int $cols
     * @return Matrix
     */
    public static function randn(int $rows, int $cols): Matrix
    {
        return new self([\zmatrix_randn($rows, $cols)]);
    }

    /**
     * @param int $size
     * @return Matrix
     */
    public static function eye(int $size): Matrix
    {
        return new self(\zmatrix_eye($size));
    }


    /**
     * @param mixed $result
     * @return Matrix
     */
    protected function fromNativeResult(mixed $result): Matrix
    {
        if (!is_array($result)) {
            throw new \RuntimeException("Erro ao processar operação da matriz.");
        }
        return new Matrix($result);
    }

    /**
     * @return array
     */
    public function getData(): array
    {
        return $this->data;
    }

    /**
     * @return float
     */
    public function determinant(): float
    {
        return \zmatrix_determinant($this->data);
    }

    /**
     * @param Matrix $other
     * @return Matrix
     */
    public function add(Matrix $other): Matrix
    {
        return $this->fromNativeResult(\zmatrix_add($this->data, $other->data));
    }

    /**
     * @param Matrix $other
     * @return Matrix
     */
    public function subtract(Matrix $other): Matrix
    {
        return $this->fromNativeResult(\zmatrix_subtract($this->data, $other->data));
    }

    /**
     * @param Matrix $other
     * @return Matrix
     */
    public function multiply(Matrix $other): Matrix
    {
        return $this->fromNativeResult(\zmatrix_multiply($this->data, $other->data));
    }

    /**
     * @param float|int $scalar
     * @return Matrix
     */
    public function scalarMultiply(float|int $scalar): Matrix
    {
        return $this->fromNativeResult(\zmatrix_scalar_multiply($this->data, $scalar));
    }

    /**
     * @return Matrix
     */
    public function transpose(): Matrix
    {
        return $this->fromNativeResult(\zmatrix_transpose($this->data));
    }

    /**
     * @return float
     */
    public function trace(): float
    {
        return \zmatrix_trace($this->data);
    }

    /**
     * @return bool
     */
    public function isSquare(): bool
    {
        return \zmatrix_is_square($this->data);
    }

    /**
     * @return bool
     */
    public function isSymmetric(): bool
    {
        return \zmatrix_is_symmetric($this->data);
    }

    /**
     * @param int $size
     * @return Matrix
     */
    public static function identity(int $size): Matrix
    {
        $result = \zmatrix_identity($size);
        if (!is_array($result)) {
            throw new \RuntimeException("Erro ao gerar matriz identidade.");
        }
        return new Matrix($result);
    }

    /**
     * @param $rows
     * @param $cols
     * @return Matrix
     */
    public static function ones($rows, $cols): Matrix
    {
        $matrix = \zmatrix_zeros([$rows, $cols]);
        return new self(\zmatrix_map($matrix, static function ($val) {
            return 1.0;
        }));
    }

    /**
     * @param int $rows
     * @param int $cols
     * @param float $min
     * @param float $max
     * @return Matrix
     */
    public static function random(int $rows, int $cols, float $min = 0.0, float $max = 1.0): Matrix
    {
        if ($rows <= 0 || $cols <= 0 || $min > $max) {
            throw new InvalidArgumentException("Parâmetros inválidos para matriz aleatória");
        }
        return new Matrix(\zmatrix_random($rows, $cols, $min, $max));
    }

    /**
     * @param Matrix $other
     * @return Matrix
     */
    public function dot(Matrix $other): Matrix
    {
        return $this->fromNativeResult(\zmatrix_dot($this->data, $other->data));
    }

    /**
     * @return Matrix
     */
    public function abs(): Matrix
    {
        return $this->fromNativeResult(\zmatrix_abs($this->data));
    }

    /**
     * @return float
     */
    public function mean(): float
    {
        return \zmatrix_mean($this->data);
    }

    /**
     * @return Matrix
     */
    public function sumRows(): Matrix
    {
        return $this->fromNativeResult(\zmatrix_sum_rows($this->data));
    }

    /**
     * @return Matrix
     */
    public function sigmoid(): Matrix
    {
        return $this->fromNativeResult(\zmatrix_sigmoid($this->data));
    }

    /**
     * @param int|null $index
     * @return array
     */
    public function toArray(int|null $index = null): array
    {
        return $this->data[$index] ?? $this->data;
    }


    /**
     * @param callable $func
     * @return Matrix
     */
    public function map(callable $func): Matrix
    {
        $result = [];
        foreach ($this->data as $row) {
            $resultRow = [];
            foreach ($row as $val) {
                $resultRow[] = $func($val);
            }
            $result[] = $resultRow;
        }
        return new Matrix($result);
    }

    /**
     * @return int
     */
    public function ndim(): int
    {
        return \zmatrix_ndim($this->data);
    }

    /**
     * @return array
     */
    public function shape(): array
    {
        return \zmatrix_shape($this->data);
    }

    /**
     * @return int
     */
    public function size(): int
    {
        return \zmatrix_size($this->data);
    }

    /**
     * @param $rows
     * @param $cols
     * @return Matrix
     */
    public function reshape($rows, $cols): Matrix
    {
        return $this->fromNativeResult(\zmatrix_reshape($this->data, $rows, $cols));
    }

    /**
     * @return array
     */
    public function flatten(): array
    {
        return \zmatrix_flatten($this->data);
    }

    /**
     * @return array
     */
    public function ravel(): array
    {
        return \zmatrix_ravel($this->data);
    }

    /**
     * @return float
     */
    public function std(): float
    {
        return \zmatrix_std($this->data);
    }

    /**
     * @return int
     */
    public function min(): int
    {
        return \zmatrix_min($this->data);
    }

    /**
     * @return int
     */
    public function max(): int
    {
        return \zmatrix_max($this->data);
    }

    /**
     * @return int
     */
    public function sum(): int
    {
        return \zmatrix_sum($this->data);
    }

    /**
     * Cria um tensor aleatório com dimensões especificadas
     *
     * @param array $shape Array com as dimensões do tensor
     * @param float $min Valor mínimo (opcional, padrão 0.0)
     * @param float $max Valor máximo (opcional, padrão 1.0)
     * @return Matrix Nova instância com o tensor aleatório
     * @return Matrix
     * @throws InvalidArgumentException Se as dimensões forem inválidas
     */
    public static function randomTensor(array $shape, float $min = 0.0, float $max = 1.0): Matrix
    {
        foreach ($shape as $dim) {
            if ($dim <= 0) {
                throw new InvalidArgumentException("Todas as dimensões devem ser positivas");
            }
        }

        return new self(\zmatrix_random_tensor($shape, $min, $max));
    }


    /**
     * Reduz o tensor aplicando uma operação ao longo de um eixo
     *
     * @param string $operation Operação a ser aplicada ('sum', 'mean', 'min', 'max', etc.)
     * @param int $axis Eixo ao longo do qual aplicar a operação (opcional)
     * @return float|Matrix Resultado da redução
     * @return Matrix|array
     */
    public function reduce(string $operation, int $axis = 0): array|Matrix
    {
        return $this->fromNativeResult(\zmatrix_reduce($this->data, $operation, $axis));
    }


    /**
     * Executa benchmark de operações matriciais
     *
     * @param string $operation Operação a ser testada ('add', 'multiply', etc.)
     * @param int $size Tamanho da matriz para o benchmark
     * @param int $iterations Número de iterações (opcional, padrão 10)
     * @return mixed Resultados do benchmark
     */
    public static function benchmark(string $operation, int $size, int $iterations = 10): mixed
    {
        return \zmatrix_benchmark($operation, $size, $iterations);
    }

    /**
     * Concatena esta matriz com outra ao longo de um eixo específico
     *
     * @param int $axis Eixo ao longo do qual concatenar (opcional, padrão 0)
     * @return Matrix Resultado da concatenação
     * @return Matrix
     */
    public function concatenate(int $axis = 0): Matrix
    {
        return $this->fromNativeResult(\zmatrix_concatenate($this->data,  $axis));
    }


    /**
     * Cria uma matriz de zeros com dimensões especificadas
     *
     * @param int $rows Número de linhas
     * @param int $cols Número de colunas
     * @return Matrix Nova instância com a matriz de zeros
     * @throws InvalidArgumentException Se as dimensões forem inválidas
     */
    public static function zeros(int $rows, int $cols): Matrix
    {
        if ($rows <= 0 || $cols <= 0) {
            throw new InvalidArgumentException("Dimensões devem ser positivas");
        }

        return new self(\zmatrix_zeros([$rows, $cols]));
    }
}
