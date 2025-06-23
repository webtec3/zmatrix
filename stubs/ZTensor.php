<?php

declare(strict_types=1);

namespace ZMatrix;

use InvalidArgumentException;
use RuntimeException;
use TypeError;

/**
 * ZTensor Interface
 *
 * Defines the structure and documentation for the ZTensor class,
 * representing a multidimensional tensor implemented in C++.
 *
 * @package ZMatrix
 */
final class ZTensor
{
    /**
     * ZTensor constructor.
     *
     * Creates a tensor from a PHP multidimensional array or an empty tensor.
     * If no argument is provided, it creates an empty tensor with no defined shape.
     * If an array is provided, it infers the shape and populates the data.
     * Throws an exception if the argument is not an array or null, or if the array
     * structure is invalid/inconsistent.
     *
     * @param ZTensor|array<int|float>|null $dataOrShape Initial data as a nested PHP array,
     * or null to create an empty tensor.
     * @throws InvalidArgumentException If the argument is not an array or null.
     * @throws RuntimeException If the array structure is invalid or if there is an allocation failure.
     * @throws TypeError If array elements are not numeric (int/float).
     */
    public function __construct(ZTensor|array|null $dataOrShape = null)
    {
    }

    /**
     * Creates a new ZTensor from a PHP array or another ZTensor, safely.
     * @param ZTensor|array $arrayData The input data.
     * @return ZTensor A new ZTensor instance.
     */
    public static function safe(ZTensor|array $arrayData): ZTensor
    {
    }

    /**
     * Creates a new ZTensor from a PHP array.
     *
     * This is a convenient factory method, similar to new ZTensor($array).
     *
     * @param ZTensor|array<int|float> $arrayData Initial data as a nested PHP array.
     * @return ZTensor The new tensor created from the array.
     * @throws InvalidArgumentException If the argument is not an array.
     * @throws RuntimeException If the array structure is invalid or if there is an allocation failure.
     * @throws TypeError If array elements are not numeric (int/float).
     */
    public static function arr(ZTensor|array $arrayData): ZTensor
    {
    }

    /**
     * Creates a deep copy of the current tensor.
     * @return ZTensor A new ZTensor instance that is a copy of the original.
     */
    public function copy(): ZTensor
    {
    }

    /**
     * Adds another tensor (or array) to this tensor, element-wise.
     *
     * Both tensors must have the exact same shape or be broadcastable.
     *
     * @param ZTensor|float|int|array<int|float> $other The other tensor or array to add.
     * @return ZTensor A new tensor with the addition result.
     * @throws RuntimeException If shapes are incompatible (ZMATRIX_ERR_SHAPE_MISMATCH)
     * or if another internal error occurs.
     */
    public function add(ZTensor|array|float|int $other): Ztensor
    {
    }

    /**
     * Subtracts another tensor (or array) from this tensor, element-wise.
     *
     * Both tensors must have the exact same shape or be broadcastable.
     *
     * @param ZTensor|float|int|array<int|float> $other The other tensor or array to subtract.
     * @return ZTensor A new tensor containing the subtraction result.
     * @throws RuntimeException If shapes are incompatible (ZMATRIX_ERR_SHAPE_MISMATCH)
     * or if another internal error occurs.
     */
    public function sub(ZTensor|array|float|int $other): ZTensor
    {
    }

    /**
     * Sums all elements of the tensor or along a specific axis.
     *
     * - If `$axis` is `null`, returns a scalar tensor with the total sum of all elements.
     * - If `$axis` is an integer, it sums along the specified axis and writes the result into `$other`.
     *
     * @param ZTensor|array<int|float> $other Output tensor for the result (requires a compatible shape if $axis is an integer).
     * @param int|null $axis The axis to sum along, or null for a total sum.
     * @return ZTensor A new tensor containing the total sum (or returns $this if summing by axis into an output tensor).
     * @throws RuntimeException If shapes are incompatible or the axis is invalid.
     */
    public function sum(ZTensor|array $other, int|null $axis = null): ZTensor
    {
    }

    /**
     * Clips the tensor values within a specified range.
     *
     * This static method receives a `ZTensor` or a multidimensional PHP array,
     * and returns a new `ZTensor` object with all values limited to the range
     * defined by `$min` and `$max`. All values less than `$min` will be converted to `$min`,
     * and all values greater than `$max` will be converted to `$max`.
     *
     * Example:
     * ```php
     * $t = ZTensor::arr([[1, 5, 10], [-2, 0, 15]]);
     * $clipped = ZTensor::clip($t, 0, 10);
     * print_r($clipped->toArray());
     * // Result:
     * // [
     * //   [1.0, 5.0, 10.0],
     * //   [0.0, 0.0, 10.0]
     * // ]
     * ```
     *
     * @param ZTensor|array $array Input tensor (ZTensor object or multidimensional numeric array)
     * @param float $min Minimum allowed value for each element
     * @param float $max Maximum allowed value for each element
     * @return ZTensor Returns a new ZTensor object with clipped values
     * @throws RuntimeException If arguments are invalid or shapes are incompatible
     */
    public static function clip(ZTensor|array $array, float $min, float $max): ZTensor
    {
    }

    /**
     * Multiplies this tensor by another tensor (or array), element-wise (Hadamard product).
     *
     * Both tensors must have the exact same shape or be broadcastable.
     *
     * @param ZTensor|float|int|array<int|float> $other The other tensor or array to be multiplied.
     * @return ZTensor A new tensor containing the result of the element-wise multiplication.
     * @throws RuntimeException If shapes are incompatible (ZMATRIX_ERR_SHAPE_MISMATCH)
     * or if another internal error occurs.
     */
    public function mul(ZTensor|array|float|int $other): Ztensor
    {
    }

    /**
     * Multiplies each element of this tensor by a scalar value.
     *
     * @param float $scalar The scalar value to multiply by.
     * @return ZTensor A new tensor with each element multiplied by the scalar.
     * @throws RuntimeException If an internal error occurs.
     */
    public function scalarMultiply(float $scalar): ZTensor
    {
    }

    /**
     * Divides each element of this tensor by a scalar value.
     *
     * @param float|ZTensor $scalar The scalar value or tensor to divide by.
     * @return ZTensor A new tensor with each element divided by the scalar.
     * @throws RuntimeException If an internal error occurs.
     */
    public function scalarDivide(float|ZTensor $scalar): ZTensor
    {
    }

    /**
     * Computes the transpose of this tensor.
     *
     * Currently implemented only for 2D tensors (matrices).
     * For an M x N matrix, it returns a new N x M matrix.
     *
     * @return ZTensor A new tensor representing the transpose.
     * @throws RuntimeException If the tensor is not 2D (ZMATRIX_ERR_UNSUPPORTED_OP)
     * or if another internal error occurs.
     */
    public function transpose(): ZTensor
    {
    }

    /**
     * Computes the absolute value of each element in the tensor.
     *
     * @return ZTensor A new tensor with the absolute values of the elements.
     * @throws RuntimeException If an internal error occurs.
     */
    public function abs(): ZTensor
    {
    }

    /**
     * Applies the sigmoid function to each element of the tensor (in-place).
     *
     * sigmoid(x) = 1 / (1 + exp(-x))
     *
     * @return ZTensor The modified tensor itself.
     * @throws RuntimeException If an internal error occurs (e.g., overflow in exp).
     */
    public function sigmoid(): ZTensor
    {
    }

    /**
     * Applies the derivative of the sigmoid function: σ(x) ⋅ (1 − σ(x))
     * Requires the tensor to already contain the values of sigmoid(x).
     *
     * @return ZTensor The modified tensor itself.
     * @throws RuntimeException
     */
    public function sigmoidDerivative(): ZTensor
    {
    }

    /**
     * Softmax: exp(x)/sum(exp(x)) along the last axis (1D or per-row in 2D).
     *
     * @return ZTensor The modified tensor itself.
     * @throws RuntimeException
     */
    public function softmax(): ZTensor
    {
    }

    /**
     * Simplified derivative of softmax: s ⋅ (1 − s)
     * (Diagonal of the Jacobian). Requires the tensor to contain softmax(x).
     *
     * @return ZTensor The modified tensor itself.
     * @throws RuntimeException
     */
    public function softmaxDerivative(): ZTensor
    {
    }

    /**
     * ReLU: Applies max(0, x) to each element.
     *
     * @return ZTensor The modified tensor itself.
     * @throws RuntimeException
     */
    public function relu(): ZTensor
    {
    }

    /**
     * Derivative of ReLU: 1 if x > 0, else 0.
     *
     * @return ZTensor The modified tensor itself.
     * @throws RuntimeException
     */
    public function reluDerivative(): ZTensor
    {
    }

    /**
     * tanh function applied to each element.
     *
     * @return ZTensor The modified tensor itself.
     * @throws RuntimeException
     */
    public function tanh(): ZTensor
    {
    }

    /**
     * Derivative of tanh: 1 − tanh²(x). Requires the tensor to already contain tanh(x).
     *
     * @return ZTensor The modified tensor itself.
     * @throws RuntimeException
     */
    public function tanhDerivative(): ZTensor
    {
    }

    /**
     * Leaky ReLU: x if x > 0, else αx (default α = 0.01)
     *
     * @param float $alpha The slope value for x < 0 (optional)
     * @return ZTensor The modified tensor itself.
     * @throws RuntimeException
     */
    public function leakyRelu(float $alpha = 0.01): ZTensor
    {
    }

    /**
     * Derivative of Leaky ReLU: 1 if x > 0, else α.
     *
     * @param float $alpha The same value used in the leakyRelu() function.
     * @return ZTensor The modified tensor itself.
     * @throws RuntimeException
     */
    public function leakyReluDerivative(float $alpha = 0.01): ZTensor
    {
    }

    /**
     * Calculates the sum of all elements in the tensor.
     *
     * @return float The total sum of the elements. Returns 0.0 for an empty tensor.
     * @throws RuntimeException If an internal error occurs.
     */
    public function sumtotal(): float
    {
    }

    /**
     * Calculates the average of all elements in the tensor.
     *
     * @return float The average of the elements. Returns NAN if the tensor is empty.
     * @throws RuntimeException If an internal error occurs.
     */
    public function mean(): float
    {
    }

    /**
     * Finds the minimum value among all elements of the tensor.
     *
     * @return float The minimum value. Returns NAN if the tensor is empty.
     * @throws RuntimeException If an internal error occurs.
     */
    public function min(): float
    {
    }

    /**
     * Finds the maximum value among all elements of the tensor.
     *
     * @return float The maximum value. Returns NAN if the tensor is empty.
     * @throws RuntimeException If an internal error occurs.
     */
    public function max(): float
    {
    }

    /**
     * Computes the element-wise minimum of a tensor and a scalar.
     *
     * @param ZTensor|array<int|float> $a The input tensor.
     * @param float $b The scalar value.
     * @return ZTensor A new tensor with the element-wise minimum.
     * @throws RuntimeException If an internal error occurs.
     */
    public static function minimum(array|ZTensor $a, float $b): ZTensor
    {
    }

    /**
     * Computes the element-wise maximum of a tensor and a scalar.
     *
     * @param ZTensor|array<int|float> $a The input tensor.
     * @param float $b The scalar value.
     * @return ZTensor A new tensor with the element-wise maximum.
     * @throws RuntimeException If an internal error occurs.
     */
    public static function maximum(array|ZTensor $a, float $b): ZTensor
    {
    }

    /**
     * Calculates the sample standard deviation (N-1) of all elements in the tensor.
     *
     * @return float The sample standard deviation. Returns NAN if the tensor has fewer than 2 elements or is empty.
     * @throws RuntimeException If an internal error occurs.
     */
    public function std(): float
    {
    }

    /**
     * Returns the shape (format/dimensions) of the tensor.
     *
     * @return array<int> A PHP array containing the size of each dimension.
     * Returns an empty array [] if the tensor has not been initialized with a shape.
     */
    public function shape(): array
    {
    }

    /**
     * Reshapes this tensor to a new format without copying data.
     *
     * @param int[] $shape An array with the new dimensions.
     * Must have the same total product of elements.
     * Can contain zeros to generate an empty tensor.
     * @return ZTensor A new tensor with the same data buffer, just a different view.
     * @throws RuntimeException If the number of elements does not match or dimensions are invalid.
     */
    public function reshape(array $shape): ZTensor
    {
    }

    /**
     * Returns the number of dimensions of the tensor (ndim).
     *
     * Equivalent to count($tensor->shape()).
     *
     * @return int The number of dimensions.
     */
    public function ndim(): int
    {
    }

    /**
     * Returns the total number of elements in the tensor.
     *
     * It is the product of all values in the shape.
     *
     * @return int The total number of elements. Returns 0 if any dimension is 0 or the shape is empty.
     */
    public function size(): int
    {
    }

    /**
     * Checks if the tensor is empty.
     *
     * A tensor is considered empty if it has not been initialized, if its shape is empty,
     * or if any of its dimensions has size 0.
     *
     * @return bool True if the tensor is empty, False otherwise.
     */
    public function isEmpty(): bool
    {
    }

    /**
     * Converts the tensor back to a multidimensional PHP array.
     *
     * @return array<int|float> The PHP array representing the data and shape of the tensor.
     * @throws RuntimeException If an internal error occurs during conversion.
     */
    public function toArray(): array
    {
    }

    /**
     * Creates a new tensor filled with zeros, with the specified shape.
     *
     * @param array<int> $shape An array of integers defining the dimensions of the new tensor.
     * The dimensions must be non-negative.
     * @return ZTensor The new tensor filled with zeros.
     * @throws InvalidArgumentException If the shape is not an array or contains invalid values (ZMATRIX_ERR_INVALID_SHAPE).
     * @throws RuntimeException If allocation fails or another error occurs.
     */
    public static function zeros(array $shape): ZTensor
    {
    }

    /**
     * Creates a new tensor filled with ones, with the specified shape.
     *
     * @param array<int> $shape An array of integers defining the dimensions of the new tensor.
     * The dimensions must be non-negative.
     * @return ZTensor The new tensor filled with ones.
     * @throws InvalidArgumentException If the shape is not an array or contains invalid values (ZMATRIX_ERR_INVALID_SHAPE).
     * @throws RuntimeException If allocation fails or another error occurs.
     */
    public static function ones(array $shape): ZTensor
    {
    }

    /**
     * Creates a new tensor filled with a constant scalar value, with the specified shape.
     *
     * @param array<int> $shape An array of integers defining the dimensions of the new tensor.
     * The dimensions must be non-negative.
     * @param float $value The scalar value to be used to fill the tensor.
     * @return ZTensor The new tensor filled with the constant value.
     * @throws InvalidArgumentException If the shape is not an array or contains invalid values (ZMATRIX_ERR_INVALID_SHAPE).
     * @throws RuntimeException If allocation fails or another error occurs.
     */
    public static function full(array $shape, float $value): ZTensor
    {
    }

    /**
     * Creates a 2D identity matrix (rank-2 tensor).
     *
     * Returns a square matrix of size $size x $size with 1s on the main diagonal and 0s elsewhere.
     *
     * @param int $size The size of the identity matrix (number of rows and columns). Must be positive.
     * @return ZTensor The identity matrix.
     * @throws InvalidArgumentException If $size is not positive.
     * @throws RuntimeException If allocation fails or another error occurs.
     */
    public static function identity(int $size): ZTensor
    {
    }

    /**
     * Creates a new tensor with uniformly distributed random values.
     *
     * The values are generated in the interval [$min, $max).
     *
     * @param array<int> $shape An array of integers defining the dimensions of the new tensor.
     * The dimensions must be non-negative (dimension 0 is allowed).
     * @param float $min The lower bound (inclusive) of the generation interval (default: 0.0).
     * @param float $max The upper bound (exclusive) of the generation interval (default: 1.0).
     * @return ZTensor The new tensor filled with random values.
     * @throws InvalidArgumentException If the shape is invalid (ZMATRIX_ERR_INVALID_SHAPE) or if $min > $max.
     * @throws RuntimeException If allocation fails or another error occurs.
     */
    public static function random(array $shape, float $min = 0.0, float $max = 1.0): ZTensor
    {
    }

    /**
     * Performs matrix multiplication (matrix product).
     *
     * Currently implemented only for 2D tensors (matrices).
     * Multiplies this tensor (A) by another tensor (B), resulting in C = A * B.
     * The number of columns in A must be equal to the number of rows in B.
     *
     * @param ZTensor|array<int|float> $other The matrix (2D tensor or PHP array) to be multiplied.
     * @return ZTensor A new tensor containing the result of the matrix multiplication.
     * @throws RuntimeException If the tensors are not 2D, if dimensions are incompatible
     * for multiplication (ZMATRIX_ERR_INCOMPATIBLE_DIMS), or if another internal error occurs (e.g., BLAS error).
     */
    public function matmul(ZTensor|array $other): ZTensor
    {
    }

    /**
     * Divides this tensor by another tensor or array, element-wise.
     *
     * @param ZTensor|float|int|array<int|float> $other Another tensor or array.
     * @return ZTensor
     * @throws RuntimeException
     */
    public function divide(ZTensor|array|float|int $other): ZTensor
    {
    }

    /**
     * Raises each element of this tensor to a power.
     *
     * @param float $exponent Exponent for the operation.
     * @return ZTensor
     * @throws RuntimeException
     */
    public function pow(float $exponent): ZTensor
    {
    }

    /**
     * Exponentiation: exp(x) for each element.
     *
     * @return ZTensor
     * @throws RuntimeException
     */
    public function exp(): ZTensor
    {
    }

    /**
     * Natural logarithm: log(x) for each element.
     *
     * @return ZTensor
     * @throws RuntimeException
     */
    public function log(): ZTensor
    {
    }

    /**
     * Square root: sqrt(x) for each element.
     *
     * @return ZTensor
     * @throws RuntimeException
     */
    public function sqrt(): ZTensor
    {
    }

    /**
     * Creates a new tensor with random values from a normal (Gaussian) distribution.
     *
     * @param array<int> $shape The shape of the tensor to be created.
     * @param float $mean The mean of the normal distribution (default: 0.0).
     * @param float $std_dev The standard deviation of the normal distribution (default: 1.0). Must be non-negative.
     * @return ZTensor The new tensor with normally distributed random values.
     * @throws InvalidArgumentException If the shape is invalid or std_dev is negative.
     * @throws RuntimeException If allocation fails.
     */
    public static function randn(array $shape, float $mean = 0.0, float $std_dev = 1.0): ZTensor
    {
    }

    /**
     * Creates a new 1D tensor with evenly spaced values within a given interval.
     *
     * Values are generated within the half-open interval [start, stop) (stop is not included).
     * Use cases:
     * - arange(stop): Values from 0 up to stop-1 with step 1.
     * - arange(start, stop): Values from start up to stop-1 with step 1.
     * - arange(start, stop, step): Values from start with increment step while < stop (or > stop if step is negative).
     *
     * @param float $start_or_stop If $stop is null, this is the 'stop' value and 'start' is 0. Otherwise, this is the 'start' value.
     * @param ?float $stop The end of the interval (exclusive). If null, $start_or_stop is used as stop.
     * @param float $step The spacing between values (default: 1.0). Cannot be zero.
     * @return ZTensor The new 1D tensor.
     * @throws InvalidArgumentException If $step is zero or the parameters result in an invalid interval.
     * @throws RuntimeException If allocation fails.
     */
    public static function arange(float $start_or_stop, ?float $stop = null, float $step = 1.0): ZTensor
    {
    }

    /**
     * Creates a new 1D tensor with N evenly spaced values between start and stop.
     *
     * @param float $start The starting value of the sequence.
     * @param float $stop The final value of the sequence.
     * @param int $num The number of samples to generate (default: 50). Must be non-negative.
     * @param bool $endpoint If true (default), 'stop' is the last value. Otherwise, it is not included.
     * @return ZTensor The new 1D tensor.
     * @throws InvalidArgumentException If $num is negative.
     * @throws RuntimeException If allocation fails.
     */
    public static function linspace(float $start, float $stop, int $num = 50, bool $endpoint = true): ZTensor
    {
    }

    /**
     * Creates a new 1D tensor with N logarithmically spaced values.
     *
     * The values are base^start to base^stop.
     *
     * @param float $start The starting exponent (base^start).
     * @param float $stop The final exponent (base^stop).
     * @param int $num The number of samples to generate (default: 50). Must be non-negative.
     * @param bool $endpoint If true (default), base^stop is the last value.
     * @param float $base The base of the logarithm (default: 10.0).
     * @return ZTensor The new 1D tensor.
     * @throws InvalidArgumentException If $num is negative.
     * @throws RuntimeException If allocation fails.
     */
    public static function logspace(float $start, float $stop, int $num = 50, bool $endpoint = true, float $base = 10.0): ZTensor
    {
    }

    /**
     * Creates a matrix (2D tensor) with ones on the k-th diagonal and zeros elsewhere.
     *
     * @param int $N Number of rows.
     * @param ?int $M Number of columns. If null, M = N (square matrix). (Default: null)
     * @param int $k Diagonal index: 0 for the main (default), positive for above, negative for below.
     * @return ZTensor The resulting matrix.
     * @throws InvalidArgumentException If N or M are negative.
     * @throws RuntimeException If allocation fails.
     */
    public static function eye(int $N, ?int $M = null, int $k = 0): ZTensor
    {
    }

    /**
     * Calculates the dot product of two tensors.
     *
     * - If both are 1D of the same size, returns the inner product (scalar float).
     * - If A is 2D and B is 1D (column vector), calculates A @ B (matrix-vector product, returns 1D tensor).
     * - If both are 2D, calculates A @ B (matrix multiplication, returns 2D tensor).
     *
     * @param ZTensor|array<int|float> $other The other tensor or array for the dot product.
     * @return ZTensor|float The result of the dot product.
     * @throws RuntimeException If shapes are incompatible for the supported operations.
     */
    public function dot(ZTensor|array $other): ZTensor|float
    {
    }

    /**
     * Retrieves an element at the specified indices.
     * @param array<int> $indices
     * @return float
     * @throws RuntimeException If the index is out of bounds or invalid.
     */
    public function key(array $indices): float
    {
    }

    /**
     * Adds a 1D vector [C] to each row of a 2D tensor [N×C], performing broadcasting on dimension 0.
     *
     * Example:
     * ```php
     * $mat = ZTensor::arr([[1, 2], [3, 4]]);
     * $bias = ZTensor::arr([10, 20]);
     * $res = $mat->broadcast($bias); // [[11, 22], [13, 24]]
     * ```
     *
     * @param ZTensor $bias 1D vector with shape [C] to be added to each row of the current tensor [N×C].
     * @return ZTensor A new tensor with the summed values.
     * @throws RuntimeException If shapes are incompatible or the tensor is not 2D.
     */
    public function broadcast(ZTensor $bias): ZTensor
    {
    }

    /**
     * Returns a new ZTensor where each element is 1.0 if the corresponding
     * element in $this is greater than in $other, or 0.0 otherwise.
     *
     * @param ZTensor|array $other A ZTensor or array of the same dimensions as $this.
     * @return ZTensor A new ZTensor with 1.0 or 0.0 values.
     */
    public function greater(ZTensor|array $other): ZTensor
    {
    }

    /**
     * Repeats the tensor vertically (multiplies the number of rows).
     *
     * Example:
     * ```
     * $a = Ztensor::arr([[1, 2], [3, 4]]);
     * Ztensor::tile($a, 3);
     * // → [[1,2],[3,4],[1,2],[3,4],[1,2],[3,4]]
     * ```
     *
     * @param ZTensor $tensor The tensor to be repeated.
     * @param int $times How many times to repeat (must be >= 1).
     * @return ZTensor A new tensor resulting from the repetition.
     */
    public static function tile(ZTensor $tensor, int $times): ZTensor
    {
    }
}
