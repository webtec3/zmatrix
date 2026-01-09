<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

// =============================================================================
// TENSOR CREATION AND INITIALIZATION
// =============================================================================

echo "=== CREATION AND INITIALIZATION ===\n";

// Constructor - create empty tensor
$empty = new ZTensor();
echo "Empty tensor: " . ($empty->isEmpty() ? "yes" : "no") . "\n";

// Constructor - create tensor from array
$tensor1d = new ZTensor([1, 2, 3, 4, 5]);
echo "1D Tensor: " . print_r($tensor1d->toArray(), true);

$tensor2d = new ZTensor([
    [1, 2, 3],
    [4, 5, 6]
]);
echo "2D Tensor: " . print_r($tensor2d->toArray(), true);

// safe() method - safe creation
$safe_tensor = ZTensor::safe([
    [1.5, 2.7],
    [3.1, 4.9]
]);
echo "Safe tensor: " . print_r($safe_tensor->shape(), true);

// arr() method - factory method
$arr_tensor = ZTensor::arr([
    [10, 20],
    [30, 40],
    [50, 60]
]);
echo "Tensor via arr(): " . print_r($arr_tensor->toArray(), true);

// copy() - deep copy
$original = ZTensor::arr([1, 2, 3]);
$copy = $original->copy();
echo "Original: " . print_r($original->toArray(), true);
echo "Copy: " . print_r($copy->toArray(), true);

// =============================================================================
// SPECIAL TENSORS
// =============================================================================

echo "\n=== SPECIAL TENSORS ===\n";

// zeros() - tensor filled with zeros
$zeros = ZTensor::zeros([2, 3]);
echo "Zeros (2x3): " . print_r($zeros->toArray(), true);

// ones() - tensor filled with ones
$ones = ZTensor::ones([3, 2]);
echo "Ones (3x2): " . print_r($ones->toArray(), true);

// full() - tensor filled with constant value
$full = ZTensor::full([2, 2], 7.5);
echo "Full (2x2, value 7.5): " . print_r($full->toArray(), true);

// identity() - identity matrix
$identity = ZTensor::identity(3);
echo "Identity (3x3): " . print_r($identity->toArray(), true);

// eye() - matrix with diagonal
$eye = ZTensor::eye(3, 4, 1); // 3 rows, 4 columns, upper diagonal
echo "Eye (3x4, k=1): " . print_r($eye->toArray(), true);

// =============================================================================
// SEQUENCE GENERATION
// =============================================================================

echo "\n=== SEQUENCE GENERATION ===\n";

// arange() - sequence with interval
$arange1 = ZTensor::arange(5); // 0 to 4
echo "Arange(5): " . print_r($arange1->toArray(), true);

$arange2 = ZTensor::arange(2, 8); // 2 to 7
echo "Arange(2, 8): " . print_r($arange2->toArray(), true);

$arange3 = ZTensor::arange(0, 10, 2.5); // 0, 2.5, 5.0, 7.5
echo "Arange(0, 10, 2.5): " . print_r($arange3->toArray(), true);

// linspace() - equally spaced values
$linspace = ZTensor::linspace(0, 1, 5);
echo "Linspace(0, 1, 5): " . print_r($linspace->toArray(), true);

// logspace() - logarithmically spaced values
$logspace = ZTensor::logspace(1, 3, 3); // 10^1, 10^2, 10^3
echo "Logspace(1, 3, 3): " . print_r($logspace->toArray(), true);

// =============================================================================
// RANDOM NUMBER GENERATION
// =============================================================================

echo "\n=== RANDOM NUMBERS ===\n";

// random() - uniform random values
$random = ZTensor::random([2, 3], 0.0, 10.0);
echo "Random (2x3, 0-10): shape " . print_r($random->shape(), true);

// randn() - normal distribution
$randn = ZTensor::randn([2, 2], 0.0, 1.0);
echo "Random normal (2x2): shape " . print_r($randn->shape(), true);

// =============================================================================
// BASIC ARITHMETIC OPERATIONS
// =============================================================================

echo "\n=== ARITHMETIC OPERATIONS ===\n";

$a = ZTensor::arr([
    [1, 2],
    [3, 4]
]);

$b = ZTensor::arr([
    [5, 6],
    [7, 8]
]);

// add() - element-wise addition
$sum = $a->add($b);
echo "Sum A + B: " . print_r($sum->toArray(), true);

// sub() - element-wise subtraction
$diff = $a->sub($b);
echo "Difference A - B: " . print_r($diff->toArray(), true);

// mul() - element-wise multiplication (Hadamard)
$product = $a->mul($b);
echo "Element-wise product: " . print_r($product->toArray(), true);

// divide() - element-wise division
$division = $a->divide($b);
echo "Division A / B: " . print_r($division->toArray(), true);

// scalarMultiply() - scalar multiplication
$scalar_mul = $a->scalarMultiply(2.5);
echo "A * 2.5: " . print_r($scalar_mul->toArray(), true);

// scalarDivide() - scalar division
$scalar_div = $a->scalarDivide(2.0);
echo "A / 2.0: " . print_r($scalar_div->toArray(), true);

// pow() - exponentiation
$power = $a->pow(2);
echo "A^2: " . print_r($power->toArray(), true);

// =============================================================================
// LINEAR ALGEBRA
// =============================================================================

echo "\n=== LINEAR ALGEBRA ===\n";

$matrix_a = ZTensor::arr([
    [1, 2, 3],
    [4, 5, 6]
]);

$matrix_b = ZTensor::arr([
    [7, 8],
    [9, 10],
    [11, 12]
]);

// matmul() - matrix multiplication
$matmul = $matrix_a->matmul($matrix_b);
echo "Matrix multiplication (2x3 × 3x2): " . print_r($matmul->toArray(), true);

// transpose() - transposition
$transpose = $matrix_a->transpose();
echo "Transposition of A: " . print_r($transpose->toArray(), true);

// dot() - dot product
$vec1 = ZTensor::arr([1, 2, 3]);
$vec2 = ZTensor::arr([4, 5, 6]);
$dot_product = $vec1->dot($vec2);
echo "Dot product: $dot_product\n";

// =============================================================================
// MATHEMATICAL FUNCTIONS
// =============================================================================

echo "\n=== MATHEMATICAL FUNCTIONS ===\n";

$math_tensor = ZTensor::arr([
    [1, 4, 9],
    [16, 25, 36]
]);

// abs() - absolute value
$negative = ZTensor::arr([-1, -2, 3, -4]);
$absolute = $negative->abs();
echo "Absolute value: " . print_r($absolute->toArray(), true);

// sqrt() - square root
$sqrt_result = $math_tensor->sqrt();
echo "Square root: " . print_r($sqrt_result->toArray(), true);

// exp() - exponential
$small_values = ZTensor::arr([0, 1, 2]);
$exp_result = $small_values->exp();
echo "Exponential: " . print_r($exp_result->toArray(), true);

// log() - natural logarithm
$log_values = ZTensor::arr([1, 2.718, 7.389]);
$log_result = $log_values->log();
echo "Natural logarithm: " . print_r($log_result->toArray(), true);

// =============================================================================
// ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
// =============================================================================

echo "\n=== ACTIVATION FUNCTIONS ===\n";

$activation_data = ZTensor::arr([
    [-2, -1, 0],
    [1, 2, 3]
]);

// sigmoid() - sigmoid function
$sigmoid_tensor = $activation_data->copy();
$sigmoid_result = $sigmoid_tensor->sigmoid();
echo "Sigmoid: " . print_r($sigmoid_result->toArray(), true);

// sigmoidDerivative() - sigmoid derivative
$sigmoid_deriv = $sigmoid_result->copy()->sigmoidDerivative();
echo "Sigmoid derivative: " . print_r($sigmoid_deriv->toArray(), true);

// relu() - ReLU
$relu_tensor = $activation_data->copy();
$relu_result = $relu_tensor->relu();
echo "ReLU: " . print_r($relu_result->toArray(), true);

// reluDerivative() - ReLU derivative
$relu_deriv_tensor = $activation_data->copy();
$relu_deriv = $relu_deriv_tensor->reluDerivative();
echo "ReLU derivative: " . print_r($relu_deriv->toArray(), true);

// leakyRelu() - Leaky ReLU
$leaky_relu_tensor = $activation_data->copy();
$leaky_relu = $leaky_relu_tensor->leakyRelu(0.1);
echo "Leaky ReLU (α=0.1): " . print_r($leaky_relu->toArray(), true);

// leakyReluDerivative() - Leaky ReLU derivative
$leaky_deriv_tensor = $activation_data->copy();
$leaky_deriv = $leaky_deriv_tensor->leakyReluDerivative(0.1);
echo "Leaky ReLU derivative: " . print_r($leaky_deriv->toArray(), true);

// tanh() - hyperbolic tangent
$tanh_tensor = $activation_data->copy();
$tanh_result = $tanh_tensor->tanh();
echo "Tanh: " . print_r($tanh_result->toArray(), true);

// tanhDerivative() - tanh derivative
$tanh_deriv = $tanh_result->copy()->tanhDerivative();
echo "Tanh derivative: " . print_r($tanh_deriv->toArray(), true);

// softmax() - softmax function
$softmax_data = ZTensor::arr([
    [1, 2, 3],
    [4, 5, 6]
]);
$softmax_result = $softmax_data->softmax();
echo "Softmax: " . print_r($softmax_result->toArray(), true);

// softmaxDerivative() - softmax derivative
$softmax_deriv = $softmax_result->copy()->softmaxDerivative();
echo "Softmax derivative: " . print_r($softmax_deriv->toArray(), true);

// =============================================================================
// STATISTICS AND AGGREGATIONS
// =============================================================================

echo "\n=== STATISTICS ===\n";

$stats_tensor = ZTensor::arr([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

// sumtotal() - total sum
$total_sum = $stats_tensor->sumtotal();
echo "Total sum: $total_sum\n";

// mean() - average
$mean_value = $stats_tensor->mean();
echo "Mean: $mean_value\n";

// min() - minimum value
$min_value = $stats_tensor->min();
echo "Minimum: $min_value\n";

// max() - maximum value
$max_value = $stats_tensor->max();
echo "Maximum: $max_value\n";

// std() - standard deviation
$std_value = $stats_tensor->std();
echo "Standard deviation: $std_value\n";

// sum() - sum along axes
$sum_result = ZTensor::zeros([3]);
$stats_tensor->sum($sum_result, 1); // sum along axis 1
echo "Sum by row: " . print_r($sum_result->toArray(), true);

// =============================================================================
// COMPARISON AND CLIPPING OPERATIONS
// =============================================================================

echo "\n=== COMPARISON AND CLIPPING ===\n";

$comp_a = ZTensor::arr([1, 5, 3, 8, 2]);
$comp_b = ZTensor::arr([2, 4, 3, 6, 9]);

// greater() - greater than comparison
$greater_result = $comp_a->greater($comp_b);
echo "A > B: " . print_r($greater_result->toArray(), true);

// clip() - value clipping
$clip_data = ZTensor::arr([[-2, 5, 10], [0, 15, 1]]);
$clipped = ZTensor::clip($clip_data, 0, 10);
echo "Clipped (0-10): " . print_r($clipped->toArray(), true);

// minimum() - element-wise minimum with scalar
$min_result = ZTensor::minimum($comp_a, 4.0);
echo "Minimum with 4.0: " . print_r($min_result->toArray(), true);

// maximum() - element-wise maximum with scalar
$max_result = ZTensor::maximum($comp_a, 4.0);
echo "Maximum with 4.0: " . print_r($max_result->toArray(), true);

// =============================================================================
// SHAPE MANIPULATION
// =============================================================================

echo "\n=== SHAPE MANIPULATION ===\n";

$shape_tensor = ZTensor::arr([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]);

// shape() - get shape
$current_shape = $shape_tensor->shape();
echo "Current shape: " . print_r($current_shape, true);

// ndim() - number of dimensions
$dimensions = $shape_tensor->ndim();
echo "Number of dimensions: $dimensions\n";

// size() - total number of elements
$total_size = $shape_tensor->size();
echo "Total elements: $total_size\n";

// isEmpty() - check if empty
$is_empty = $shape_tensor->isEmpty();
echo "Is empty: " . ($is_empty ? "yes" : "no") . "\n";

// reshape() - reshape
$reshaped = $shape_tensor->reshape([4, 2]);
echo "Reshaped (4x2): " . print_r($reshaped->toArray(), true);

// toArray() - convert to PHP array
$php_array = $shape_tensor->toArray();
echo "As PHP array: " . print_r($php_array, true);

// key() - access element by indices
$element = $shape_tensor->key([1, 2]);
echo "Element [1,2]: $element\n";

// =============================================================================
// SPECIAL OPERATIONS
// =============================================================================

echo "\n=== SPECIAL OPERATIONS ===\n";

// broadcast() - bias broadcasting
$matrix_2d = ZTensor::arr([
    [1, 2],
    [3, 4],
    [5, 6]
]);
$bias_1d = ZTensor::arr([10, 20]);
$broadcasted = $matrix_2d->broadcast($bias_1d);
echo "Broadcasting bias: " . print_r($broadcasted->toArray(), true);

// tile() - tensor repetition
$tile_tensor = ZTensor::arr([
    [1, 2],
    [3, 4]
]);
$tiled = ZTensor::tile($tile_tensor, 3);
echo "Tiled (3x): " . print_r($tiled->toArray(), true);

echo "\n=== EXAMPLES COMPLETED ===\n";
echo "All public methods of the ZTensor class have been demonstrated!\n";
