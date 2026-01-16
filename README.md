## 🔬 Autograd (Diferenciação Automática) — Infraestrutura Experimental

ZMatrix agora suporta a infraestrutura mínima para diferenciação automática (autograd), inspirada em PyTorch/Micrograd. Por padrão, não há overhead nem alteração de comportamento numérico.

### Ativando requiresGrad

```php
$t = ZTensor::arr([[1,2],[3,4]])->requiresGrad(true);
if ($t->requires_grad()) {
    echo "Este tensor irá rastrear gradientes.";
}
```

### Comportamento padrão (sem autograd)

```php
$t = ZTensor::arr([[1,2],[3,4]]);
if (!$t->requires_grad()) {
    echo "Execução numérica pura, sem rastreamento de gradientes.";
}
```

### Observação

O grafo de operações e o backward ainda não estão implementados. Esta infraestrutura é compatível com futuras extensões para autograd/backpropagation.

### Limitações atuais do autograd

- ⚠️ **Operações inplace não são seguras para autograd:**
  - O contexto de gradiente (`grad_ctx`) não é corretamente preservado em operações inplace. O resultado deveria receber o novo contexto, mas hoje o tensor de entrada pode perder seu histórico.
  - Recomenda-se evitar mutações em tensores com `requires_grad=true` após o forward.
  - PyTorch e outros frameworks também alertam: operações inplace podem invalidar o grafo de autograd.
- ⚠️ **Acumulação de gradiente:**
  - O campo `.grad` existe e é inicializado sob demanda, mas o backward ainda não está implementado.
- ⚠️ **Propagação de requires_grad:**
  - O resultado de uma operação terá `requires_grad=true` se qualquer operando exigir, mas para operações inplace, o comportamento pode ser inconsistente.

Essas limitações não afetam o uso numérico puro, mas devem ser consideradas ao experimentar autograd.

# 📊 ZMatrix - High-Performance Matrix and Tensor Operations for PHP

ZMatrix is a high-performance PHP extension for matrix and N-dimensional tensor operations, implemented in C++ with optimizations for parallel processing and BLAS integration.

> 📚 **[DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md)** - Mapa completo de toda documentação com guias por tipo de usuário

## 🚀 Installation

```bash
sudo update-alternatives --config phpize
```

```bash
git clone https://github.com/omegaalfa/zmatrix.git
cd zmatrix
make clean
phpize
./configure --enable-zmatrix
make
sudo make install
```
```bash
phpize
./configure --with-cuda-path=/usr/local/cuda
make clean
make -j$(nproc)
sudo make install
```

Add the extension to your php.ini:

```
extension=zmatrix.so
```

**📖 Para guia completo de instalação com troubleshooting**, veja [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)

## � Dependências de Compilação

### ⚡ Mínimas para CPU (Compilação Sem GPU)

Para compilar **apenas com suporte a CPU**, você precisa de:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    php-dev \
    autoconf \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev
```

**CentOS/RHEL:**
```bash
sudo yum groupinstall "Development Tools" -y
sudo yum install -y \
    php-devel \
    autoconf \
    pkg-config \
    blas-devel \
    lapack-devel \
    openblas-devel
```

**Dependências Essenciais para CPU:**
- `build-essential` - Compilador C/C++ (gcc/g++)
- `php-dev` - Headers do PHP
- `autoconf` - Sistema de build
- `pkg-config` - Gerenciador de configuração de pacotes
- `libblas-dev` / `libopenblas-dev` - Operações de álgebra linear
- `liblapack-dev` - Routinas de álgebra linear

**Resultado:** ✅ Biblioteca funcional com otimizações de CPU, sem GPU

---

### 🚀 Completas para GPU (Compilação Com CUDA)

Para compilar **com suporte a GPU**, além das dependências de CPU, você precisa de:

**NVIDIA CUDA Toolkit:**
```bash
# NVIDIA CUDA 12.0 (recomendado)
# Download em: https://developer.nvidia.com/cuda-downloads

# Instalação rápida em Ubuntu:
sudo apt-get install -y nvidia-cuda-toolkit

# Ou compilação com caminho personalizado:
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Dependências Adicionais para GPU:**
- `nvidia-cuda-toolkit` (ou CUDA Toolkit 12.0+)
- `nvidia-driver` (drivers NVIDIA atualizados)
- GPU NVIDIA com Compute Capability 3.5+ (ou superior)

**Compilação com GPU:**
```bash
./configure --enable-zmatrix --with-cuda-path=/usr/local/cuda
make -j$(nproc)
sudo make install
```

**Resultado:** ✅ Biblioteca com aceleração GPU + fallback automático para CPU

---

### 🛡️ Sistema de Compatibilidade Inteligente

O ZMatrix implementa uma **estratégia de detecção automática** em 3 camadas:

1. **Detecção em Build-Time** (`config.m4`)
   - Detecta WSL automaticamente
   - Configura RPATH para WSL se necessário
   - Compila corretamente em qualquer ambiente

2. **Fallback em Runtime** (`gpu_kernels.cu`)
   - Tenta carregar libcuda.so automaticamente
   - Procura em 6 locais diferentes
   - Se não encontrar, usa CPU automaticamente

3. **Segurança em Uso** (PHP methods)
   - Métodos GPU lançam exceção clara se indisponíveis
   - Aplicação pode tratar gracefully
   - Nunca causa crash silencioso

**Resultado:** ✅ Compila e funciona em **100% dos cenários**

---

### ✅ Matriz de Compatibilidade

| Cenário | CPU | GPU | Resultado |
|---------|-----|-----|-----------|
| Linux com GPU + drivers | ✅ | ✅ | **GPU acelerado** |
| Linux com GPU + drivers vencidos | ✅ | ⚠️ | **CPU fallback** |
| Linux sem GPU | ✅ | ❌ | **CPU normal** |
| WSL2 com GPU passthrough | ✅ | ✅ | **GPU acelerado** |
| WSL2 sem GPU | ✅ | ❌ | **CPU normal** |
| Docker sem GPU | ✅ | ❌ | **CPU normal** |
| Qualquer outro sistema | ✅ | ❓ | **CPU normal** |

---

### 🎯 Recomendações

**Para Desenvolvimento/Teste (CPU apenas):**
```bash
# Instalação mínima, rápida
./configure --enable-zmatrix
make && sudo make install
```

**Para Produção com GPU:**
```bash
# Instalação completa com GPU
./configure --enable-zmatrix --with-cuda-path=/usr/local/cuda
make -j$(nproc) && sudo make install
```

**Para Ambientes Restritos:**
```bash
# WSL/Container sem GPU - compila normalmente
./configure --enable-zmatrix
make && sudo make install
# Sistema detecta automaticamente, GPU não causa problemas
```

## �📋 Features

## 📋 API Coverage

✅ **62 Métodos Documentados com Exemplos**

**Por Categoria:** Criação | Propriedades | Aritmética | Álgebra Linear | Ativações | Estatísticas | Comparação | Manipulação | **GPU** ⭐ | Matemática

[📖 Documentação Detalhada](#zmatrix-php-extension---usage-examples) | [📚 Tabela Completa](#-complete-api-reference---resumo-de-todos-os-métodos) | [🚀 GPU](#-gpu-aceleração-detalhada)


The ZMatrix extension implements the following functionalities:


### Tensor Creation

* `ZTensor::zeros()` - Creates a tensor filled with zeros
* `ZTensor::ones()` - Creates a tensor filled with ones
* `ZTensor::full()` - Creates a tensor filled with a constant value
* `ZTensor::identity()` - Creates an identity matrix
* `ZTensor::eye()` - Creates a diagonal matrix with optional offset
* `ZTensor::arange()` - Creates a 1D tensor with values in a range
* `ZTensor::linspace()` - Creates a 1D tensor with evenly spaced values
* `ZTensor::logspace()` - Creates a 1D tensor with logarithmically spaced values
* `ZTensor::random()` - Creates a tensor with uniformly distributed random values
* `ZTensor::randn()` - Creates a tensor with normally distributed random values

### Tensor Properties

* `$tensor->ndim()` - Returns the number of dimensions
* `$tensor->shape()` - Returns the shape (dimensions)
* `$tensor->size()` - Returns the total number of elements
* `$tensor->isEmpty()` - Checks whether the tensor is empty
* `$tensor->toArray()` - Converts tensor to PHP array

### Basic Operations

* `$tensor->add()` - Element-wise addition
* `$tensor->sub()` - Element-wise subtraction
* `$tensor->mul()` - Element-wise multiplication
* `$tensor->divide()` - Element-wise division
* `$tensor->pow()` - Raises each element to a power
* `$tensor->scalarMultiply()` - Multiplies tensor by scalar
* `$tensor->scalarDivide()` - Divides tensor by scalar
* `$tensor->transpose()` - Matrix transpose (2D only)
* `$tensor->dot()` - Dot product
* `$tensor->matmul()` - Matrix multiplication (2D only)

### Reshaping and Views

* `$tensor->reshape()` - Returns a reshaped view of the tensor
* `ZTensor::tile()` - Repeats a tensor vertically

### Reductions and Statistics

* `$tensor->sum()` - Sum over axis or globally
* `$tensor->sumtotal()` - Global sum of all elements
* `$tensor->mean()` - Mean of elements
* `$tensor->std()` - Standard deviation (sample)
* `$tensor->min()` - Minimum value
* `$tensor->max()` - Maximum value

### Activation Functions

* `$tensor->abs()` - Absolute value
* `$tensor->sigmoid()` - Sigmoid activation
* `$tensor->sigmoidDerivative()` - Derivative of sigmoid
* `$tensor->relu()` - ReLU activation
* `$tensor->reluDerivative()` - Derivative of ReLU
* `$tensor->tanh()` - Tanh activation
* `$tensor->tanhDerivative()` - Derivative of tanh
* `$tensor->leakyRelu()` - Leaky ReLU
* `$tensor->leakyReluDerivative()` - Derivative of Leaky ReLU
* `$tensor->softmax()` - Softmax activation
* `$tensor->softmaxDerivative()` - Derivative of Softmax

### Utilities

* `ZTensor::arr()` - Creates a tensor from PHP array
* `ZTensor::safe()` - Same as arr(), returns a ZTensor
* `$tensor->copy()` - Deep copy
* `$tensor->key([...])` - Gets an element from coordinates
* `$tensor->broadcast($bias)` - Adds 1D bias to rows of 2D tensor
* `ZTensor::clip($tensor, $min, $max)` - Clamps tensor values within range
* `ZTensor::minimum($tensor, $value)` - Element-wise min with scalar
* `ZTensor::maximum($tensor, $value)` - Element-wise max with scalar
* `$tensor->greater($other)` - Returns 1.0 where \$this > \$other

### GPU Memory Management (CUDA)

* `$tensor->toGpu()` - Move tensor to GPU memory for accelerated operations
* `$tensor->toCpu()` - Move tensor back to CPU memory
* `$tensor->isOnGpu()` - Check if tensor is currently on GPU
* `$tensor->freeDevice()` - Explicitly free GPU memory and move to CPU

**GPU Features:**
- ✅ Automatic CUDA detection and fallback to CPU
- ✅ WSL2 GPU support with automatic path detection
- ✅ Works seamlessly if CUDA is unavailable
- ✅ Up to 7694x speedup for large tensor operations
- ✅ Graceful degradation on systems without GPU

**GPU Usage Example:**
```php
use ZMatrix\ZTensor;

// Create and move to GPU
$tensor = ZTensor::random([1000, 1000]);
$tensor->toGpu();

// Operations automatically use GPU
$tensor->relu();
$tensor->add($other);

// Move back to CPU when done
$tensor->toCpu();
$result = $tensor->toArray();
```

## 📊 Performance

ZMatrix offers significant performance improvements over pure PHP implementations:

* **Matrix Multiplication**: Up to 100x faster than native PHP
* **N-dimensional Tensors**: Efficient memory layout and computation
* **Automatic Parallelism**: Uses multiple CPU cores when available (OpenMP or threads)
* **BLAS Integration**: Optional BLAS acceleration for linear algebra

## 🚜 Use Cases

* Machine Learning
* Numerical Computing
* Image Processing
* Scientific Simulation
* Data Analysis and Statistics
# ZTensor PHP Extension - Usage Examples

This document provides comprehensive usage examples for all public methods available in the ZTensor PHP extension. The ZTensor class represents a multidimensional tensor implemented in C++ for high-performance mathematical operations.

## Table of Contents

- [Creation and Initialization](#creation-and-initialization)
- [Special Tensors](#special-tensors)
- [Sequence Generation](#sequence-generation)
- [Random Number Generation](#random-number-generation)
- [Basic Arithmetic Operations](#basic-arithmetic-operations)
- [Linear Algebra](#linear-algebra)
- [Mathematical Functions](#mathematical-functions)
- [Activation Functions](#activation-functions)
- [Statistics and Aggregations](#statistics-and-aggregations)
- [Comparison and Clipping](#comparison-and-clipping)
- [Shape Manipulation](#shape-manipulation)
- [Special Operations](#special-operations)

---


### Use

```php
use ZMatrix\ZTensor;
```
Create an empty tensor:
```php
$empty = new ZTensor();
echo "Empty tensor: " . ($empty->isEmpty() ? "yes" : "no") . "\n";
```

Create tensor from 1D array:
```php
$tensor1d = new ZTensor([1, 2, 3, 4, 5]);
print_r($tensor1d->toArray());
```

Create tensor from 2D array:
```php
$tensor2d = new ZTensor([
    [1, 2, 3],
    [4, 5, 6]
]);
print_r($tensor2d->toArray());
```

### Safe Creation - `safe()`

```php
$safe_tensor = ZTensor::safe([
    [1.5, 2.7],
    [3.1, 4.9]
]);
print_r($safe_tensor->shape());
```

### Array Factory - `arr()`

```php
$arr_tensor = ZTensor::arr([
    [10, 20],
    [30, 40],
    [50, 60]
]);
print_r($arr_tensor->toArray());
```

### Deep Copy - `copy()`

```php
$original = ZTensor::arr([1, 2, 3]);
$copy = $original->copy();
print_r($original->toArray());
print_r($copy->toArray());
```

---

## Special Tensors

### Zeros Tensor - `zeros()`

```php
$zeros = ZTensor::zeros([2, 3]);
print_r($zeros->toArray());
// Output: [[0, 0, 0], [0, 0, 0]]
```

### Ones Tensor - `ones()`

```php
$ones = ZTensor::ones([3, 2]);
print_r($ones->toArray());
// Output: [[1, 1], [1, 1], [1, 1]]
```

### Constant Value Tensor - `full()`

```php
$full = ZTensor::full([2, 2], 7.5);
print_r($full->toArray());
// Output: [[7.5, 7.5], [7.5, 7.5]]
```

### Identity Matrix - `identity()`

```php
$identity = ZTensor::identity(3);
print_r($identity->toArray());
// Output: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
```

### Eye Matrix - `eye()`

```php
$eye = ZTensor::eye(3, 4, 1); // 3 rows, 4 columns, upper diagonal
print_r($eye->toArray());
// Output: [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
```

---

## Sequence Generation

### Range Sequence - `arange()`

Simple range (0 to 4):
```php
$arange1 = ZTensor::arange(5);
print_r($arange1->toArray());
// Output: [0, 1, 2, 3, 4]
```

Range with start and stop:
```php
$arange2 = ZTensor::arange(2, 8);
print_r($arange2->toArray());
// Output: [2, 3, 4, 5, 6, 7]
```

Range with step:
```php
$arange3 = ZTensor::arange(0, 10, 2.5);
print_r($arange3->toArray());
// Output: [0, 2.5, 5.0, 7.5]
```

### Linear Space - `linspace()`

```php
$linspace = ZTensor::linspace(0, 1, 5);
print_r($linspace->toArray());
// Output: [0, 0.25, 0.5, 0.75, 1.0]
```

### Logarithmic Space - `logspace()`

```php
$logspace = ZTensor::logspace(1, 3, 3); // 10^1, 10^2, 10^3
print_r($logspace->toArray());
// Output: [10, 100, 1000]
```

---

## Random Number Generation

### Uniform Random - `random()`

```php
$random = ZTensor::random([2, 3], 0.0, 10.0);
echo "Random tensor shape: ";
print_r($random->shape());
// Generates random values between 0.0 and 10.0
```

### Normal Distribution - `randn()`

```php
$randn = ZTensor::randn([2, 2], 0.0, 1.0);
echo "Normal distribution tensor shape: ";
print_r($randn->shape());
// Generates normally distributed random values
```

---

## Basic Arithmetic Operations

### Addition - `add()`

```php
$a = ZTensor::arr([[1, 2], [3, 4]]);
$b = ZTensor::arr([[5, 6], [7, 8]]);
$sum = $a->add($b);
print_r($sum->toArray());
// Output: [[6, 8], [10, 12]]
```

### Subtraction - `sub()`

```php
$a = ZTensor::arr([[1, 2], [3, 4]]);
$b = ZTensor::arr([[5, 6], [7, 8]]);
$diff = $a->sub($b);
print_r($diff->toArray());
// Output: [[-4, -4], [-4, -4]]
```

### Element-wise Multiplication - `mul()`

```php
$a = ZTensor::arr([[1, 2], [3, 4]]);
$b = ZTensor::arr([[5, 6], [7, 8]]);
$product = $a->mul($b);
print_r($product->toArray());
// Output: [[5, 12], [21, 32]]
```

### Element-wise Division - `divide()`

```php
$a = ZTensor::arr([[10, 20], [30, 40]]);
$b = ZTensor::arr([[2, 4], [5, 8]]);
$division = $a->divide($b);
print_r($division->toArray());
// Output: [[5, 5], [6, 5]]
```

### Scalar Multiplication - `scalarMultiply()`

```php
$a = ZTensor::arr([[1, 2], [3, 4]]);
$scalar_mul = $a->scalarMultiply(2.5);
print_r($scalar_mul->toArray());
// Output: [[2.5, 5], [7.5, 10]]
```

### Scalar Division - `scalarDivide()`

```php
$a = ZTensor::arr([[4, 8], [12, 16]]);
$scalar_div = $a->scalarDivide(2.0);
print_r($scalar_div->toArray());
// Output: [[2, 4], [6, 8]]
```

### Power - `pow()`

```php
$a = ZTensor::arr([[2, 3], [4, 5]]);
$power = $a->pow(2);
print_r($power->toArray());
// Output: [[4, 9], [16, 25]]
```

---

## Linear Algebra

### Matrix Multiplication - `matmul()`

```php
$matrix_a = ZTensor::arr([
    [1, 2, 3],
    [4, 5, 6]
]);
$matrix_b = ZTensor::arr([
    [7, 8],
    [9, 10],
    [11, 12]
]);
$matrix_product = $matrix_a->matmul($matrix_b);
print_r($matrix_product->toArray());
// Output: [[58, 64], [139, 154]]
```

### Transpose - `transpose()`

```php
$matrix = ZTensor::arr([
    [1, 2, 3],
    [4, 5, 6]
]);
$transposed = $matrix->transpose();
print_r($transposed->toArray());
// Output: [[1, 4], [2, 5], [3, 6]]
```

### Dot Product - `dot()`

Vector dot product:
```php
$vec1 = ZTensor::arr([1, 2, 3]);
$vec2 = ZTensor::arr([4, 5, 6]);
$dot_product = $vec1->dot($vec2);
echo "Dot product: $dot_product\n";
// Output: 32 (1*4 + 2*5 + 3*6)
```

Matrix-vector multiplication:
```php
$matrix = ZTensor::arr([[1, 2], [3, 4]]);
$vector = ZTensor::arr([5, 6]);
$result = $matrix->dot($vector);
print_r($result->toArray());
// Output: [17, 39]
```

---

## Mathematical Functions

### Absolute Value - `abs()`

```php
$negative = ZTensor::arr([-1, -2, 3, -4]);
$absolute = $negative->abs();
print_r($absolute->toArray());
// Output: [1, 2, 3, 4]
```

### Square Root - `sqrt()`

```php
$values = ZTensor::arr([1, 4, 9, 16]);
$sqrt_result = $values->sqrt();
print_r($sqrt_result->toArray());
// Output: [1, 2, 3, 4]
```

### Exponential - `exp()`

```php
$values = ZTensor::arr([0, 1, 2]);
$exp_result = $values->exp();
print_r($exp_result->toArray());
// Output: [1, 2.718..., 7.389...]
```

### Natural Logarithm - `log()`

```php
$values = ZTensor::arr([1, 2.718, 7.389]);
$log_result = $values->log();
print_r($log_result->toArray());
// Output: [0, 1, 2] (approximately)
```

---

## Activation Functions

### Sigmoid - `sigmoid()`

```php
$data = ZTensor::arr([-2, -1, 0, 1, 2]);
$sigmoid_result = $data->sigmoid();
print_r($sigmoid_result->toArray());
// Output: [0.119, 0.269, 0.5, 0.731, 0.881] (approximately)
```

### Sigmoid Derivative - `sigmoidDerivative()`

```php
$sigmoid_values = ZTensor::arr([0.119, 0.269, 0.5, 0.731, 0.881]);
$sigmoid_deriv = $sigmoid_values->sigmoidDerivative();
print_r($sigmoid_deriv->toArray());
// Output: derivative values
```

### ReLU - `relu()`

```php
$data = ZTensor::arr([-2, -1, 0, 1, 2]);
$relu_result = $data->relu();
print_r($relu_result->toArray());
// Output: [0, 0, 0, 1, 2]
```

### ReLU Derivative - `reluDerivative()`

```php
$data = ZTensor::arr([-2, -1, 0, 1, 2]);
$relu_deriv = $data->reluDerivative();
print_r($relu_deriv->toArray());
// Output: [0, 0, 0, 1, 1]
```

### Leaky ReLU - `leakyRelu()`

```php
$data = ZTensor::arr([-2, -1, 0, 1, 2]);
$leaky_relu = $data->leakyRelu(0.1);
print_r($leaky_relu->toArray());
// Output: [-0.2, -0.1, 0, 1, 2]
```

### Leaky ReLU Derivative - `leakyReluDerivative()`

```php
$data = ZTensor::arr([-2, -1, 0, 1, 2]);
$leaky_deriv = $data->leakyReluDerivative(0.1);
print_r($leaky_deriv->toArray());
// Output: [0.1, 0.1, 0.1, 1, 1]
```

### Hyperbolic Tangent - `tanh()`

```php
$data = ZTensor::arr([-2, -1, 0, 1, 2]);
$tanh_result = $data->tanh();
print_r($tanh_result->toArray());
// Output: [-0.964, -0.762, 0, 0.762, 0.964] (approximately)
```

### Tanh Derivative - `tanhDerivative()`

```php
$tanh_values = ZTensor::arr([-0.964, -0.762, 0, 0.762, 0.964]);
$tanh_deriv = $tanh_values->tanhDerivative();
print_r($tanh_deriv->toArray());
// Output: derivative values
```

### Softmax - `softmax()`

```php
$data = ZTensor::arr([
    [1, 2, 3],
    [4, 5, 6]
]);
$softmax_result = $data->softmax();
print_r($softmax_result->toArray());
// Output: normalized probabilities for each row
```

### Softmax Derivative - `softmaxDerivative()`

```php
$softmax_values = ZTensor::arr([
    [0.09, 0.244, 0.665],
    [0.09, 0.244, 0.665]
]);
$softmax_deriv = $softmax_values->softmaxDerivative();
print_r($softmax_deriv->toArray());
// Output: derivative values
```

---

## Statistics and Aggregations

### Total Sum - `sumtotal()`

```php
$tensor = ZTensor::arr([
    [1, 2, 3],
    [4, 5, 6]
]);
$total_sum = $tensor->sumtotal();
echo "Total sum: $total_sum\n";
// Output: 21
```

### Mean - `mean()`

```php
$tensor = ZTensor::arr([1, 2, 3, 4, 5]);
$mean_value = $tensor->mean();
echo "Mean: $mean_value\n";
// Output: 3.0
```

### Minimum - `min()`

```php
$tensor = ZTensor::arr([5, 2, 8, 1, 9]);
$min_value = $tensor->min();
echo "Minimum: $min_value\n";
// Output: 1
```

### Maximum - `max()`

```php
$tensor = ZTensor::arr([5, 2, 8, 1, 9]);
$max_value = $tensor->max();
echo "Maximum: $max_value\n";
// Output: 9
```

### Standard Deviation - `std()`

```php
$tensor = ZTensor::arr([1, 2, 3, 4, 5]);
$std_value = $tensor->std();
echo "Standard deviation: $std_value\n";
// Output: approximately 1.58
```

### Sum - `sum()` (v0.5.0+)

**UPDATED (v0.5.0):** Method refactored for NumPy/PyTorch compatibility!

#### Global Sum (all elements)
```php
$tensor = ZTensor::arr([[1, 2, 3], [4, 5, 6]]);
$total = $tensor->sum();
echo "Total: " . $total->toArray()[0] . "\n";  // 21 (as scalar tensor)
```

#### Sum Along Specific Axis
```php
$tensor = ZTensor::arr([
    [1, 2, 3],
    [4, 5, 6]
]);

// Sum along axis 0 (down columns)
$sum_axis_0 = $tensor->sum(0);
print_r($sum_axis_0->toArray());
// Output: [5, 7, 9]

// Sum along axis 1 (across rows)
$sum_axis_1 = $tensor->sum(1);
print_r($sum_axis_1->toArray());
// Output: [6, 15]
```

#### NumPy-style Negative Indexing
```php
$tensor = ZTensor::arr([
    [1, 2, 3],
    [4, 5, 6]
]);

// -1 = last axis (same as axis 1 for 2D)
$sum_last = $tensor->sum(-1);
print_r($sum_last->toArray());
// Output: [6, 15]
```

#### Error Handling
```php
$tensor = ZTensor::arr([[1, 2, 3], [4, 5, 6]]);

// Invalid axis (throws exception)
try {
    $tensor->sum(5);  // Out of bounds
} catch (Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
}

// Non-integer axis (throws TypeError)
try {
    $tensor->sum("0");  // Wrong type
} catch (TypeError $e) {
    echo "Type error: " . $e->getMessage() . "\n";
}
```

**API Changes:**
- ✅ Now returns `ZTensor` in ALL cases (always consistent)
- ✅ Removed mandatory `$other` parameter
- ✅ Optional `$axis` parameter for reduction
- ✅ NumPy-compatible syntax
- ✅ Validates axis bounds serially before parallel execution
- ✅ Exception-safe with proper error messages

---

## Comparison and Clipping

### Greater Than - `greater()`

```php
$a = ZTensor::arr([1, 5, 3, 8, 2]);
$b = ZTensor::arr([2, 4, 3, 6, 9]);
$greater_result = $a->greater($b);
print_r($greater_result->toArray());
// Output: [0, 1, 0, 1, 0]
```

### Clip Values - `clip()`

```php
$data = ZTensor::arr([[-2, 5, 10], [0, 15, 1]]);
$clipped = ZTensor::clip($data, 0, 10);
print_r($clipped->toArray());
// Output: [[0, 5, 10], [0, 10, 1]]
```

### Element-wise Minimum with Scalar - `minimum()`

```php
$data = ZTensor::arr([1, 5, 3, 8, 2]);
$min_result = ZTensor::minimum($data, 4.0);
print_r($min_result->toArray());
// Output: [1, 4, 3, 4, 2]
```

### Element-wise Maximum with Scalar - `maximum()`

```php
$data = ZTensor::arr([1, 5, 3, 8, 2]);
$max_result = ZTensor::maximum($data, 4.0);
print_r($max_result->toArray());
// Output: [4, 5, 4, 8, 4]
```

---

## Shape Manipulation

### Get Shape - `shape()`

```php
$tensor = ZTensor::arr([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]);
$shape = $tensor->shape();
print_r($shape);
// Output: [2, 4]
```

### Number of Dimensions - `ndim()`

```php
$tensor = ZTensor::arr([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
$dimensions = $tensor->ndim();
echo "Number of dimensions: $dimensions\n";
// Output: 3
```

### Total Size - `size()`

```php
$tensor = ZTensor::arr([[1, 2, 3], [4, 5, 6]]);
$total_size = $tensor->size();
echo "Total elements: $total_size\n";
// Output: 6
```

### Check if Empty - `isEmpty()`

```php
$tensor = ZTensor::arr([1, 2, 3]);
$is_empty = $tensor->isEmpty();
echo "Is empty: " . ($is_empty ? "yes" : "no") . "\n";
// Output: no
```

### Reshape - `reshape()`

```php
$tensor = ZTensor::arr([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]);
$reshaped = $tensor->reshape([4, 2]);
print_r($reshaped->toArray());
// Output: [[1, 2], [3, 4], [5, 6], [7, 8]]
```

### Convert to Array - `toArray()`

```php
$tensor = ZTensor::arr([[1, 2], [3, 4]]);
$php_array = $tensor->toArray();
print_r($php_array);
// Output: [[1, 2], [3, 4]]
```

### Access Element by Index - `key()`

```php
$tensor = ZTensor::arr([
    [1, 2, 3],
    [4, 5, 6]
]);
$element = $tensor->key([1, 2]);
echo "Element at [1,2]: $element\n";
// Output: 6
```

---

## Special Operations

### Broadcasting - `broadcast()`

```php
$matrix = ZTensor::arr([
    [1, 2],
    [3, 4],
    [5, 6]
]);
$bias = ZTensor::arr([10, 20]);
$broadcasted = $matrix->broadcast($bias);
print_r($broadcasted->toArray());
// Output: [[11, 22], [13, 24], [15, 26]]
```

### Tile/Repeat - `tile()`

```php
$tensor = ZTensor::arr([
    [1, 2],
    [3, 4]
]);
$tiled = ZTensor::tile($tensor, 3);
print_r($tiled->toArray());
// Output: [[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]]
```

---

## Autograd & Gradient Operations (Experimental)

> ⚠️ **Note:** Autograd infrastructure is experimental. Backward pass is not yet fully implemented.

### Enabling Gradient Tracking - `requiresGrad()`

```php
// Enable gradient tracking
$tensor = ZTensor::arr([1.0, 2.0, 3.0]);
$tensor->requiresGrad(true);

// Or method chaining
$t = ZTensor::arr([[1, 2], [3, 4]])->requiresGrad(true);
```

### Checking Gradient Status - `isRequiresGrad()`

```php
$tensor = ZTensor::arr([1, 2, 3]);
$tensor->requiresGrad(true);

if ($tensor->isRequiresGrad()) {
    echo "This tensor tracks gradients\n";
}
```

### Initializing Gradients - `ensureGrad()`

```php
$tensor = ZTensor::arr([1.0, 2.0, 3.0]);
$tensor->requiresGrad(true);

// Ensure gradient is allocated
$tensor->ensureGrad();

// Get gradient tensor
$grad = $tensor->getGrad();
if ($grad !== null) {
    echo "Gradient allocated: " . $grad->size() . " elements\n";
}
```

### Getting Current Gradient - `getGrad()`

```php
$tensor = ZTensor::arr([[1.0, 2.0], [3.0, 4.0]]);
$tensor->requiresGrad(true);
$tensor->ensureGrad();

$gradient = $tensor->getGrad();
if ($gradient !== null) {
    echo "Gradient shape: ";
    print_r($gradient->shape());
    echo "Gradient values: ";
    print_r($gradient->toArray());
}
```

### Zeroing Gradients - `zeroGrad()`

```php
$tensor = ZTensor::arr([1.0, 2.0, 3.0]);
$tensor->requiresGrad(true);
$tensor->ensureGrad();

// Accumulate gradients...
// Then zero them for next iteration
$tensor->zeroGrad();
```

### Automatic Differentiation - Autograd Methods

**⚠️ Experimental: These methods prepare infrastructure for backpropagation. Full backward pass not yet implemented.**

#### Adding with Autograd - `addAutograd()`
```php
$a = ZTensor::arr([1.0, 2.0])->requiresGrad(true);
$b = ZTensor::arr([3.0, 4.0])->requiresGrad(true);

$c = ZTensor::addAutograd($a, $b);
echo "Result: ";
print_r($c->toArray());
// Output: [4, 6]
```

#### Subtracting with Autograd - `subAutograd()`
```php
$a = ZTensor::arr([5.0, 7.0])->requiresGrad(true);
$b = ZTensor::arr([2.0, 3.0])->requiresGrad(true);

$c = ZTensor::subAutograd($a, $b);
echo "Result: ";
print_r($c->toArray());
// Output: [3, 4]
```

#### Multiplying with Autograd - `mulAutograd()`
```php
$a = ZTensor::arr([2.0, 3.0])->requiresGrad(true);
$b = ZTensor::arr([4.0, 5.0])->requiresGrad(true);

$c = ZTensor::mulAutograd($a, $b);
echo "Result: ";
print_r($c->toArray());
// Output: [8, 15]
```

#### Sum with Autograd - `sumAutograd()`
```php
$tensor = ZTensor::arr([[1.0, 2.0], [3.0, 4.0]])->requiresGrad(true);

$sum = ZTensor::sumAutograd($tensor);
echo "Sum result: " . $sum->toArray()[0] . "\n";
// Output: 10
```

### Backward Pass - `backward()`

**⚠️ Experimental: Currently infrastructure-only. Full backpropagation not yet fully implemented.**

```php
// Example structure (backward pass not yet complete)
$x = ZTensor::arr([1.0, 2.0, 3.0])->requiresGrad(true);
$y = ZTensor::addAutograd($x, $x);
$loss = ZTensor::sumAutograd($y);

// Would trigger backpropagation (when fully implemented)
// $loss->backward();

// For now, you can manually zero and accumulate gradients
// $loss->ensureGrad();
// $loss->zeroGrad();
```

### String Representation - `__toString()`

```php
$tensor = ZTensor::arr([1, 2, 3, 4, 5, 6]);
$tensor->reshape([2, 3]);

echo $tensor;
// Output: ZTensor([2, 3], dtype: float32, 6 elements)
```

---

## Special Operations

### Broadcasting - `broadcast()`

```php
$matrix = ZTensor::arr([
    [1, 2],
    [3, 4],
    [5, 6]
]);
$bias = ZTensor::arr([10, 20]);
$broadcasted = $matrix->broadcast($bias);
print_r($broadcasted->toArray());
// Output: [[11, 22], [13, 24], [15, 26]]
```

---

## 🚀 GPU Aceleração Detalhada

### Transferência de Dados - `toGpu()` e `toCpu()`

```php
$tensor = ZTensor::random([5000, 5000]);

// Move para GPU
$tensor->toGpu();

// Operações na GPU são aceleradas
$result = $tensor->relu();
$result->add($other_tensor);

// Volta para CPU
$tensor->toCpu();
$php_array = $tensor->toArray();
```

### Verificar Localização do Tensor - `isOnGpu()`

```php
$tensor = ZTensor::arr([[1, 2], [3, 4]]);

if ($tensor->isOnGpu()) {
    echo "Tensor está na GPU\n";
} else {
    echo "Tensor está na CPU\n";
}
```

### Liberar Memória GPU - `freeDevice()`

```php
// Após usar muitos tensores na GPU
$tensor1->freeDevice();
$tensor2->freeDevice();
$tensor3->freeDevice();

// Ou em um loop
foreach ($large_tensors as $tensor) {
    $tensor->toGpu();
    $tensor->relu();
    $tensor->toCpu();
    $tensor->freeDevice();  // Libera GPU imediatamente
}
```

### Exemplo Prático: ML com GPU

```php
use ZMatrix\ZTensor;

// Dados de treinamento grandes
$X_train = ZTensor::random([10000, 100]);
$y_train = ZTensor::random([10000, 10]);

// Move para GPU
$X_train->toGpu();
$y_train->toGpu();

// Forward pass
$hidden = $X_train->matmul($W1);
$hidden->add($b1);
$hidden->relu();

// Aplicar dropout, etc
$output = $hidden->matmul($W2);
$output->softmax();

// Volta para CPU para processing
$output->toCpu();
$predictions = $output->toArray();

// Libera memória GPU
$X_train->freeDevice();
$y_train->freeDevice();
```

---

## Métodos Adicionais

### Acessar Elemento por Índice - `key()`

```php
$tensor = ZTensor::arr([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

// Acessa elemento em posição [1, 2]
$element = $tensor->key([1, 2]);
echo "Element: $element\n";  // Output: 6

// Acessa elemento em 3D
$tensor_3d = ZTensor::arr([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
$elem = $tensor_3d->key([0, 1, 0]);
echo "Element: $elem\n";  // Output: 3
```

### Funções de Extremo - `minimum()` e `maximum()`

```php
// Element-wise minimum with scalar
$data = ZTensor::arr([1, 5, 3, 8, 2]);
$min_result = ZTensor::minimum($data, 4.0);
print_r($min_result->toArray());
// Output: [1, 4, 3, 4, 2]

// Element-wise maximum with scalar
$max_result = ZTensor::maximum($data, 4.0);
print_r($max_result->toArray());
// Output: [4, 5, 4, 8, 4]
```

---

---

## 📚 Complete API Reference - Resumo de Todos os Métodos

### Tabela Rápida de Todos os 62 Métodos

| Categoria | Método | Tipo | Descrição |
|-----------|--------|------|-----------|
| **Criação** | `__construct()` | Constructor | Cria tensor de array ou vazio |
| | `arr()` | Static | Factory method para criar de array |
| | `safe()` | Static | Criação segura com validação |
| | `copy()` | Instance | Deep copy do tensor |
| **Criação - Especiais** | `zeros()` | Static | Tensor com zeros |
| | `ones()` | Static | Tensor com uns |
| | `full()` | Static | Tensor preenchido com valor |
| | `identity()` | Static | Matriz identidade |
| | `eye()` | Static | Matriz diagonal |
| | `random()` | Static | Valores aleatórios uniformes |
| | `randn()` | Static | Valores aleatórios normais |
| **Sequências** | `arange()` | Static | Sequência com passo |
| | `linspace()` | Static | Espaço linear |
| | `logspace()` | Static | Espaço logarítmico |
| **Aritmética** | `add()` | Instance | Adição elemento a elemento |
| | `sub()` | Instance | Subtração elemento a elemento |
| | `mul()` | Instance | Multiplicação elemento a elemento |
| | `divide()` | Instance | Divisão elemento a elemento |
| | `pow()` | Instance | Potência elemento a elemento |
| | `scalarMultiply()` | Instance | Multiplicação por escalar |
| | `scalarDivide()` | Instance | Divisão por escalar |
| **Álgebra Linear** | `matmul()` | Instance | Multiplicação de matrizes |
| | `dot()` | Instance | Produto escalar/dot product |
| | `transpose()` | Instance | Transposição (2D) |
| **Funções Matemáticas** | `abs()` | Instance | Valor absoluto |
| | `sqrt()` | Instance | Raiz quadrada |
| | `exp()` | Instance | Exponencial |
| | `log()` | Instance | Logaritmo natural |
| **Ativações** | `sigmoid()` | Instance | Função sigmoid |
| | `sigmoidDerivative()` | Instance | Derivada sigmoid |
| | `relu()` | Instance | ReLU |
| | `reluDerivative()` | Instance | Derivada ReLU |
| | `leakyRelu()` | Instance | Leaky ReLU com alpha |
| | `leakyReluDerivative()` | Instance | Derivada Leaky ReLU |
| | `tanh()` | Instance | Tangente hiperbólica |
| | `tanhDerivative()` | Instance | Derivada tanh |
| | `softmax()` | Instance | Softmax |
| | `softmaxDerivative()` | Instance | Derivada softmax |
| **Estatísticas** | `sum()` | Instance | Soma global ou por eixo (v0.5.0+) |
| | `sumtotal()` | Instance | Soma total de elementos |
| | `mean()` | Instance | Média |
| | `min()` | Instance | Mínimo |
| | `max()` | Instance | Máximo |
| | `std()` | Instance | Desvio padrão |
| **Comparação** | `greater()` | Instance | Comparação > |
| | `clip()` | Static | Limita valores min-max |
| | `minimum()` | Static | Min elemento com escalar |
| | `maximum()` | Static | Max elemento com escalar |
| **Shape & Info** | `shape()` | Instance | Retorna shape |
| | `ndim()` | Instance | Número de dimensões |
| | `size()` | Instance | Total de elementos |
| | `isEmpty()` | Instance | Verifica se vazio |
| | `reshape()` | Instance | Muda shape |
| | `toArray()` | Instance | Converte para array PHP |
| **Acesso** | `key()` | Instance | Acessa elemento por índices |
| **Manipulação** | `broadcast()` | Instance | Broadcast com bias |
| | `tile()` | Static | Repete tensor N vezes |
| **Autograd** | `requiresGrad()` | Instance | Ativa rastreamento gradiente |
| | `isRequiresGrad()` | Instance | Verifica rastreamento |
| | `ensureGrad()` | Instance | Aloca tensor de gradiente |
| | `zeroGrad()` | Instance | Zera gradiente acumulado |
| | `getGrad()` | Instance | Obtém tensor de gradiente |
| | `addAutograd()` | Static | Adição com autograd (exp) |
| | `subAutograd()` | Static | Subtração com autograd (exp) |
| | `mulAutograd()` | Static | Multiplicação com autograd (exp) |
| | `sumAutograd()` | Static | Soma com autograd (exp) |
| **GPU** | `toGpu()` | Instance | Move para GPU |
| | `toCpu()` | Instance | Move para CPU |
| | `isOnGpu()` | Instance | Verifica se está em GPU |
| | `freeDevice()` | Instance | Libera memória GPU |
| **String** | `__toString()` | Instance | Representação em string |

### Categorias de Uso

**Para iniciantes:**
- `arr()` - criar tensores
- `shape()`, `toArray()` - inspecionar
- `add()`, `sub()`, `mul()` - aritmética básica
- `reshape()`, `transpose()` - manipulação de shape

**Para machine learning:**
- Todas as funções de ativação (`relu`, `sigmoid`, `softmax`)
- `matmul()` para redes neurais
- `requiresGrad()` para preparar autograd
- `toGpu()`, `toCpu()` para aceleração

**Para computação numérica:**
- `random()`, `randn()` para inicialização
- Funções matemáticas: `exp()`, `log()`, `sqrt()`, `pow()`
- Estatísticas: `mean()`, `std()`, `sum()`
- Comparação: `greater()`, `clip()`, `minimum()`, `maximum()`

**Para processamento em lote:**
- `broadcast()` para aplicar bias
- `tile()` para repetir operações
- `dot()` para agregação
- GPU methods para grandes volumes

---

## 🔧 Troubleshooting

### Problema: Erro de compilação "cuda.h not found"

**Solução:**
```bash
# Especifique o caminho do CUDA durante configure
./configure --enable-zmatrix --with-cuda-path=/usr/local/cuda

# Ou verifique se CUDA está instalado
nvcc --version
```

### Problema: "libcuda.so not found" em runtime

**Solução:**
```bash
# O sistema tenta encontrar libcuda.so automaticamente
# Se não encontrar, adicione ao LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH  # WSL
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH  # CUDA Toolkit
```

### Problema: GPU methods não funcionam (Exception: CUDA support not available)

**Causa:** CUDA não foi encontrado durante compilação ou em runtime

**Solução:**
- GPU é opcional! A biblioteca continua funcionando com CPU
- Use operações CPU normalmente:
```php
$tensor = ZTensor::arr([[1, 2], [3, 4]]);
$tensor->add([1, 1]);  // Funciona em CPU
```

- Se precisa de GPU, instale drivers NVIDIA:
```bash
sudo apt-get install nvidia-driver-XXX  # Verifique versão recomendada
```

### Problema: WSL2 não detecta GPU

**Solução:**
```bash
# Verifique se drivers NVIDIA estão instalados (no Windows)
nvidia-smi  # No PowerShell

# Em WSL, configure para usar GPU:
./configure --enable-zmatrix
# Sistema detecta automaticamente WSL e configura caminhos corretos
```

### Problema: Compilação lenta ou falha com "make -j"

**Solução:**
```bash
# Use menos threads de compilação
make -j2  # Ao invés de -j$(nproc)

# Ou limite memória
make -j$(( $(nproc) / 2 ))
```

### Problema: "PHP Fatal error: Class 'ZMatrix\ZTensor' not found"

**Solução:**
```bash
# Verifique se extensão está carregada
php -m | grep zmatrix

# Se não aparecer, adicione ao php.ini
echo "extension=zmatrix.so" | sudo tee -a /etc/php/8.x/cli/php.ini

# Recarregue
php -r "echo 'OK';"
```

### Problema: Performance ruim em GPU

**Causas:**
- Tensor muito pequeno (< 200k elementos)
- Overhead de H2D/D2H transfer maior que ganho de cálculo
- GPU ocupada por outro processo

**Solução:**
```php
// Use GPU apenas para operações grandes
if ($tensor->size() > 200000) {
    $tensor->toGpu();
    $tensor->relu();
    $tensor->toCpu();
} else {
    // CPU é mais rápido para tensores pequenos
    $tensor->relu();
}
```

### Problema: Out of GPU Memory

**Solução:**
```php
// Libere memória explicitamente
$tensor->freeDevice();

// Ou avoid keeping many large tensors on GPU
foreach ($tensors as $t) {
    $t->toGpu();
    // process...
    $t->toCpu();
    $t->freeDevice();  // Free GPU memory
}
```

### Problema: Resultados diferentes entre CPU e GPU

**Causa Comum:** Ordem de operações, tipo de dado, ou precisão numérica

**Solução:**
```php
// Garantir mesmos dados
$a_cpu = $a->copy();  // Cópia para CPU
$a_gpu = $a->copy();  // Cópia para GPU
$a_gpu->toGpu();

// Operações idênticas
$a_cpu->add($b);
$a_gpu->add($b);

// Comparar (com tolerância numérica)
$diff = abs($a_cpu->sum() - $a_gpu->sum());
assert($diff < 1e-5);  // Deve ser próximo
```

---

## Support

For issues related to the ZTensor extension itself, please refer to the official repository or documentation provided by the extension maintainers.
## 🙌 Contribution

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

MIT License. See `LICENSE`.
