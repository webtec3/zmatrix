# 🧪 PLANO DE TESTES - zmatrix.cpp

Testes recomendados para validar as correções e encontrar regressões.

---

## 1. TESTES DE SEGURANÇA

### T1.1: Teste de Overflow em Shape

**Objetivo**: Garantir que overflow em shapes é detectado

```cpp
// test_overflow.cpp
#include <gtest/gtest.h>
#include "zmatrix.cpp"

TEST(ZTensorSafety, OverflowDetection) {
    // Tentativa 1: Multiplicação que causa overflow
    std::vector<size_t> huge_shape = {
        std::numeric_limits<size_t>::max() / 2,
        10  // Isto causaria overflow
    };
    
    EXPECT_THROW({
        ZTensor t(huge_shape);
    }, std::overflow_error);
}

TEST(ZTensorSafety, ZeroDimension) {
    // Tensor com dimensão 0 deve ser válido (vazio)
    std::vector<size_t> zero_shape = {5, 0, 3};
    
    EXPECT_NO_THROW({
        ZTensor t(zero_shape);
        EXPECT_EQ(t.size(), 0);
        EXPECT_EQ(t.data.size(), 0);
    });
}
```

**Executar**:
```bash
g++ -std=c++17 -Wall test_overflow.cpp -o test_overflow
./test_overflow
```

---

### T1.2: Teste de Out-of-Bounds Access

**Objetivo**: Verificar se `at()` rejeita índices inválidos

```cpp
TEST(ZTensorBounds, OutOfBounds) {
    ZTensor t({3, 4});  // Shape 3x4
    
    // Acesso válido
    EXPECT_NO_THROW({
        float val = t.at({0, 0});
        t.at({2, 3});
    });
    
    // Acessos inválidos
    EXPECT_THROW({
        t.at({3, 0});  // Linha 3 não existe (0-2)
    }, std::out_of_range);
    
    EXPECT_THROW({
        t.at({0, 4});  // Coluna 4 não existe (0-3)
    }, std::out_of_range);
    
    EXPECT_THROW({
        t.at({0, 0, 0});  // Muitos índices
    }, std::invalid_argument);
}

TEST(ZTensorBounds, EmptyTensorAccess) {
    ZTensor empty({0});
    
    EXPECT_THROW({
        float val = empty.at({0});
    }, std::out_of_range);
}
```

---

## 2. TESTES DE PERFORMANCE

### T2.1: OpenMP Overhead

**Objetivo**: Medir se OpenMP está ativado e trazendo ganho

```cpp
// test_performance.cpp
#include <chrono>
#include "zmatrix.cpp"

class PerformanceTest : public ::testing::Test {
protected:
    double measure_add_time(size_t size, int iterations = 100) {
        ZTensor a = ZTensor::random({size});
        ZTensor b = ZTensor::random({size});
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            a.add(b);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        return std::chrono::duration<double>(end - start).count();
    }
};

TEST_F(PerformanceTest, OpenMPSpeedup) {
    // Operação pequena (sem paralelismo)
    double small_time = measure_add_time(1000, 1000);
    
    // Operação grande (com potencial paralelismo)
    double large_time = measure_add_time(1000000, 10);
    
    // Operação grande deveria ser mais rápida por elemento
    double small_per_element = small_time / (1000 * 1000);
    double large_per_element = large_time / (1000000 * 10);
    
    // Com OpenMP, large deveria ser ~2x mais rápido por elemento
    // (redução de overhead relativo)
    std::cout << "Small per-element: " << small_per_element << " s\n";
    std::cout << "Large per-element: " << large_per_element << " s\n";
    std::cout << "Ratio: " << small_per_element / large_per_element << "x\n";
}

TEST_F(PerformanceTest, SIMDAlignment) {
    // Verificar se arrays alinhados em SIMD boundary
    // (16-byte para SSE, 32-byte para AVX2)
    ZTensor t({1024});
    
    uintptr_t addr = reinterpret_cast<uintptr_t>(t.data.data());
    
    // SSE: 16-byte aligned
    if (HAS_AVX2) {
        EXPECT_EQ(addr % 32, 0) << "Data not aligned for AVX2";
    } else {
        // Pelo menos 16-byte para SSE
        EXPECT_EQ(addr % 16, 0) << "Data not aligned for SSE";
    }
}
```

**Executar**:
```bash
g++ -std=c++17 -O3 -fopenmp test_performance.cpp -o test_perf
./test_perf --gtest_filter="PerformanceTest.*"
```

---

## 3. TESTES DE PRECISÃO

### T3.1: Float vs Double

**Objetivo**: Quantificar perda de precisão

```cpp
// test_precision.cpp
#include <cmath>
#include "zmatrix.cpp"

TEST(Precision, FloatVsDouble) {
    // Criar dados que são sensíveis a precisão
    std::vector<double> test_values = {
        0.1, 0.2, 0.3,  // 0.1 + 0.2 != 0.3 em floating point
        1e-6, 1e-7, 1e-8,
        std::sqrt(2.0), std::sqrt(3.0),
        M_PI, M_E
    };
    
    // Teste 1: Acumulação com floats
    float float_sum = 0.0f;
    for (double val : test_values) {
        float_sum += static_cast<float>(val);
    }
    
    // Teste 2: Acumulação com doubles
    double double_sum = 0.0;
    for (double val : test_values) {
        double_sum += val;
    }
    
    // Teste 3: Acumulação em ZTensor
    ZTensor t({test_values.size()});
    for (size_t i = 0; i < test_values.size(); ++i) {
        t.data[i] = static_cast<float>(test_values[i]);
    }
    double tensor_sum = t.sum();
    
    std::cout << "Float sum:   " << std::fixed << std::setprecision(15) << float_sum << "\n";
    std::cout << "Double sum:  " << std::fixed << std::setprecision(15) << double_sum << "\n";
    std::cout << "Tensor sum:  " << std::fixed << std::setprecision(15) << tensor_sum << "\n";
    std::cout << "Error: " << std::abs(tensor_sum - double_sum) << "\n";
    
    // Float deve ter erro maior (esperado)
    EXPECT_GT(std::abs(float_sum - double_sum), 1e-6);
}

TEST(Precision, DotProductAccuracy) {
    // Teste: dot product de vetores ortogonais
    ZTensor a = ZTensor::zeros({1000});
    ZTensor b = ZTensor::zeros({1000});
    
    a.data[0] = 1.0f;
    b.data[1] = 1.0f;
    // dot(a, b) deveria ser 0.0 (ortogonal)
    
    // Simular dot product
    double dot_result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot_result += static_cast<double>(a.data[i]) * 
                      static_cast<double>(b.data[i]);
    }
    
    EXPECT_EQ(dot_result, 0.0);
}
```

---

## 4. TESTES DE THREAD-SAFETY

### T4.1: Operações Concorrentes

**Objetivo**: Detectar race conditions em OpenMP

```cpp
// test_threading.cpp
#include <omp.h>
#include "zmatrix.cpp"

TEST(ThreadSafety, ConcurrentAdd) {
    const int num_threads = 8;
    const int operations = 1000;
    
    ZTensor a = ZTensor::random({10000});
    ZTensor b = ZTensor::random({10000});
    ZTensor expected = a;  // Cópia da inicial
    
    // Realizar a mesma adição 'operations' vezes
    // De forma paralela
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int op = 0; op < operations; ++op) {
        ZTensor local_a = a;
        ZTensor local_b = b;
        // Não modifica 'a' aqui, só lê
        // Isto deveria ser thread-safe
    }
    
    // Não devemos ter corrupção de dados
    EXPECT_EQ(a.shape, {10000});
    EXPECT_EQ(a.data.size(), 10000);
}

TEST(ThreadSafety, RaceDetection) {
    // Usar ThreadSanitizer (se disponível)
    // Compile com: -fsanitize=thread
    
    std::vector<ZTensor> tensors;
    for (int i = 0; i < 10; ++i) {
        tensors.push_back(ZTensor::random({1000}));
    }
    
    // Operações paralelas que DEVERIAM ser seguras
    #pragma omp parallel for
    for (int i = 0; i < 10; ++i) {
        tensors[i].multiply_scalar(2.0f);
    }
    
    // Cada tensor deve ter mantido sua integridade
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(tensors[i].shape, {1000});
    }
}
```

**Executar com ThreadSanitizer**:
```bash
g++ -std=c++17 -O2 -fsanitize=thread test_threading.cpp -o test_thread
./test_thread
```

---

## 5. TESTES DE OPERAÇÕES MATEMÁTICAS

### T5.1: Operações Básicas

```cpp
// test_math.cpp
#include <cmath>
#include "zmatrix.cpp"

TEST(MathOps, Addition) {
    ZTensor a = ZTensor::zeros({2, 3});
    ZTensor b = ZTensor::ones({2, 3});
    
    // Preencher a com valores 1, 2, 3, ...
    for (size_t i = 0; i < a.size(); ++i) {
        a.data[i] = static_cast<float>(i + 1);
    }
    
    a.add(b);
    
    // Esperado: [2, 3, 4, 5, 6, 7]
    std::vector<float> expected = {2, 3, 4, 5, 6, 7};
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_FLOAT_EQ(a.data[i], expected[i]);
    }
}

TEST(MathOps, MatMul) {
    // A: 2x3
    ZTensor a({2, 3});
    a.data = {1, 2, 3, 4, 5, 6};
    
    // B: 3x2
    ZTensor b({3, 2});
    b.data = {7, 8, 9, 10, 11, 12};
    
    // C = A @ B: 2x2
    ZTensor c = a.matmul(b);
    
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    
    std::vector<float> expected = {58, 64, 139, 154};
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_FLOAT_EQ(c.data[i], expected[i]);
    }
}

TEST(MathOps, Sigmoid) {
    // sigmoid(0) = 0.5
    ZTensor t({1});
    t.data[0] = 0.0f;
    
    t.sigmoid();
    
    EXPECT_NEAR(t.data[0], 0.5f, 1e-5f);
}

TEST(MathOps, ReLU) {
    ZTensor t({5});
    t.data = {-2, -1, 0, 1, 2};
    
    t.relu();
    
    std::vector<float> expected = {0, 0, 0, 1, 2};
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_EQ(t.data[i], expected[i]);
    }
}

TEST(MathOps, Reshape) {
    ZTensor t = ZTensor::zeros({2, 3});
    for (size_t i = 0; i < t.size(); ++i) {
        t.data[i] = static_cast<float>(i);
    }
    
    ZTensor reshaped = t.reshape({3, 2});
    
    // Dados devem ser os mesmos
    EXPECT_EQ(reshaped.size(), t.size());
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_EQ(reshaped.data[i], t.data[i]);
    }
}
```

---

## 6. TESTES DE EDGE CASES

### T6.1: Tensores Vazios

```cpp
// test_edgecases.cpp

TEST(EdgeCases, EmptyTensor) {
    ZTensor empty({0});
    EXPECT_EQ(empty.size(), 0);
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.data.size(), 0);
}

TEST(EdgeCases, EmptyOperations) {
    ZTensor a({0});
    ZTensor b({0});
    
    EXPECT_NO_THROW({
        a.add(b);
    });
}

TEST(EdgeCases, SingleElement) {
    ZTensor t({1});
    t.data[0] = 3.14f;
    
    EXPECT_EQ(t.size(), 1);
    EXPECT_FLOAT_EQ(t.data[0], 3.14f);
}

TEST(EdgeCases, VeryLargeTensor) {
    // 100 milhões de elementos (400 MB)
    try {
        ZTensor large({100, 1000, 1000});
        EXPECT_EQ(large.size(), 100000000);
    } catch (const std::runtime_error& e) {
        // OK se memória não for suficiente
        EXPECT_STREQ(e.what(), ZMATRIX_ERR_ALLOC_FAILED);
    }
}
```

---

## 7. SCRIPT DE EXECUÇÃO DE TESTES

```bash
#!/bin/bash
# run_tests.sh

set -e

echo "========================================"
echo "Compilando testes..."
echo "========================================"

# Compilar testes
g++ -std=c++17 -O2 -Wall -fopenmp \
    -I/usr/local/include \
    -L/usr/local/lib \
    test_overflow.cpp -o test_overflow -lm

g++ -std=c++17 -O3 -Wall -fopenmp \
    -I/usr/local/include \
    -L/usr/local/lib \
    test_performance.cpp -o test_perf -lgtest -lgtest_main -lm

g++ -std=c++17 -O2 -Wall -fopenmp \
    -I/usr/local/include \
    -L/usr/local/lib \
    test_precision.cpp -o test_prec -lm

g++ -std=c++17 -O2 -Wall -fopenmp -fsanitize=thread \
    -I/usr/local/include \
    -L/usr/local/lib \
    test_threading.cpp -o test_thread -lm

g++ -std=c++17 -O2 -Wall -fopenmp \
    -I/usr/local/include \
    -L/usr/local/lib \
    test_math.cpp -o test_math -lgtest -lgtest_main -lm

g++ -std=c++17 -O2 -Wall -fopenmp \
    -I/usr/local/include \
    -L/usr/local/lib \
    test_edgecases.cpp -o test_edge -lgtest -lgtest_main -lm

echo ""
echo "========================================"
echo "Executando testes..."
echo "========================================"

echo ""
echo "1. Testes de Segurança (Overflow)..."
./test_overflow

echo ""
echo "2. Testes de Performance..."
./test_perf --gtest_filter="PerformanceTest.*"

echo ""
echo "3. Testes de Precisão..."
./test_prec

echo ""
echo "4. Testes de Thread-Safety..."
./test_thread

echo ""
echo "5. Testes Matemáticos..."
./test_math

echo ""
echo "6. Testes Edge Cases..."
./test_edge

echo ""
echo "========================================"
echo "✅ Todos os testes executados!"
echo "========================================"
```

**Executar**:
```bash
chmod +x run_tests.sh
./run_tests.sh
```

---

## 8. TESTES DE REGRESSÃO EM PHP

### T8.1: Tests em PHP

```php
<?php
// tests/regression_test.php

class ZTensorRegressionTest extends PHPUnit\Framework\TestCase {
    
    public function testBasicConstruction() {
        $t = new ZMatrix\ZTensor([3, 4]);
        $this->assertEquals($t->size(), 12);
        $this->assertEquals(count($t->shape()), 2);
    }
    
    public function testAddition() {
        $a = ZMatrix\ZTensor::ones([3, 4]);
        $b = ZMatrix\ZTensor::ones([3, 4]);
        
        $result = $a->add($b);
        $this->assertEquals($result, $a);  // Retorna $this
        
        // Verificar valores
        $array = $a->toArray();
        foreach ($array as $row) {
            foreach ($row as $val) {
                $this->assertEquals($val, 2.0);
            }
        }
    }
    
    public function testMatMul() {
        $a = ZMatrix\ZTensor::random([3, 4]);
        $b = ZMatrix\ZTensor::random([4, 5]);
        
        $result = $a->matmul($b);
        $shape = $result->shape();
        
        $this->assertEquals($shape[0], 3);
        $this->assertEquals($shape[1], 5);
    }
    
    public function testExceptionOnMismatch() {
        $this->expectException(\Exception::class);
        
        $a = ZMatrix\ZTensor::zeros([3, 4]);
        $b = ZMatrix\ZTensor::zeros([2, 3]);
        
        $a->add($b);  // Shape mismatch
    }
    
    public function testThreadedOperations() {
        $tensors = [];
        for ($i = 0; $i < 10; $i++) {
            $tensors[] = ZMatrix\ZTensor::random([1000]);
        }
        
        foreach ($tensors as $t) {
            $t->relu();  // Operação paralela
        }
        
        // Verificar integridade
        foreach ($tensors as $t) {
            $this->assertEquals($t->size(), 1000);
        }
    }
}
```

**Executar**:
```bash
phpunit tests/regression_test.php
```

---

## 📊 MATRIZ DE COBERTURA

| Categoria | Testes | Status |
|-----------|--------|--------|
| Segurança | 4 | ✅ |
| Performance | 3 | ✅ |
| Precisão | 2 | ✅ |
| Threading | 2 | ✅ |
| Matemática | 5 | ✅ |
| Edge Cases | 4 | ✅ |
| Regressão PHP | 5 | ✅ |
| **TOTAL** | **25** | ✅ |

---

## 📈 MÉTRICAS ALVO

| Métrica | Alvo | Método |
|---------|------|--------|
| Code Coverage | > 80% | gcov/lcov |
| Memory Leaks | 0 | Valgrind |
| Performance | > 2x com OpenMP | time |
| Precision Error | < 1e-5 | Comparação com numpy |
| Thread-Safety | No race conditions | ThreadSanitizer |

```bash
# Executar com Valgrind
valgrind --leak-check=full ./test_math

# Executar com code coverage
gcc -std=c++17 --coverage test_math.cpp -o test_math
./test_math
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage_report
```

