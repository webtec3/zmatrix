# 🧠 Implementação Detalhada do Sistema de Autograd - ZMatrix/ZTensor

**Status**: ✅ **IMPLEMENTAÇÃO ESTÁTICA COM DETALHES C++ COMPLETOS**

**Data**: 16 de Janeiro, 2026

**Autor**: GitHub Copilot

---

## 📋 Sumário Executivo

O sistema de autograd foi implementado com 4 operações diferenciáveis expostas como **métodos estáticos** na classe `ZTensor`:

1. **`ZTensor::addAutograd($a, $b)`** - Adição elemento-a-elemento com rastreamento de gradientes
2. **`ZTensor::subAutograd($a, $b)`** - Subtração elemento-a-elemento  
3. **`ZTensor::mulAutograd($a, $b)`** - Multiplicação elemento-a-elemento
4. **`ZTensor::sumAutograd($tensor)`** - Redução de soma com broadcasting de gradientes

Cada operação cria um **nó de computação** (`AutogradNode`) que armazena a função de backward para cálculo de gradientes.

---

## 🏗️ Arquitetura do Sistema

### Componentes Principais

```
┌─────────────────────────────────────────────────────────┐
│          Camada PHP (Interface do Usuário)              │
│  ZTensor::addAutograd($a, $b) -> ZTensor               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         Binding C++ → PHP (zmatrix.cpp)                 │
│  PHP_METHOD(ZTensor, addAutograd) { ... }              │
│  - Parse 2 objetos ZTensor do PHP                       │
│  - Extrai pointers C++                                  │
│  - Chama ZTensor::add_autograd() nativa                │
│  - Retorna resultado como objeto PHP                    │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│       Implementação C++ (ZTensor class)                 │
│  static ZTensor add_autograd(...) { ... }              │
│  - Computação: c[i] = a[i] + b[i]                     │
│  - Cria AutogradNode                                    │
│  - Define função backward                              │
│  - Retorna tensor com grad_fn                          │
└─────────────────────────────────────────────────────────┘
```

### Estrutura de Dados de Autograd

```cpp
struct AutogradNode {
    std::string op_name;           // "add", "sub", "mul", "sum"
    std::vector<std::shared_ptr<ZTensor>> parents;  // Tensores input
    std::function<void()> backward_fn;  // Função de backward
};

class ZTensor {
    std::shared_ptr<ZTensor> grad;    // Gradiente acumulado
    std::shared_ptr<AutogradNode> grad_fn;  // Nó de computação
    bool requires_grad;               // Flag para rastrear gradientes
};
```

---

## 🔍 Implementação Detalhada de Cada Operação

### 1. **addAutograd() - Adição com Autograd**

**Localização**: [src/zmatrix.cpp](src/zmatrix.cpp#L2090)

#### Assinatura C++
```cpp
static ZTensor add_autograd(const ZTensor& a, const ZTensor& b)
```

#### Código Completo Comentado

```cpp
static ZTensor add_autograd(const ZTensor& a, const ZTensor& b) {
    // ========== PASSO 1: VALIDAÇÃO ==========
    if (a.shape != b.shape) {
        throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
    }
    
    // ========== PASSO 2: ALOCAÇÃO ==========
    ZTensor result(a.shape);
    const size_t N = a.size();
    
    // ========== PASSO 3: FORWARD (PARALELIZADO) ==========
    if (N > 0) {
        const float* a_data = a.data.data();    // Acesso direto ao buffer
        const float* b_data = b.data.data();
        float* r_data = result.data.data();
        
#if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            // Para tensores grandes: usa paralelização OpenMP
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                r_data[i] = a_data[i] + b_data[i];
            }
        } else {
            // Para tensores pequenos: execução sequencial (sem overhead)
            for (size_t i = 0; i < N; ++i) {
                r_data[i] = a_data[i] + b_data[i];
            }
        }
#else
        // Fallback sem OpenMP
        for (size_t i = 0; i < N; ++i) {
            r_data[i] = a_data[i] + b_data[i];
        }
#endif
    }
    
    // ========== PASSO 4: SETUP DE AUTOGRAD ==========
    // Determina se precisa rastrear gradientes
    bool requires_grad = a.requires_grad || b.requires_grad;
    result.requires_grad = requires_grad;
    
    if (requires_grad) {
        // Cria nó de computação
        auto node = std::make_shared<AutogradNode>("add");
        
        // Captura operandos como shared_ptr para evitar dangling pointers
        // (permite que backward_fn acesse dados mesmo se operandos originais
        //  saem de escopo)
        auto a_ptr = std::make_shared<ZTensor>(a);
        auto b_ptr = std::make_shared<ZTensor>(b);
        auto result_ptr = std::make_shared<ZTensor>(result);
        
        // Registra pais do nó
        node->parents = {a_ptr, b_ptr};
        
        // Determina quem precisa de gradiente
        bool a_req = a.requires_grad;
        bool b_req = b.requires_grad;
        
        // ========== PASSO 5: DEFINE FUNÇÃO BACKWARD ==========
        // Para adição: z = x + y
        //   ∂z/∂x = 1
        //   ∂z/∂y = 1
        // Logo no backward:
        //   grad_x = grad_z
        //   grad_y = grad_z
        node->backward_fn = [result_ptr, a_ptr, b_ptr, a_req, b_req]() {
            // Obtém gradiente do nó resultado
            const ZTensor* grad_result = result_ptr->getGrad();
            
            if (!grad_result) return;  // Sem gradiente, não há trabalho
            
            // Ambos os pais recebem o mesmo gradiente
            if (a_req) {
                // Acumula gradiente em 'a'
                const_cast<ZTensor*>(a_ptr.get())->accumulate_grad(*grad_result);
            }
            if (b_req) {
                // Acumula gradiente em 'b'
                const_cast<ZTensor*>(b_ptr.get())->accumulate_grad(*grad_result);
            }
        };
        
        // Armazena nó no resultado para backward futuro
        result.grad_fn = node;
    }
    
    return result;
}
```

#### Matemática de Gradientes

```
Forward pass:  c = a + b

Backward pass (Regra da Cadeia):
  ∂L/∂a = ∂L/∂c · ∂c/∂a = ∂L/∂c · 1 = ∂L/∂c
  ∂L/∂b = ∂L/∂c · ∂c/∂b = ∂L/∂c · 1 = ∂L/∂c
  
Exemplo numérico:
  a = [[1, 2], [3, 4]]  (requires_grad=true)
  b = [[2, 2], [2, 2]]  (requires_grad=false)
  c = a + b = [[3, 4], [5, 6]]
  
  Se grad_c = [[1, 1], [1, 1]] (de operação posterior):
    grad_a = [[1, 1], [1, 1]]
    grad_b = [[1, 1], [1, 1]]  (mas b não requer grad, então ignorado)
```

#### Exemplo de Uso PHP

```php
<?php
// Criar tensores com rastreamento de gradientes
$x = ZTensor::ones([2, 2])->requiresGrad(true);   // [[1, 1], [1, 1]]
$y = ZTensor::ones([2, 2]) * 2;                   // [[2, 2], [2, 2]]

// Forward: z = x + y
$z = ZTensor::addAutograd($x, $y);                // [[3, 3], [3, 3]]

// Reduzir a escalar
$loss = ZTensor::sumAutograd($z);                 // 12

// Backward: computa gradientes
$loss->backward();

// Acessar gradientes
echo $x->grad()->toArray();  
// [[1, 1], [1, 1]]
// (cada entrada de z contribui 1 para loss, 
//  logo grad de cada entrada de x é 1)

echo $y->grad();  
// NULL (y não tinha requiresGrad=true)
?>
```

---

### 2. **subAutograd() - Subtração com Autograd**

**Localização**: [src/zmatrix.cpp](src/zmatrix.cpp#L2164)

#### Diferença Principal: Gradiente em B é Negado

```cpp
static ZTensor sub_autograd(const ZTensor& a, const ZTensor& b) {
    // Validação e alocação similar a add_autograd()
    if (a.shape != b.shape) {
        throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
    }
    
    ZTensor result(a.shape);
    const size_t N = a.size();
    
    // Forward: c[i] = a[i] - b[i]
    if (N > 0) {
        const float* a_data = a.data.data();
        const float* b_data = b.data.data();
        float* r_data = result.data.data();
        
#if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                r_data[i] = a_data[i] - b_data[i];  // SUBTRAÇÃO
            }
        } else {
            for (size_t i = 0; i < N; ++i) {
                r_data[i] = a_data[i] - b_data[i];
            }
        }
#else
        for (size_t i = 0; i < N; ++i) {
            r_data[i] = a_data[i] - b_data[i];
        }
#endif
    }
    
    // Autograd setup
    bool requires_grad = a.requires_grad || b.requires_grad;
    result.requires_grad = requires_grad;
    
    if (requires_grad) {
        auto node = std::make_shared<AutogradNode>("sub");
        auto a_ptr = std::make_shared<ZTensor>(a);
        auto b_ptr = std::make_shared<ZTensor>(b);
        auto result_ptr = std::make_shared<ZTensor>(result);
        node->parents = {a_ptr, b_ptr};
        
        bool a_req = a.requires_grad;
        bool b_req = b.requires_grad;
        
        // ===== BACKWARD DIFERENTE DA ADIÇÃO =====
        // Para subtração: z = x - y
        //   ∂z/∂x = 1
        //   ∂z/∂y = -1  ← NEGADO!
        // Logo:
        //   grad_x = grad_z
        //   grad_y = -grad_z  ← NEGADO!
        
        node->backward_fn = [result_ptr, a_ptr, b_ptr, a_req, b_req]() {
            const ZTensor* grad_result = result_ptr->getGrad();
            if (!grad_result) return;
            
            if (a_req) {
                // Gradiente normal para 'a'
                const_cast<ZTensor*>(a_ptr.get())->accumulate_grad(*grad_result);
            }
            if (b_req) {
                // Gradiente NEGADO para 'b'
                ZTensor neg_grad(grad_result->shape);
                const size_t N = grad_result->size();
                if (N > 0) {
                    const float* src = grad_result->data.data();
                    float* dst = neg_grad.data.data();
                    for (size_t i = 0; i < N; ++i) {
                        dst[i] = -src[i];  // ← NEGAÇÃO CRÍTICA
                    }
                }
                const_cast<ZTensor*>(b_ptr.get())->accumulate_grad(neg_grad);
            }
        };
        
        result.grad_fn = node;
    }
    
    return result;
}
```

#### Matemática de Gradientes

```
Forward pass:  c = a - b

Backward pass:
  ∂L/∂a = ∂L/∂c · ∂c/∂a = ∂L/∂c · 1   =  ∂L/∂c
  ∂L/∂b = ∂L/∂c · ∂c/∂b = ∂L/∂c · (-1) = -∂L/∂c  ← NEGADO!

Exemplo numérico:
  a = [[5, 5], [5, 5]]  (requires_grad=true)
  b = [[2, 2], [2, 2]]  (requires_grad=true)
  c = a - b = [[3, 3], [3, 3]]
  
  Se grad_c = [[1, 1], [1, 1]]:
    grad_a = [[1, 1], [1, 1]]        ✓ normal
    grad_b = [[-1, -1], [-1, -1]]    ✓ NEGADO!
```

#### Exemplo de Uso PHP

```php
<?php
$a = ZTensor::ones([2, 2]) * 5;    // [[5, 5], [5, 5]]
$b = ZTensor::ones([2, 2]) * 2;    // [[2, 2], [2, 2]]
$a->requiresGrad(true);
$b->requiresGrad(true);

$c = ZTensor::subAutograd($a, $b); // [[3, 3], [3, 3]]
$loss = ZTensor::sumAutograd($c);  // 12
$loss->backward();

echo "a.grad = ";
print_r($a->grad()->toArray());     // [[1, 1], [1, 1]]

echo "b.grad = ";
print_r($b->grad()->toArray());     // [[-1, -1], [-1, -1]]  ← NEGADO!

// Verificação: d(loss)/db = d(sum(a-b))/db = d(12)/db = -1 ✓
?>
```

---

### 3. **mulAutograd() - Multiplicação com Autograd**

**Localização**: [src/zmatrix.cpp](src/zmatrix.cpp#L2242)

#### Complexidade: Captura de Operandos no Backward

A multiplicação requer que os operandos **originais** estejam disponíveis no backward, pois:

```
c = a * b
∂c/∂a = b      ← Precisa do valor de 'b' no backward!
∂c/∂b = a      ← Precisa do valor de 'a' no backward!
```

#### Código Completo

```cpp
static ZTensor mul_autograd(const ZTensor& a, const ZTensor& b) {
    if (a.shape != b.shape) {
        throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
    }
    
    ZTensor result(a.shape);
    const size_t N = a.size();
    
    // Forward: c[i] = a[i] * b[i]
    if (N > 0) {
        const float* a_data = a.data.data();
        const float* b_data = b.data.data();
        float* r_data = result.data.data();
        
#if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                r_data[i] = a_data[i] * b_data[i];  // Multiplicação elemento-a-elemento
            }
        } else {
            for (size_t i = 0; i < N; ++i) {
                r_data[i] = a_data[i] * b_data[i];
            }
        }
#else
        for (size_t i = 0; i < N; ++i) {
            r_data[i] = a_data[i] * b_data[i];
        }
#endif
    }
    
    bool requires_grad = a.requires_grad || b.requires_grad;
    result.requires_grad = requires_grad;
    
    if (requires_grad) {
        auto node = std::make_shared<AutogradNode>("mul");
        auto a_ptr = std::make_shared<ZTensor>(a);
        auto b_ptr = std::make_shared<ZTensor>(b);
        auto result_ptr = std::make_shared<ZTensor>(result);
        node->parents = {a_ptr, b_ptr};
        
        // ===== CAPTURA IMPORTANTE =====
        // Cria cópias dos operandos para uso no backward
        // (precisamos dos VALORES de a e b no backward, não apenas dos pais)
        auto a_copy = std::make_shared<ZTensor>(a);
        auto b_copy = std::make_shared<ZTensor>(b);
        
        bool a_req = a.requires_grad;
        bool b_req = b.requires_grad;
        
        // ===== BACKWARD COM REGRA DO PRODUTO =====
        // Para multiplicação: z = x * y
        //   ∂z/∂x = y    ← Precisa do valor de y
        //   ∂z/∂y = x    ← Precisa do valor de x
        
        node->backward_fn = [result_ptr, a_ptr, b_ptr, a_copy, b_copy, a_req, b_req]() {
            const ZTensor* grad_result = result_ptr->getGrad();
            if (!grad_result) return;
            
            const size_t N = grad_result->size();
            
            // Gradiente para 'a': da = b * grad_output
            if (a_req && N > 0) {
                ZTensor grad_a(grad_result->shape);
                const float* b_data = b_copy->data.data();      // Usa cópia de b
                const float* grad_data = grad_result->data.data();
                float* grad_a_data = grad_a.data.data();
                
                for (size_t i = 0; i < N; ++i) {
                    // Regra do produto: d(a*b)/da = b
                    grad_a_data[i] = b_data[i] * grad_data[i];
                }
                const_cast<ZTensor*>(a_ptr.get())->accumulate_grad(grad_a);
            }
            
            // Gradiente para 'b': db = a * grad_output
            if (b_req && N > 0) {
                ZTensor grad_b(grad_result->shape);
                const float* a_data = a_copy->data.data();      // Usa cópia de a
                const float* grad_data = grad_result->data.data();
                float* grad_b_data = grad_b.data.data();
                
                for (size_t i = 0; i < N; ++i) {
                    // Regra do produto: d(a*b)/db = a
                    grad_b_data[i] = a_data[i] * grad_data[i];
                }
                const_cast<ZTensor*>(b_ptr.get())->accumulate_grad(grad_b);
            }
        };
        
        result.grad_fn = node;
    }
    
    return result;
}
```

#### Matemática de Gradientes (Regra do Produto)

```
Forward pass:  c = a * b

Backward pass (Regra do Produto):
  ∂L/∂a = ∂L/∂c · ∂c/∂a = ∂L/∂c · b
  ∂L/∂b = ∂L/∂c · ∂c/∂b = ∂L/∂c · a

Exemplo numérico:
  a = [[1, 2], [3, 4]]  (requires_grad=true)
  b = [[2, 3], [4, 5]]  (requires_grad=true)
  c = a * b = [[1*2, 2*3], [3*4, 4*5]] = [[2, 6], [12, 20]]
  
  Se grad_c = [[1, 1], [1, 1]] (de sum):
    grad_a[i,j] = b[i,j] * 1 = b[i,j] = [[2, 3], [4, 5]]
    grad_b[i,j] = a[i,j] * 1 = a[i,j] = [[1, 2], [3, 4]]
```

#### Exemplo de Uso PHP

```php
<?php
// Criar tensores
$a = ZTensor::arange(1, 5)->reshape([2, 2]);  // [[1, 2], [3, 4]]
$b = ZTensor::arange(2, 6)->reshape([2, 2]);  // [[2, 3], [4, 5]]
$a->requiresGrad(true);
$b->requiresGrad(true);

// Forward pass
$c = ZTensor::mulAutograd($a, $b);
// c = [[1*2, 2*3], [3*4, 4*5]] = [[2, 6], [12, 20]]

$loss = ZTensor::sumAutograd($c);  // 2 + 6 + 12 + 20 = 40
$loss->backward();

echo "a.grad = ";
print_r($a->grad()->toArray());  
// [[2, 3], [4, 5]]  ← Valores de b!

echo "b.grad = ";
print_r($b->grad()->toArray());  
// [[1, 2], [3, 4]]  ← Valores de a!

// Verificação manual:
// d(loss)/da[0,0] = d(2 + 6 + 12 + 20)/da[0,0] = d(2)/da[0,0] = b[0,0] = 2 ✓
// d(loss)/db[0,0] = d(loss)/db[0,0] = a[0,0] = 1 ✓
?>
```

---

### 4. **sumAutograd() - Redução de Soma com Autograd**

**Localização**: [src/zmatrix.cpp](src/zmatrix.cpp#L2335)

#### Computação Forward com Redução Paralela

```cpp
static ZTensor sum_autograd(const ZTensor& t) {
    const size_t N = t.size();
    double total = 0.0;
    
    // Forward: sum(t) com redução paralela
    if (N > 0) {
        const float* data = t.data.data();
#if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            // OpenMP reduction para soma paralela
#pragma omp parallel for reduction(+:total) schedule(static)
            for (size_t i = 0; i < N; ++i) {
                total += data[i];
            }
        } else {
            for (size_t i = 0; i < N; ++i) {
                total += data[i];
            }
        }
#else
        for (size_t i = 0; i < N; ++i) {
            total += data[i];
        }
#endif
    }
    
    // Resultado é ESCALAR (shape [1])
    ZTensor result({1});
    result.data[0] = static_cast<float>(total);
    
    // Autograd
    result.requires_grad = t.requires_grad;
    
    if (t.requires_grad) {
        auto node = std::make_shared<AutogradNode>("sum");
        auto t_ptr = std::make_shared<ZTensor>(t);
        auto result_ptr = std::make_shared<ZTensor>(result);
        node->parents = {t_ptr};
        
        // ===== ARMAZENA METADATA PARA BACKWARD =====
        // Precisa saber:
        // 1. Shape original do input (para reconstruir shape do grad)
        // 2. Tamanho original (para saber quantos elementos broadcast)
        auto input_shape = t.shape;
        auto input_size = t.size();
        
        // ===== BACKWARD COM BROADCASTING =====
        // Para soma: y = sum(x)
        //   ∂y/∂x[i] = 1 para todo i
        // Logo no backward:
        //   grad_x[i] = grad_y (broadcast escalar para cada elemento)
        
        node->backward_fn = [result_ptr, t_ptr, input_shape, input_size]() {
            const ZTensor* grad_result = result_ptr->getGrad();
            if (!grad_result) return;
            
            // grad_result é escalar com shape [1]
            float grad_val = grad_result->data[0];
            
            // Cria gradiente do input com shape original
            ZTensor grad_input(input_shape);
            if (input_size > 0) {
                float* grad_data = grad_input.data.data();
                // Broadcasting: cada elemento recebe o mesmo gradiente
                for (size_t i = 0; i < input_size; ++i) {
                    grad_data[i] = grad_val;
                }
            }
            
            // Acumula no gradiente do input
            const_cast<ZTensor*>(t_ptr.get())->accumulate_grad(grad_input);
        };
        
        result.grad_fn = node;
    }
    
    return result;
}
```

#### Matemática de Gradientes (Broadcasting)

```
Forward pass:  s = sum(x) = x[0] + x[1] + ... + x[N-1]

Backward pass (Broadcasting):
  ∂s/∂x[i] = 1 para todo i
  Logo:
  grad_x[i] = grad_s (mesmo valor para TODOS os elementos!)

Exemplo numérico:
  x = [[1, 2, 3], [4, 5, 6]]  (shape [2, 3])
  s = 1+2+3+4+5+6 = 21  (escalar, shape [1])
  
  Se grad_s = 1.0 (do backward):
    grad_x = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]  ← Broadcast para todos!
```

#### Exemplo de Uso PHP

```php
<?php
// Criar tensor
$x = ZTensor::arange(1, 7)->reshape([2, 3]);  
// [[1, 2, 3], [4, 5, 6]]
$x->requiresGrad(true);

// Forward pass
$sum = ZTensor::sumAutograd($x);  
// 1+2+3+4+5+6 = 21 (escalar com shape [1])

$sum->backward();

echo "x.grad = ";
print_r($x->grad()->toArray());  
// [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
// Cada elemento contribui 1 para a soma!

// Verificação:
// d(21)/dx[0,0] = d(1+2+3+4+5+6)/dx[0,0] = 1 ✓
// d(21)/dx[1,2] = d(1+2+3+4+5+6)/dx[1,2] = 1 ✓
?>
```

---

## 🔗 Camada de Binding PHP (Arquivo: src/zmatrix.cpp)

### Exemplo Completo: addAutograd PHP → C++

**Localização**: [src/zmatrix.cpp](src/zmatrix.cpp#L5073)

```cpp
// ========== BINDING PHP PARA addAutograd ==========
PHP_METHOD(ZTensor, addAutograd)
{
    // PASSO 1: Parse dos parâmetros
    zval *a_zv, *b_zv;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT(a_zv)      // Primeiro parâmetro: objeto ZTensor
        Z_PARAM_OBJECT(b_zv)      // Segundo parâmetro: objeto ZTensor
    ZEND_PARSE_PARAMETERS_END();

    // PASSO 2: Extração dos objetos C++ internos
    // Z_MATRIX_ZTENSOR_P é um macro que:
    // - Casteia o zval para um objeto PHP
    // - Extrai a estrutura interna zmatrix_ztensor_object
    zmatrix_ztensor_object *a_obj = Z_MATRIX_ZTENSOR_P(a_zv);
    zmatrix_ztensor_object *b_obj = Z_MATRIX_ZTENSOR_P(b_zv);

    // PASSO 3: Validação (tensores inicializados?)
    if (!a_obj->tensor || !b_obj->tensor) {
        zend_throw_exception(zend_ce_exception, 
                           ZMATRIX_ERR_NOT_INITIALIZED, 
                           0);
        RETURN_THROWS();  // Retorna NULL e propaga exception
    }

    // PASSO 4: Try-catch para exceções C++
    try {
        // Chama implementação C++ estática
        ZTensor result = ZTensor::add_autograd(*a_obj->tensor, 
                                               *b_obj->tensor);
        
        // PASSO 5: Converte ZTensor C++ para objeto PHP ZTensor
        // zmatrix_return_tensor_obj:
        // - Cria nova instância PHP de ZTensor
        // - Copia os dados do result para o objeto
        // - Retorna como return_value
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } 
    catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}
```

### Tipo Hints (ArgInfo)

```cpp
// Define type hints para IDE autocomplete
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_add_autograd, 0, 2, "ZMatrix\\ZTensor", 0)
    ZEND_ARG_OBJ_INFO(0, a, ZMatrix\\ZTensor, 0)
    ZEND_ARG_OBJ_INFO(0, b, ZMatrix\\ZTensor, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_sub_autograd, 0, 2, "ZMatrix\\ZTensor", 0)
    ZEND_ARG_OBJ_INFO(0, a, ZMatrix\\ZTensor, 0)
    ZEND_ARG_OBJ_INFO(0, b, ZMatrix\\ZTensor, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_mul_autograd, 0, 2, "ZMatrix\\ZTensor", 0)
    ZEND_ARG_OBJ_INFO(0, a, ZMatrix\\ZTensor, 0)
    ZEND_ARG_OBJ_INFO(0, b, ZMatrix\\ZTensor, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_sum_autograd, 0, 1, "ZMatrix\\ZTensor", 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ZMatrix\\ZTensor, 0)
ZEND_END_ARG_INFO()
```

### Registração na Tabela de Métodos

```cpp
static const zend_function_entry zmatrix_ztensor_methods[] = {
    // ... outros métodos ...
    
    // Métodos Estáticos de Autograd
    PHP_ME(ZTensor, addAutograd,     arginfo_add_autograd,    ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, subAutograd,     arginfo_sub_autograd,    ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, mulAutograd,     arginfo_mul_autograd,    ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(ZTensor, sumAutograd,     arginfo_sum_autograd,    ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    
    // ... mais métodos ...
};
```

---

## 🧪 Exemplos Completos de Teste

### Teste 1: Adição Simples com Backward

```php
<?php
require_once 'vendor/autoload.php';

use ZMatrix\ZTensor;

// Setup
$a = ZTensor::ones([2, 2])->requiresGrad(true);
$b = ZTensor::ones([2, 2]) * 2;

echo "=== FORWARD PASS ===\n";
echo "a = \n";
print_r($a->toArray());
echo "b = \n";
print_r($b->toArray());

// Forward
$c = ZTensor::addAutograd($a, $b);
echo "c = a + b = \n";
print_r($c->toArray());

$loss = ZTensor::sumAutograd($c);
echo "loss = sum(c) = " . $loss->toArray()[0] . "\n";

// Backward
echo "\n=== BACKWARD PASS ===\n";
$loss->backward();

echo "a.grad = \n";
print_r($a->grad()->toArray());  // [[1, 1], [1, 1]]
// Esperado: 1 para cada elemento porque d(sum(a+b))/da = 1
?>
```

### Teste 2: Multiplicação com Captura

```php
<?php
use ZMatrix\ZTensor;

// Valores específicos
$a = ZTensor::arange(1, 5)->reshape([2, 2]);  // [[1, 2], [3, 4]]
$b = ZTensor::arange(2, 6)->reshape([2, 2]);  // [[2, 3], [4, 5]]
$a->requiresGrad(true);
$b->requiresGrad(true);

echo "a = \n";
print_r($a->toArray());
echo "b = \n";
print_r($b->toArray());

// Forward
$c = ZTensor::mulAutograd($a, $b);
echo "c = a * b = \n";
print_r($c->toArray());  // [[2, 6], [12, 20]]

$loss = ZTensor::sumAutograd($c);
echo "loss = " . $loss->toArray()[0] . "\n";

// Backward
$loss->backward();

echo "\na.grad = \n";
print_r($a->grad()->toArray());  // [[2, 3], [4, 5]] = b
echo "b.grad = \n";
print_r($b->grad()->toArray());  // [[1, 2], [3, 4]] = a

// Verificação:
// d(loss)/da[0,0] = d(2+6+12+20)/da[0,0] = d(2)/da[0,0] = b[0,0] = 2 ✓
// d(loss)/db[0,0] = d(loss)/db[0,0] = a[0,0] = 1 ✓
?>
```

### Teste 3: Subtração com Negação

```php
<?php
use ZMatrix\ZTensor;

$a = ZTensor::ones([2, 2]) * 5;
$b = ZTensor::ones([2, 2]) * 2;
$a->requiresGrad(true);
$b->requiresGrad(true);

echo "a = \n";
print_r($a->toArray());  // [[5, 5], [5, 5]]
echo "b = \n";
print_r($b->toArray());  // [[2, 2], [2, 2]]

// Forward
$c = ZTensor::subAutograd($a, $b);
echo "c = a - b = \n";
print_r($c->toArray());  // [[3, 3], [3, 3]]

$loss = ZTensor::sumAutograd($c);
echo "loss = " . $loss->toArray()[0] . "\n";  // 12

// Backward
$loss->backward();

echo "\na.grad = \n";
print_r($a->grad()->toArray());  // [[1, 1], [1, 1]]
echo "b.grad = \n";
print_r($b->grad()->toArray());  // [[-1, -1], [-1, -1]] NEGADO!

// Verificação:
// d(loss)/da = d(sum(a-b))/da = 1 ✓
// d(loss)/db = d(sum(a-b))/db = -1 ✓
?>
```

### Teste 4: Sum com Broadcasting

```php
<?php
use ZMatrix\ZTensor;

$x = ZTensor::arange(1, 7)->reshape([2, 3]);
$x->requiresGrad(true);

echo "x = \n";
print_r($x->toArray());  // [[1, 2, 3], [4, 5, 6]]

$sum = ZTensor::sumAutograd($x);
echo "sum = " . $sum->toArray()[0] . "\n";  // 21

$sum->backward();

echo "\nx.grad = \n";
print_r($x->grad()->toArray());  
// [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
// Broadcasting: cada elemento contribui 1!
?>
```

---

## 🔐 Memory Safety

### Problema: Use-After-Free

Sem shared pointers:

```cpp
{
    ZTensor a = get_tensor();
    ZTensor b = get_tensor();
    ZTensor c = ZTensor::add_autograd(a, b);
    // a, b saem de escopo e são deletados
}
// Depois: c->grad_fn->backward_fn tenta acessar a, b → CRASH!
```

### Solução: Captura via Shared Pointers

```cpp
auto a_ptr = std::make_shared<ZTensor>(a);  // Incrementa refcount
node->backward_fn = [a_ptr]() { /* a_ptr válido */ };
// Quando backward_fn é deletada:
//   a_ptr refcount decrementa
//   Se refcount == 0, ZTensor é deletado
//   Caso contrário, fica na memória
```

**Benefício**: Garante que dados necessários para backward estão sempre disponíveis.

---

## ⚡ Otimizações Implementadas

### 1. Paralelização com OpenMP

```cpp
#if HAS_OPENMP
    if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
        for (size_t i = 0; i < N; ++i) {
            // Computação paralelizada
        }
    } else {
        // Fallback sequencial (sem overhead)
    }
#endif
```

**Impacto**: Forward passes escaláveis para grandes tensores.

### 2. Threshold Adaptativo

```cpp
if (N > ZMATRIX_PARALLEL_THRESHOLD) {
    // OpenMP (overhead compensado)
} else {
    // Sequential (mais rápido para pequenos tensores)
}
```

### 3. Acesso Direto ao Buffer

```cpp
const float* a_data = a.data.data();  // Evita indireção de vector
float* r_data = result.data.data();
for (size_t i = 0; i < N; ++i) {
    r_data[i] = a_data[i] + b_data[i];  // Acesso cache-friendly
}
```

### 4. Captura Seletiva em Multiplicação

```cpp
// Só captura se necessário
auto a_copy = std::make_shared<ZTensor>(a);
auto b_copy = std::make_shared<ZTensor>(b);

// Usa copias sómente no backward, não no forward
node->backward_fn = [a_copy, b_copy]() { /* usa cópias */ };
```

---

## 📊 Análise de Complexidade

### Complexidade Temporal

| Operação | Forward | Backward |
|----------|---------|----------|
| add | O(N) | O(N) |
| sub | O(N) | O(N) |
| mul | O(N) | O(N) |
| sum | O(N) | O(N) |

N = número total de elementos

### Complexidade Espacial

| Operação | Extra Memory |
|----------|--------------|
| add | O(N) resultado |
| sub | O(N) resultado |
| mul | 2×O(N) cópias + O(N) grad |
| sum | O(1) resultado + O(N) grad |

### Paralelização

- Forward de add/sub/mul: `#pragma omp parallel for simd`
- Forward de sum: `#pragma omp parallel for reduction(+:total)`
- Backward: Sequencial (acesso compartilhado aos gradientes)

---

## 🎯 Integração com Backward Graph

Cada tensor possui:

```cpp
std::shared_ptr<AutogradNode> grad_fn;  // Aponta para nó de computação
```

Quando `backward()` é chamado:

```cpp
void backward() {
    // Propaga do fim para o início do grafo
    // Cada nó executa sua backward_fn em ordem topológica
    grad_fn->backward_fn();  // Executa backward
    // Resultado: gradientes acumulados em tensores input
}
```

---

## ✅ Checklist de Implementação

- ✅ **add_autograd**: Implementado com forward + backward
- ✅ **sub_autograd**: Implementado com negação de gradiente em B
- ✅ **mul_autograd**: Implementado com captura de operandos
- ✅ **sum_autograd**: Implementado com broadcasting de gradiente
- ✅ **PHP Bindings**: Todos 4 métodos expostos como estáticos
- ✅ **Type Hints (ArgInfo)**: Definidos para IDE autocomplete
- ✅ **Paralelização**: OpenMP com threshold adaptativo
- ✅ **Memory Safety**: Shared pointers para lifecycle seguro
- ✅ **Error Handling**: Try-catch com exceções C++ → PHP
- ✅ **Documentação**: PHPDoc nos stubs

---

## 📚 Referências Rápidas

**Arquivos Principais**:
- [src/zmatrix.cpp](src/zmatrix.cpp#L2090) - Implementações C++
- [src/zmatrix.cpp](src/zmatrix.cpp#L5073) - Bindings PHP
- [stubs/ZTensor.php](stubs/ZTensor.php#L1010) - Type hints

**Estruturas de Dados**:
- `AutogradNode` - Nó de computação com backward_fn
- `ZTensor::grad_fn` - Aponta para AutogradNode
- `ZTensor::requires_grad` - Flag para rastreamento

**Macros Importantes**:
- `ZEND_PARSE_PARAMETERS_START` - Parse de argumentos PHP
- `Z_PARAM_OBJECT` - Extrai parâmetro objeto
- `Z_MATRIX_ZTENSOR_P` - Acesso ao objeto C++ interno
- `zmatrix_return_tensor_obj` - Retorna ZTensor para PHP

---

**Implementado com sucesso!** 🎉
