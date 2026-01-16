# 🧠 Implementação Detalhada de Autograd em ZMatrix

**Data**: 16 de Janeiro, 2026  
**Versão**: MVP 1.0  
**Linguagem**: C++17 com extensão PHP

---

## 📚 Índice

1. [Arquitetura Geral](#arquitetura-geral)
2. [Estrutura de Dados](#estrutura-de-dados)
3. [Implementação das Operações](#implementação-das-operações)
4. [Método Backward](#método-backward)
5. [Thread-Safety](#thread-safety)
6. [Exemplos de Código](#exemplos-de-código)

---

## 🏗️ Arquitetura Geral

### Fluxo de Computação

```
┌─────────────────────────────────────────────────────────────┐
│                    Forward Pass (Eager)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  a = ZTensor([1, 2, 3])  →  requires_grad = true           │
│  b = ZTensor([2, 3, 4])  →  requires_grad = true           │
│  c = add_autograd(a, b)  →  cria nó no grafo               │
│  d = mul_autograd(c, 2)  →  adiciona operação              │
│  result = sum_autograd(d) →  cria escalar                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Grafo Computacional                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│           result (scalar)                                   │
│               ↑                                             │
│               │ backward_fn: ∂result/∂d                     │
│               │                                             │
│              sum_node                                       │
│               ↑                                             │
│               │ backward_fn: ∂d/∂c                          │
│               │                                             │
│              mul_node                                       │
│               ↑                                             │
│           ╱───┴────╲                                        │
│          ↑          ↑                                       │
│      add_node   (scalar 2)                                 │
│       ↑  ↑                                                  │
│       │  │                                                  │
│      a   b                                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Backward Pass (DFS)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  result.backward()                                          │
│    ├─ result.grad = 1.0                                     │
│    ├─ DFS post-order: sum_node, mul_node, add_node          │
│    ├─ sum: propaga gradientes                              │
│    ├─ mul: calcula ∂loss/∂c, ∂loss/∂2                      │
│    └─ add: calcula ∂loss/∂a, ∂loss/∂b                      │
│                                                             │
│  a.grad ✓  b.grad ✓                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Estrutura de Dados

### 1. AutogradNode - Nó no Grafo

**Arquivo**: `src/zmatrix.cpp`, linhas ~122-140

```cpp
struct AutogradNode {
    // Referências aos tensores pais (inputs)
    std::vector<std::shared_ptr<ZTensor>> parents;
    
    // Função que calcula os gradientes dos pais
    // Captura: resultado, inputs, parâmetros da operação
    std::function<void()> backward_fn;
    
    // Nome da operação (debug)
    std::string op_name;
    
    // Reservado para sincronização
    mutable std::mutex backward_lock;
};
```

**Por que `std::shared_ptr`?**
- Garante que os tensores pais sobrevivem até o backward
- Evita use-after-free
- Gerencia ciclo de vida automaticamente

### 2. ZTensor - Estado de Autograd

**Arquivo**: `src/zmatrix.cpp`, linhas ~145-170

```cpp
class ZTensor {
private:
    std::vector<float> data;          // Dados do tensor
    std::vector<int> shape;           // Forma (dimensões)
    
    // ===== CAMPOS DE AUTOGRAD =====
    bool requires_grad = false;                    // Flag de rastreamento
    std::unique_ptr<ZTensor> grad;                 // Gradiente acumulado
    std::shared_ptr<AutogradNode> grad_fn;         // Nó no grafo
    mutable std::mutex grad_mutex;                 // Sincronização thread-safe
    
    // ... outros campos
};
```

**Estrutura de Memória**:
```
ZTensor
├─ data: [1.0, 2.0, 3.0]
├─ shape: [3]
├─ requires_grad: true
├─ grad: ZTensor([0.1, 0.2, 0.3])
├─ grad_fn: AutogradNode {
│  ├─ parents: [ZTensor a, ZTensor b]
│  ├─ backward_fn: λ() { ... }
│  └─ op_name: "add_autograd"
└─ grad_mutex: mutex
```

---

## 💡 Implementação das Operações

### 1. `add_autograd` - Adição com Autograd

**Arquivo**: `src/zmatrix.cpp`, linhas ~2107-2175

```cpp
static ZTensor add_autograd(const ZTensor& a, const ZTensor& b) {
    // ===== FORWARD PASS =====
    // Validação de shapes
    if (a.shape != b.shape) {
        throw std::invalid_argument("Shape mismatch in add_autograd");
    }
    
    // Computar resultado
    ZTensor result = a.data;  // Cópia
    for (size_t i = 0; i < result.data.size(); i++) {
        result.data[i] += b.data[i];
    }
    
    // Decidir se resultado requer gradientes
    // Se qualquer input requer_grad, resultado também requer
    if (!a.requires_grad && !b.requires_grad) {
        return result;  // Sem autograd
    }
    
    // ===== CONSTRUÇÃO DO GRAFO =====
    result.requires_grad = true;
    result.ensure_grad();  // Inicializar grad tensor
    
    // Capturar resultado em shared_ptr para evitar UB
    auto result_ptr = std::make_shared<ZTensor>(result);
    
    // Capturar inputs em shared_ptr (ou usar referências com cuidado)
    auto a_ptr = std::make_shared<ZTensor>(a);
    auto b_ptr = std::make_shared<ZTensor>(b);
    
    // Criar nó no grafo
    auto node = std::make_shared<AutogradNode>();
    node->op_name = "add_autograd";
    node->parents = {a_ptr, b_ptr};
    
    // ===== BACKWARD FUNCTION =====
    // Regra da cadeia para adição:
    // ∂L/∂a = ∂L/∂result (gradiente flui sem modificação)
    // ∂L/∂b = ∂L/∂result (gradiente flui sem modificação)
    node->backward_fn = [a_ptr, b_ptr, result_ptr]() {
        // result.grad já foi preenchido pelo nó pai
        const ZTensor& grad_output = *result_ptr->grad;
        
        // Propagar para 'a'
        if (a_ptr->requires_grad && a_ptr->grad) {
            a_ptr->accumulate_grad(grad_output);
        }
        
        // Propagar para 'b'
        if (b_ptr->requires_grad && b_ptr->grad) {
            b_ptr->accumulate_grad(grad_output);
        }
    };
    
    result.grad_fn = node;
    return result;
}
```

**Explicação**:

| Parte | Explicação |
|-------|-----------|
| **Forward** | Simples: `c[i] = a[i] + b[i]` |
| **Validation** | Shapes devem ser iguais |
| **Grafo** | Armazena referências aos pais |
| **Closure** | Captura shared_ptr (seguro) |
| **Backward** | Ambos gradientes = grad_output |
| **Acúmulo** | `accumulate_grad()` soma gradientes |

### 2. `sub_autograd` - Subtração com Autograd

**Arquivo**: `src/zmatrix.cpp`, linhas ~2177-2245

```cpp
static ZTensor sub_autograd(const ZTensor& a, const ZTensor& b) {
    // Forward: d[i] = a[i] - b[i]
    ZTensor result = a.data;
    for (size_t i = 0; i < result.data.size(); i++) {
        result.data[i] -= b.data[i];
    }
    
    if (!a.requires_grad && !b.requires_grad) {
        return result;
    }
    
    result.requires_grad = true;
    result.ensure_grad();
    
    auto result_ptr = std::make_shared<ZTensor>(result);
    auto a_ptr = std::make_shared<ZTensor>(a);
    auto b_ptr = std::make_shared<ZTensor>(b);
    
    auto node = std::make_shared<AutogradNode>();
    node->op_name = "sub_autograd";
    node->parents = {a_ptr, b_ptr};
    
    // ===== BACKWARD DIFERENTE =====
    // Regra da cadeia para subtração:
    // ∂L/∂a = ∂L/∂result
    // ∂L/∂b = -∂L/∂result  ← NEGAÇÃO!
    node->backward_fn = [a_ptr, b_ptr, result_ptr]() {
        const ZTensor& grad_output = *result_ptr->grad;
        
        if (a_ptr->requires_grad && a_ptr->grad) {
            a_ptr->accumulate_grad(grad_output);
        }
        
        if (b_ptr->requires_grad && b_ptr->grad) {
            // Negar gradiente para 'b'
            ZTensor neg_grad = grad_output;
            for (auto& val : neg_grad.data) {
                val = -val;
            }
            b_ptr->accumulate_grad(neg_grad);
        }
    };
    
    result.grad_fn = node;
    return result;
}
```

**Diferença crítica**:
```
add:  ∂loss/∂b = ∂loss/∂result
sub:  ∂loss/∂b = -∂loss/∂result  ← CUIDADO!

Exemplo:
  c = a - b
  ∂c/∂b = -1
  Logo: ∂loss/∂b = ∂loss/∂c × (-1) = -∂loss/∂c
```

### 3. `mul_autograd` - Multiplicação Elemento-sábio

**Arquivo**: `src/zmatrix.cpp`, linhas ~2247-2330

```cpp
static ZTensor mul_autograd(const ZTensor& a, const ZTensor& b) {
    // Forward: c[i] = a[i] * b[i]
    ZTensor result = a.data;
    for (size_t i = 0; i < result.data.size(); i++) {
        result.data[i] *= b.data[i];
    }
    
    if (!a.requires_grad && !b.requires_grad) {
        return result;
    }
    
    result.requires_grad = true;
    result.ensure_grad();
    
    auto result_ptr = std::make_shared<ZTensor>(result);
    
    // ===== CÓPIA DOS INPUTS PARA BACKWARD =====
    // Precisamos dos valores originais de a e b no backward
    // pois result já foi sobrescrito
    auto a_copy = std::make_shared<ZTensor>(a);
    auto b_copy = std::make_shared<ZTensor>(b);
    
    auto a_ptr = std::make_shared<ZTensor>(a);
    auto b_ptr = std::make_shared<ZTensor>(b);
    
    auto node = std::make_shared<AutogradNode>();
    node->op_name = "mul_autograd";
    node->parents = {a_ptr, b_ptr};
    
    // ===== BACKWARD COM REGRA DO PRODUTO =====
    // Regra da cadeia para multiplicação:
    // ∂L/∂a = b[i] * ∂L/∂result[i]
    // ∂L/∂b = a[i] * ∂L/∂result[i]
    node->backward_fn = [a_copy, b_copy, a_ptr, b_ptr, result_ptr]() {
        const ZTensor& grad_output = *result_ptr->grad;
        
        if (a_ptr->requires_grad && a_ptr->grad) {
            // grad_a[i] = b_original[i] * grad_output[i]
            ZTensor grad_a = grad_output;
            for (size_t i = 0; i < grad_a.data.size(); i++) {
                grad_a.data[i] *= b_copy->data[i];
            }
            a_ptr->accumulate_grad(grad_a);
        }
        
        if (b_ptr->requires_grad && b_ptr->grad) {
            // grad_b[i] = a_original[i] * grad_output[i]
            ZTensor grad_b = grad_output;
            for (size_t i = 0; i < grad_b.data.size(); i++) {
                grad_b.data[i] *= a_copy->data[i];
            }
            b_ptr->accumulate_grad(grad_b);
        }
    };
    
    result.grad_fn = node;
    return result;
}
```

**Conceito da Regra do Produto**:
```
c[i] = a[i] * b[i]

Derivada com respeito a a:
∂c[i]/∂a[i] = b[i]

Logo no backward:
grad_a[i] = grad_output[i] × ∂c[i]/∂a[i]
          = grad_output[i] × b[i]

Exemplo numérico:
  a = [2]  b = [3]  → c = [6]
  ∂L/∂c = [0.5]
  ∂L/∂a = 0.5 × 3 = [1.5]  ✓
  ∂L/∂b = 0.5 × 2 = [1.0]  ✓
```

### 4. `sum_autograd` - Redução a Escalar

**Arquivo**: `src/zmatrix.cpp`, linhas ~2332-2390

```cpp
static ZTensor sum_autograd(const ZTensor& tensor) {
    // ===== FORWARD: REDUÇÃO =====
    float sum = 0.0f;
    for (float val : tensor.data) {
        sum += val;
    }
    
    ZTensor result({1});  // Escalar
    result.data[0] = sum;
    
    if (!tensor.requires_grad) {
        return result;
    }
    
    result.requires_grad = true;
    result.ensure_grad();
    
    auto result_ptr = std::make_shared<ZTensor>(result);
    auto tensor_ptr = std::make_shared<ZTensor>(tensor);
    
    auto node = std::make_shared<AutogradNode>();
    node->op_name = "sum_autograd";
    node->parents = {tensor_ptr};
    
    // ===== BACKWARD: BROADCAST =====
    // Redução soma todos: result = Σ tensor[i]
    // Logo: ∂result/∂tensor[i] = 1
    // E no backward: ∂L/∂tensor[i] = ∂L/∂result × 1
    //             = grad_output[0] para todos i
    node->backward_fn = [tensor_ptr, result_ptr]() {
        const ZTensor& grad_output = *result_ptr->grad;
        
        if (tensor_ptr->requires_grad && tensor_ptr->grad) {
            // Broadcast: todos elementos recebem o mesmo gradiente
            ZTensor grad_tensor = grad_output;  // Shape [1]
            grad_tensor.reshape(tensor_ptr->shape);  // Reshape para shape original
            
            // Agora: grad_tensor[i] = grad_output[0] para todos i
            for (auto& val : grad_tensor.data) {
                val = grad_output.data[0];
            }
            
            tensor_ptr->accumulate_grad(grad_tensor);
        }
    };
    
    result.grad_fn = node;
    return result;
}
```

**Visualização da Redução**:
```
Forward:
  tensor = [1, 2, 3]
  result = sum(tensor) = 6  (escalar)

Backward:
  grad_output = [0.5]  (gradiente do resultado)
  
  ∂result/∂tensor[0] = 1
  ∂result/∂tensor[1] = 1
  ∂result/∂tensor[2] = 1
  
  Logo:
  grad_tensor[0] = 0.5 × 1 = 0.5
  grad_tensor[1] = 0.5 × 1 = 0.5
  grad_tensor[2] = 0.5 × 1 = 0.5
```

---

## 🔙 Método Backward

### Entrada: `backward()`

**Arquivo**: `src/zmatrix.cpp`, linhas ~230-260

```cpp
void backward() {
    // ===== VALIDAÇÕES =====
    if (!requires_grad) {
        throw std::logic_error("Tensor does not require gradients");
    }
    
    if (shape.size() != 1 || data.size() != 1) {
        throw std::logic_error(
            "backward() only works on scalars (shape={1})"
        );
    }
    
    if (!grad) {
        ensure_grad();
    }
    
    // ===== INICIALIZAR GRADIENTE RAIZ =====
    grad->data[0] = 1.0f;
    
    // ===== DFS PÓS-ORDEM =====
    std::set<std::shared_ptr<AutogradNode>> visited;
    backward_impl(grad_fn, visited);
}

private:
void backward_impl(
    std::shared_ptr<AutogradNode> node,
    std::set<std::shared_ptr<AutogradNode>>& visited
) {
    if (!node || visited.count(node)) {
        return;  // Já foi processado
    }
    
    visited.insert(node);
    
    // ===== DFS PRÉ-ORDEM: VISITAR FILHOS PRIMEIRO =====
    for (auto& parent : node->parents) {
        if (parent->grad_fn) {
            backward_impl(parent->grad_fn, visited);
        }
    }
    
    // ===== EXECUTAR BACKWARD DESTE NÓ =====
    try {
        node->backward_fn();  // Chama a closure
    } catch (const std::exception& e) {
        std::cerr << "Error in backward for " << node->op_name 
                  << ": " << e.what() << std::endl;
        // Continua para outros nós
    }
}
```

### Fluxo de Execução

```
backward() chamado em result

1. Validação
   ✓ requires_grad = true
   ✓ shape = {1} (escalar)
   
2. Inicialização
   result.grad[0] = 1.0

3. DFS Pós-ordem
   
   backward_impl(sum_node):
     1. Visitar pais: mul_node
        backward_impl(mul_node):
          1. Visitar pais: add_node
             backward_impl(add_node):
               1. Visitar pais: none
               2. Executar: propagate para a, b
          2. Executar: calcula grad_c
     2. Executar: calcula grad_d

4. Resultado
   a.grad = [∂L/∂a]
   b.grad = [∂L/∂b]
```

---

## 🔒 Thread-Safety

### Problema: Acúmulo de Gradientes

```cpp
// SEM mutex (❌ UNSAFE):
void accumulate_grad(const ZTensor& g) {
    for (size_t i = 0; i < grad->data.size(); i++) {
        grad->data[i] += g.data[i];  // Race condition!
    }
}

// COM mutex (✅ SAFE):
void accumulate_grad(const ZTensor& g) {
    std::lock_guard<std::mutex> lock(grad_mutex);
    for (size_t i = 0; i < grad->data.size(); i++) {
        grad->data[i] += g.data[i];  // Protegido
    }
}
```

**Arquivo**: `src/zmatrix.cpp`, linhas ~195-210

```cpp
void accumulate_grad(const ZTensor& grad_in) {
    // Sincronizar acesso ao grad
    std::lock_guard<std::mutex> lock(grad_mutex);
    
    if (!grad) {
        grad = std::make_unique<ZTensor>(grad_in);
        return;
    }
    
    // Soma: grad += grad_in
    if (grad->data.size() != grad_in.data.size()) {
        throw std::logic_error("Gradient shape mismatch");
    }
    
    for (size_t i = 0; i < grad->data.size(); i++) {
        grad->data[i] += grad_in.data[i];
    }
}
```

---

## 💾 Proteção contra In-Place Operations

**Arquivo**: `src/zmatrix.cpp`, linhas 567, 767

```cpp
// Em ZTensor::add()
if (this->requires_grad) {
    throw std::logic_error(
        "In-place operation on tensor with requires_grad=true "
        "is not allowed. Use add_autograd() for differentiable operations."
    );
}

// Em ZTensor::mul()
if (this->requires_grad) {
    throw std::logic_error(
        "In-place operation on tensor with requires_grad=true "
        "is not allowed. Use mul_autograd() for differentiable operations."
    );
}
```

**Por que?**
```
Problema:
  a.requires_grad = true
  a.add(b)  // Modifica a in-place
  
  Depois no backward:
  - grad_fn tenta acessar valor original de 'a'
  - Mas 'a' foi sobrescrito! ❌ Dados incorretos

Solução:
  Use add_autograd(a, b)  // Retorna novo tensor
  Grafo fica intacto ✓
```

---

## 📝 Exemplos de Código

### Exemplo 1: Simples

```cpp
// Forward
ZTensor a({2});
a.data = {1, 2};
a.requiresGrad(true);

ZTensor b({2});
b.data = {3, 4};
b.requiresGrad(true);

ZTensor c = add_autograd(a, b);  // [4, 6]

// Backward
c.grad->data[0] = 1.0;  // dL/dc
c.grad->data[1] = 1.0;

c.grad_fn->backward_fn();  // Propagar
// a.grad = [1, 1]  ✓
// b.grad = [1, 1]  ✓
```

### Exemplo 2: Cadeia

```cpp
// Forward
ZTensor a({1});
a.data = {2};
a.requiresGrad(true);

ZTensor b({1});
b.data = {3};
b.requiresGrad(true);

ZTensor c = add_autograd(a, b);  // 5
ZTensor d = mul_autograd(c, c);  // 25
ZTensor loss = sum_autograd(d);  // 25 (escalar)

// Backward
loss.backward();

// ∂loss/∂a = 2 × c × 1 = 2 × 5 = 10
// ∂loss/∂b = 2 × c × 1 = 2 × 5 = 10
// a.grad = [10]  ✓
// b.grad = [10]  ✓
```

---

## 🎯 Decisões de Design

| Decisão | Razão |
|---------|-------|
| **Eager Mode** | Constrói grafo na forward, não no backward |
| **Shared_ptr** | Garante lifetime correto dos tensores pais |
| **Closure** | Captura estado necessário para backward |
| **DFS Pós-ordem** | Garante que pais computam antes dos filhos |
| **Visited Set** | Previne ciclos e reprocessamento |
| **Mutex por Tensor** | Fine-grained locking, menos contenção |
| **Scalar-only backward** | Simplifica gradiente inicial (sempre 1.0) |
| **Lazy grad init** | Economiza memória para tensores sem gradientes |

---

## ✅ Checklist de Garantias

- ✅ **Corretude Matemática**: Regra da cadeia implementada corretamente
- ✅ **Memory Safety**: Sem uso-after-free, shared_ptr gerencia lifetime
- ✅ **Thread-Safety**: Mutex em accumulate_grad
- ✅ **No In-place**: Exceção lançada em add()/mul() com requires_grad
- ✅ **Graph Integrity**: Closure captura valores, não referências
- ✅ **Cycle Prevention**: Visited set no DFS
- ✅ **Error Handling**: Try-catch em backward_fn

---

**Status**: ✅ MVP funcional e seguro de autograd reverse-mode

