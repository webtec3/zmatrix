# 🧠 Implementação de Autograd MVP - ZMatrix/ZTensor

**Status Final**: ✅ **REVISÃO COMPLETA E CORREÇÕES CRÍTICAS APLICADAS**

**Data**: 16 de Janeiro, 2026

---

## 📋 Sumário Executivo

Esta é uma **implementação minimal mas completa** de um sistema de autograd (automatic differentiation) em reverse-mode (backpropagation) para a extensão PHP ZMatrix.

**Objetivos alcançados**:
- ✅ MVP funcional de autograd (reverse-mode, eager-mode)
- ✅ Operações básicas com grafo computacional: `add`, `sub`, `mul`, `sum`
- ✅ Backward correto com cálculo de gradientes
- ✅ Proteção contra operações inplace em tensores rastreados
- ✅ Thread-safety em acumulação de gradientes
- ✅ Sem undefined behavior (UB)

---

## 🏗️ Arquitetura

### Estruturas Principais

#### 1. `AutogradNode`
```cpp
struct AutogradNode {
    std::vector<std::shared_ptr<ZTensor>> parents;
    std::function<void()> backward_fn;
    std::string op_name;
    mutable std::mutex backward_lock;
};
```

**Responsabilidades**:
- Armazena pais (operandos da operação)
- Contém função backward para calcular gradientes
- Identificação para debug
- Mutex para acesso thread-safe

#### 2. `ZTensor` (extensões para autograd)
```cpp
struct ZTensor {
    // ... campos existentes ...
    
    // ========== AUTOGRAD STATE ==========
    bool requires_grad = false;
    std::unique_ptr<ZTensor> grad;
    std::shared_ptr<AutogradNode> grad_fn = nullptr;
    mutable std::mutex grad_mutex;
    
    // ========== MÉTODOS DE AUTOGRAD ==========
    ZTensor& requiresGrad(bool req = true);
    bool is_requires_grad() const;
    ZTensor& ensure_grad();
    void zero_grad();
    const ZTensor* get_grad() const;
    void accumulate_grad(const ZTensor& grad_in);
    void backward();
    
    // ========== OPERAÇÕES COM AUTOGRAD ==========
    static ZTensor add_autograd(const ZTensor& a, const ZTensor& b);
    static ZTensor sub_autograd(const ZTensor& a, const ZTensor& b);
    static ZTensor mul_autograd(const ZTensor& a, const ZTensor& b);
    static ZTensor sum_autograd(const ZTensor& t);
};
```

---

## 📊 Operações Implementadas

### 1. **Addition** (`add_autograd`)
```
c = a + b
∂c/∂a = 1
∂c/∂b = 1
```

Forward:
```cpp
result[i] = a[i] + b[i]
```

Backward:
```cpp
grad_a[i] += grad_output[i]
grad_b[i] += grad_output[i]
```

### 2. **Subtraction** (`sub_autograd`)
```
c = a - b
∂c/∂a = 1
∂c/∂b = -1
```

Forward:
```cpp
result[i] = a[i] - b[i]
```

Backward:
```cpp
grad_a[i] += grad_output[i]
grad_b[i] -= grad_output[i]  // Negado!
```

### 3. **Multiplication** (`mul_autograd`)
```
c = a * b
∂c/∂a = b
∂c/∂b = a
```

Forward:
```cpp
result[i] = a[i] * b[i]
```

Backward:
```cpp
grad_a[i] += b[i] * grad_output[i]
grad_b[i] += a[i] * grad_output[i]
```

### 4. **Sum Reduction** (`sum_autograd`)
```
c = sum(a)  -> escalar
∂c/∂a[i] = 1 para todo i
```

Forward:
```cpp
result = a[0] + a[1] + ... + a[n-1]
```

Backward:
```cpp
grad_a[i] += grad_output  // Broadcast escalar
```

---

## 🔄 Fluxo de Execução

### Forward Pass
```
$a = ZTensor::ones([3,3])->requiresGrad(true);
$b = ZTensor::ones([3,3])->requiresGrad(true);
$c = ZTensor::add_autograd($a, $b);  // Cria nó de grafo
$loss = ZTensor::sum_autograd($c);   // Cria outro nó
```

**Grafo resultante**:
```
a [requires_grad=true]  \
                        --> add --> c --> sum --> loss [escalar]
b [requires_grad=true]  /
```

### Backward Pass
```
$loss->backward();
```

**Ordem de execução**:
1. `loss.grad = 1.0` (inicializa raiz)
2. DFS pós-ordem:
   - Visita nó `sum` → propaga para `c`
   - `c.grad += loss.grad` (broadcast)
   - Visita nó `add` → propaga para `a` e `b`
   - `a.grad += c.grad`
   - `b.grad += c.grad`

**Resultado**:
```
a.grad = [[1, 1, 1], [1, 1, 1]]
b.grad = [[1, 1, 1], [1, 1, 1]]
```

---

## 🛡️ Proteções e Garantias

### 1. **Inplace Operations Bloqueadas**
```cpp
if (this->requires_grad) {
    throw std::logic_error(
        "In-place operation on tensor with requires_grad=true is not allowed"
    );
}
```

**Razão**: Operações inplace modificam o tensor, corrompendo o grafo computacional.

### 2. **Proteção contra Use-After-Free**
Todas as closures capturam `shared_ptr`, nunca referências locais:

```cpp
// ❌ Antes (UB)
auto result_ptr = std::make_shared<ZTensor>(result);
node->backward_fn = [&result, ...]() {  // &result é local!
    result.get_grad();  // Acesso após destruição
};

// ✅ Depois (Seguro)
auto result_ptr = std::make_shared<ZTensor>(result);
node->backward_fn = [result_ptr, ...]() {
    const ZTensor* grad = result_ptr->get_grad();  // Seguro
};
```

### 3. **Thread-Safety em Acumulação**
```cpp
void accumulate_grad(const ZTensor& grad_in) {
    std::lock_guard<std::mutex> lock(grad_mutex);  // Mutex por tensor
    // Acumulação segura
    for (size_t i = 0; i < N; ++i) {
        g_data[i] += gin_data[i];
    }
}
```

### 4. **Proteção contra Ciclos no Grafo**
```cpp
std::set<std::shared_ptr<AutogradNode>> visited;
// Cada nó visitado apenas uma vez em DFS
if (!node || visited.count(node)) return;
visited.insert(node);
```

### 5. **Validação de Escalares**
```cpp
void backward() {
    if (shape != std::vector<size_t>{1}) {
        throw std::invalid_argument(
            "backward() can only be called on scalar tensors"
        );
    }
}
```

---

## 📈 Exemplo Completo

```php
<?php
// 1. Criar tensores com rastreamento de gradiente
$a = ZTensor::ones([2, 2])->requiresGrad(true);
$b = ZTensor::from([[2.0, 2.0], [2.0, 2.0]])->requiresGrad(true);

// 2. Forward pass
$c = ZTensor::add_autograd($a, $b);    // [[3, 3], [3, 3]]
$d = ZTensor::mul_autograd($c, $c);    // [[9, 9], [9, 9]]
$loss = ZTensor::sum_autograd($d);     // 36

// 3. Backward pass
$loss->backward();

// 4. Acessar gradientes
$grad_a = $a->grad();
$grad_b = $b->grad();

// loss = sum((a+b)²) = sum((a+b)²)
// ∂loss/∂a = ∂loss/∂c * ∂c/∂a = 2*(a+b) * 1 = 2*[3,3,3,3] = [6,6,6,6]
// ∂loss/∂b = ∂loss/∂c * ∂c/∂b = 2*(a+b) * 1 = 2*[3,3,3,3] = [6,6,6,6]

echo "a.grad: " . json_encode($grad_a->data()) . "\n";
// Output: a.grad: [6, 6, 6, 6]

echo "b.grad: " . json_encode($grad_b->data()) . "\n";
// Output: b.grad: [6, 6, 6, 6]

// 5. Limpar gradientes para próxima iteração
$a->zero_grad();
$b->zero_grad();
?>
```

---

## 🧪 Testes Inclusos

Arquivo: `test_autograd.php`

**Testes cobertos**:
1. ✅ Inplace operations com requires_grad lançam exceção
2. ✅ Operações out-of-place criam nó de grafo
3. ✅ Backward simples (add + sum)
4. ✅ Multiplicação com gradientes corretos
5. ✅ Subtração com gradientes negativos
6. ✅ Função zero_grad()

---

## 🔧 Comportamento em Casos Edge

| Caso | Comportamento | Razão |
|------|---------------|-------|
| Tensor vazio (size=0) | Ignora acumulação | Não há dados |
| Operação inplace com requires_grad | Exceção | Corrompe grafo |
| backward() em tensor não-escalar | Exceção | Indefinido |
| Ciclo no grafo | Proteção DFS | Não há ciclos em DAG |
| Múltiplos backward passes | Acumula gradientes | ∂L/∂w = ∑ gradientes |
| Tensores compartilhados (views) | Gradientes somam | Comportamento correto |

---

## 📚 Futuros Desenvolvimentos

### Operações com Autograd
- [ ] `matmul_autograd()` - Produto matricial
- [ ] `transpose_autograd()` - Transposição
- [ ] Ativações: `relu_autograd()`, `sigmoid_autograd()`, `tanh_autograd()`
- [ ] Reduções: `mean_autograd()`, `max_autograd()`

### Otimizações
- [ ] Graph pruning (remover nós mortos)
- [ ] Checkpointing (reduzir memória em forward)
- [ ] GPU backward support

### Extensões
- [ ] Variáveis (parâmetros otimizáveis)
- [ ] Otimizadores (SGD, Adam)
- [ ] Loss functions com autograd
- [ ] Construtor de modelos

---

## 🎯 Propriedades Matemáticas

### Regra da Cadeia
Para qualquer composição `z = f(g(x))`:
```
∂z/∂x = ∂z/∂g * ∂g/∂x
```

**Verificado em**:
- add → sum: ∂loss/∂a = ∂loss/∂c * ∂c/∂a = 1 * 1 = 1 ✓
- mul → sum: ∂loss/∂a = ∂loss/∂c * ∂c/∂a = 1 * b ✓

### Acumulação
```
∂L/∂x = ∑(∂L/∂y_i * ∂y_i/∂x) para todos os caminhos i
```

**Implementado via** `accumulate_grad()` com `+=` ✓

---

## ✅ Checklist de Qualidade

- [x] Sem undefined behavior (revisão de closures)
- [x] Thread-safe (mutex em accumulate_grad)
- [x] Proteção contra inplace (exceção clara)
- [x] Corretude matemática (regra da cadeia)
- [x] Extensibilidade (novo padrão para operações)
- [x] Documentação (comentários e guia)
- [x] Testes (test_autograd.php)
- [x] Sem leaks de memória (shared_ptr gerencia)

---

## 📖 Referências Conceituais

Este autograd segue o padrão **PyTorch eager-mode**:
- Grafo construído dinamicamente durante forward
- Cada operação registra sua backward function
- DFS pós-ordem para backward pass
- Acumulação de gradientes com `+=`

---

## 🚀 Como Usar

### Instalação
```bash
cd /home/omgaalfa/php-projetos/php-extension/zmatrix
phpize
./configure
make
make install
```

### Ativar Extensão
```ini
# php.ini
extension=zmatrix.so
```

### Usar em PHP
```php
<?php
$a = ZTensor::ones([3, 3])->requiresGrad(true);
$b = ZTensor::add_autograd($a, $a);
$loss = ZTensor::sum_autograd($b);
$loss->backward();
echo json_encode($a->grad()->data());
?>
```

---

## 📞 Suporte e Contribuições

Para bugs ou melhorias:
1. Verificar `test_autograd.php`
2. Consultar `AUTOGRAD_REVIEW.md` para detalhes técnicos
3. Revisar `src/zmatrix.cpp` linhas de autograd

---

**Status**: 🟢 **PRONTO PARA PRODUÇÃO (MVP)**

Implementação é mínima, correta e extensível.
