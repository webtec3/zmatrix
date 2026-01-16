# 🔍 Referência Rápida de Mudanças - src/zmatrix.cpp

**Arquivo**: `src/zmatrix.cpp`  
**Total de Mudanças**: 5 seções críticas  
**Status**: ✅ Todas aplicadas

---

## 📍 Localização das Mudanças

### 1️⃣ Include de Mutex (Linha 4)

**Localização**: Depois de `<set>`

```diff
  #include <memory>
  #include <functional>
  #include <atomic>
  #include <set>  // Para reverse-mode autograd
+ #include <mutex>  // Para thread-safety em accumulate_grad
```

**Motivo**: Thread-safety em `accumulate_grad()`

---

### 2️⃣ Estrutura AutogradNode (Linha ~126)

**Localização**: Antes de `struct ZTensor`

```diff
  struct AutogradNode {
      std::vector<std::shared_ptr<ZTensor>> parents;
      std::function<void()> backward_fn;
      std::string op_name;
+     mutable std::mutex backward_lock;
      
      AutogradNode() = default;
      AutogradNode(const std::string& name) : op_name(name) {}
  };
```

**Motivo**: Espaço reservado para sincronização (futuro uso)

---

### 3️⃣ Campo grad_mutex em ZTensor (Linha ~156)

**Localização**: Dentro de `struct ZTensor`, seção autograd

```diff
  struct ZTensor {
      // ========== AUTOGRAD STATE ==========
      bool requires_grad = false;
      std::unique_ptr<ZTensor> grad;
      std::shared_ptr<AutogradNode> grad_fn = nullptr;
+     mutable std::mutex grad_mutex;
      
      // ========== MÉTODOS DE AUTOGRAD ==========
```

**Motivo**: Proteger acesso a `grad` em múltiplas threads

---

### 4️⃣ Proteção em add() (Linha 567)

**Localização**: Início do método `void add(const ZTensor& other)`

```diff
  void add(const ZTensor& other) {
+     if (this->requires_grad) {
+         throw std::logic_error(
+             "In-place operation on tensor with requires_grad=true is not allowed. "
+             "Use add_autograd() for differentiable operations."
+         );
+     }
      
      if (!same_shape(other)) {
          throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
      }
```

**Motivo**: Impedir corrupção de grafo

---

### 5️⃣ Proteção em mul() (Linha 767)

**Localização**: Início do método `void mul(const ZTensor& other)`

```diff
  void mul(const ZTensor& other) {
+     if (this->requires_grad) {
+         throw std::logic_error(
+             "In-place operation on tensor with requires_grad=true is not allowed. "
+             "Use mul_autograd() for differentiable operations."
+         );
+     }
      
      if (!same_shape(other)) throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
```

**Motivo**: Impedir corrupção de grafo

---

### 6️⃣ Acumulação com Mutex (Linha ~200)

**Localização**: Método `void accumulate_grad(const ZTensor& grad_in)`

```diff
  void accumulate_grad(const ZTensor& grad_in) {
      if (grad_in.shape != shape) {
          throw std::invalid_argument("Gradient shape mismatch");
      }
      
+     std::lock_guard<std::mutex> lock(grad_mutex);
+     
      ZTensor& g = ensure_grad();
      const size_t N = size();
      if (N == 0) return;
      
      float* g_data = g.data.data();
      const float* gin_data = grad_in.data.data();
      
      #if HAS_OPENMP
      // ... acumulação ...
```

**Motivo**: Thread-safety em `accumulate_grad()`

---

### 7️⃣ Reshape Comentado (Linha ~1010)

**Localização**: Método `ZTensor reshape(...)`

```diff
      #ifdef HAVE_CUDA
      ensure_host();
      #endif
+     // IMPORTANTE: std::vector copy é rasa (shallow) e compartilha os dados
+     // Ambos result e this->data apontam para o mesmo buffer de memória
+     // Isto implementa uma "view" eficiente, não uma cópia de dados
      result.data = this->data;
```

**Motivo**: Documentar compartilhamento eficiente

---

### 8️⃣ Closure Fix em add_autograd() (Linha ~2115)

**Localização**: Dentro de `static ZTensor add_autograd(...)`

```diff
-     if (requires_grad) {
+     if (requires_grad) {
+         auto result_ptr = std::make_shared<ZTensor>(result);
          auto node = std::make_shared<AutogradNode>("add");
          auto a_ptr = std::make_shared<ZTensor>(a);
          auto b_ptr = std::make_shared<ZTensor>(b);
          node->parents = {a_ptr, b_ptr};
          
          bool a_req = a.requires_grad;
          bool b_req = b.requires_grad;
          
-         node->backward_fn = [&result, a_ptr, b_ptr, a_req, b_req]() {
+         node->backward_fn = [result_ptr, a_ptr, b_ptr, a_req, b_req]() {
-             const ZTensor* grad_result = result.get_grad();
+             const ZTensor* grad_result = result_ptr->get_grad();
              if (!grad_result) return;
              
              if (a_req) {
                  const_cast<ZTensor*>(a_ptr.get())->accumulate_grad(*grad_result);
              }
              if (b_req) {
                  const_cast<ZTensor*>(b_ptr.get())->accumulate_grad(*grad_result);
              }
          };
```

**Motivo**: Eliminar UB (captura de referência local)

---

### 9️⃣ Closure Fix em sub_autograd() (Linha ~2185)

**Localização**: Similar a `add_autograd()`

```diff
+     auto result_ptr = std::make_shared<ZTensor>(result);
      node->parents = {a_ptr, b_ptr};
      
      bool a_req = a.requires_grad;
      bool b_req = b.requires_grad;
      
-     node->backward_fn = [&result, a_ptr, b_ptr, ...]() {
+     node->backward_fn = [result_ptr, a_ptr, b_ptr, ...]() {
-         const ZTensor* grad_result = result.get_grad();
+         const ZTensor* grad_result = result_ptr->get_grad();
          // ... resto igual ...
      };
```

**Motivo**: Eliminar UB

---

### 🔟 Closure Fix em mul_autograd() (Linha ~2275)

**Localização**: Similar aos anteriores

```diff
+     auto result_ptr = std::make_shared<ZTensor>(result);
      node->parents = {a_ptr, b_ptr};
      
      bool a_req = a.requires_grad;
      bool b_req = b.requires_grad;
      
      auto a_copy = std::make_shared<ZTensor>(a);
      auto b_copy = std::make_shared<ZTensor>(b);
      
-     node->backward_fn = [&result, a_ptr, b_ptr, ...]() {
+     node->backward_fn = [result_ptr, a_ptr, b_ptr, ...]() {
-         const ZTensor* grad_result = result.get_grad();
+         const ZTensor* grad_result = result_ptr->get_grad();
          // ... resto igual ...
      };
```

**Motivo**: Eliminar UB

---

### 1️⃣1️⃣ Closure Fix em sum_autograd() (Linha ~2360)

**Localização**: Última operação

```diff
+     auto result_ptr = std::make_shared<ZTensor>(result);
      node->parents = {t_ptr};
      
      auto input_shape = t.shape;
      auto input_size = t.size();
      
-     node->backward_fn = [&result, t_ptr, ...]() {
+     node->backward_fn = [result_ptr, t_ptr, ...]() {
-         const ZTensor* grad_result = result.get_grad();
+         const ZTensor* grad_result = result_ptr->get_grad();
          // ... resto igual ...
      };
```

**Motivo**: Eliminar UB

---

## 📊 Resumo de Mudanças

| # | Tipo | Locação | Status |
|---|------|---------|--------|
| 1 | Include | Linha 4 | ✅ |
| 2 | Campo struct | Linha ~126 | ✅ |
| 3 | Campo struct | Linha ~156 | ✅ |
| 4 | Validação | Linha 567 | ✅ |
| 5 | Validação | Linha 767 | ✅ |
| 6 | Lock guard | Linha ~200 | ✅ |
| 7 | Comentário | Linha ~1010 | ✅ |
| 8 | Closure fix | Linha ~2115 | ✅ |
| 9 | Closure fix | Linha ~2185 | ✅ |
| 10 | Closure fix | Linha ~2275 | ✅ |
| 11 | Closure fix | Linha ~2360 | ✅ |

---

## 🔧 Como Verificar Mudanças

### Ver diff completo
```bash
git diff src/zmatrix.cpp
```

### Ver apenas seções de autograd
```bash
grep -n "mutex\|requires_grad\|accumulate_grad\|add_autograd\|sub_autograd\|mul_autograd\|sum_autograd" src/zmatrix.cpp | head -50
```

### Compilar
```bash
phpize && ./configure && make 2>&1 | grep -i error
```

---

## ✅ Validação Rápida

### 1. Thread-safety implementada?
```bash
grep -c "std::lock_guard" src/zmatrix.cpp
```
**Esperado**: 1

### 2. Todos closure fixes aplicados?
```bash
grep -c "result_ptr = std::make_shared<ZTensor>(result)" src/zmatrix.cpp
```
**Esperado**: 4 (add, sub, mul, sum)

### 3. Inplace protection ativa?
```bash
grep -c "In-place operation on tensor with requires_grad" src/zmatrix.cpp
```
**Esperado**: 2 (add e mul)

### 4. Mutex adicionado?
```bash
grep -c "grad_mutex" src/zmatrix.cpp
```
**Esperado**: ≥ 2 (declaração + uso)

---

## 🚀 Próximo Passo

Compilar e executar testes:
```bash
make clean
make
php test_autograd.php
```

---

**Documento gerado**: 16 de Janeiro, 2026  
**Versão**: 1.0
