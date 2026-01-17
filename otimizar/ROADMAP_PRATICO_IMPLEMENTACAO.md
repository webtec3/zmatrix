# 🗺️ ROADMAP PRÁTICO - Por Onde Começar com Tudo Que Já Existe

## 🎯 Situação Atual

```
JÁ IMPLEMENTADO:
├─ OpenMP         ✅ (28 métodos paralelizados)
├─ BLAS           ✅ (cblas_sgemm para matmul)
├─ CUDA           ✅ (13+ operações GPU)
├─ AVX2/AVX-512   ✅ (SIMD dispatch em simd/simd_dispatch.h)
└─ Score: 8.5/10

ADICIONAR AGORA:
├─ Tree Reduction (2 dias)
├─ Kernel Fusion (2 dias)
├─ Auto-Dispatch (1 dia)
└─ Novo Score: 9.5/10
```

---

## 📋 Arquitetura Atual (Entender Antes de Começar)

### Estrutura de Arquivos Críticos

```
src/
├── zmatrix.cpp                    ← MODIFICAR AQUI (métodos)
├── zmatrix_methods.h              ← Métodos PHP binding
├── simd/
│   └── simd_dispatch.h            ← MODIFICAR AQUI (SIMD)
├── gpu_wrapper.h                  ← Interface CUDA
├── gpu_wrapper.cu                 ← Kernels CUDA
└── config.h / config.m4           ← Build config
```

### Fluxo de Dados Atual (Exemplo: add())

```
PHP: $a->add($b)
  ↓
zmatrix_methods.h: PHP_METHOD(ZTensor, add)
  ↓
zmatrix.cpp: ZTensor::add() {
  if (GPU) gpu_add()
  else if (OpenMP && N > 40K) #pragma omp parallel for simd
  else simd::add_f32()
  else loop sequencial
}
  ↓
simd/simd_dispatch.h: add_f32() {
  #ifdef __AVX2__ { _mm256 operations }
  else { scalar loop }
}
```

---

## 🎯 PHASE 0: Preparação (Before You Start)

### Pré-requisitos

- [ ] Build compile sem warnings
  ```bash
  make clean && ./configure --enable-cuda --enable-openmp && make
  ```

- [ ] Tests passam
  ```bash
  php run-tests.php
  ```

- [ ] Git pronto para feature branch
  ```bash
  git checkout -b feature/advanced-optimizations
  ```

- [ ] Entender OpenMP thresholds
  ```cpp
  // src/zmatrix.cpp linha ~75
  #define ZMATRIX_PARALLEL_THRESHOLD 40000
  #define ZMATRIX_GPU_THRESHOLD 200000
  ```

- [ ] Entender padrão SIMD
  ```cpp
  // src/simd/simd_dispatch.h
  namespace zmatrix_simd {
      void add_f32(float* a, const float* b, size_t n);
  }
  ```

---

## 🚀 PHASE 1: TREE REDUCTION (Days 1-2)

### Por Que Começar Aqui?

```
✅ Máximo ROI (2.5-3.5x)
✅ Modificações localizadas (só simd/ + zmatrix.cpp)
✅ Não quebra nada (fallback direto)
✅ Depende de 0 coisa
✅ Testes fáceis (comparar com resultado esperado)
```

### Step 1.1: Criar Funções Base em simd_dispatch.h

**Arquivo:** `src/simd/simd_dispatch.h`

**Adicionar após as funções existentes:**

```cpp
// Adicionar no namespace zmatrix_simd

// Tree reduction para sum com SIMD horizontal add
inline double sum_f32_tree(const float* a, size_t n) {
    if (n == 0) return 0.0;
    
    const size_t BLOCK_SIZE = 256;
    std::vector<double> block_sums;
    
    #if HAS_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t b = 0; b < n; b += BLOCK_SIZE) {
        size_t end = std::min(b + BLOCK_SIZE, n);
        double local_sum = 0.0;
        
        #if HAS_AVX2
        const __m256 zero = _mm256_setzero_ps();
        __m256 sum_vec = zero;
        
        size_t simd_end = b + ((end - b) / 8) * 8;
        for (size_t i = b; i < simd_end; i += 8) {
            __m256 v = _mm256_loadu_ps(a + i);
            sum_vec = _mm256_add_ps(sum_vec, v);
        }
        
        // Horizontal sum de __m256
        float tmp[8];
        _mm256_storeu_ps(tmp, sum_vec);
        local_sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                    tmp[4] + tmp[5] + tmp[6] + tmp[7];
        
        // Tail
        for (size_t i = simd_end; i < end; ++i) {
            local_sum += a[i];
        }
        #else
        for (size_t i = b; i < end; ++i) {
            local_sum += a[i];
        }
        #endif
        
        // Armazenar resultado do bloco
        if (b / BLOCK_SIZE < block_sums.size()) {
            block_sums[b / BLOCK_SIZE] = local_sum;
        } else {
            block_sums.push_back(local_sum);
        }
    }
    
    // Redução final
    double total = 0.0;
    for (double val : block_sums) {
        total += val;
    }
    return total;
}

// Tree reduction para max
inline float max_f32_tree(const float* a, size_t n) {
    if (n == 0) return std::numeric_limits<float>::quiet_NaN();
    
    const size_t BLOCK_SIZE = 256;
    std::vector<float> block_maxs;
    block_maxs.push_back(std::numeric_limits<float>::lowest());
    
    #if HAS_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t b = 0; b < n; b += BLOCK_SIZE) {
        size_t end = std::min(b + BLOCK_SIZE, n);
        float local_max = std::numeric_limits<float>::lowest();
        
        for (size_t i = b; i < end; ++i) {
            local_max = std::max(local_max, a[i]);
        }
        
        if (b / BLOCK_SIZE < block_maxs.size()) {
            block_maxs[b / BLOCK_SIZE] = local_max;
        } else {
            block_maxs.push_back(local_max);
        }
    }
    
    float result = std::numeric_limits<float>::lowest();
    for (float val : block_maxs) {
        result = std::max(result, val);
    }
    return result;
}
```

**Checklist:**
- [ ] Código compila sem erros
- [ ] Sem warnings
- [ ] OpenMP não cria race conditions

---

### Step 1.2: Integrar em zmatrix.cpp - Método sum()

**Arquivo:** `src/zmatrix.cpp` (procure por `double sum() const`)

**Substituir método sum() atual:**

```cpp
// ANTES:
double sum() const {
    const size_t N = size();
    if (N == 0) return 0.0;
#ifdef HAVE_CUDA
    ensure_host();
#endif
    const float* a = data.data();
    
    double total_sum = 0.0;
    #if HAS_OPENMP
    if (N > ZMATRIX_PARALLEL_THRESHOLD) {
        #pragma omp parallel for reduction(+:total_sum) schedule(static)
        for (size_t i = 0; i < N; ++i) {
            total_sum += a[i];
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            total_sum += a[i];
        }
    }
    #else
    for (size_t i = 0; i < N; ++i) {
        total_sum += a[i];
    }
    #endif
    return total_sum;
}

// DEPOIS (com tree reduction):
double sum() const {
    const size_t N = size();
    if (N == 0) return 0.0;
#ifdef HAVE_CUDA
    ensure_host();
#endif
    const float* a = data.data();
    
    #if HAS_OPENMP
    if (N > ZMATRIX_PARALLEL_THRESHOLD) {
        return zmatrix_simd::sum_f32_tree(a, N);  // ← NOVO
    }
    #endif
    
    // Fallback para N pequeno
    return zmatrix_simd::sum_f32(a, N);
}
```

**Checklist:**
- [ ] Método compila
- [ ] sum_f32 antigo ainda funciona (fallback)
- [ ] Testes passam

---

### Step 1.3: Integrar em zmatrix.cpp - Método max()

**Arquivo:** `src/zmatrix.cpp` (procure por `float max() const`)

**Similar ao sum():**

```cpp
float max() const {
    const size_t N = size();
    if (N == 0) return std::numeric_limits<float>::quiet_NaN();
#ifdef HAVE_CUDA
    ensure_host();
#endif
    const float* p = data.data();

    #if HAS_OPENMP
    if (N > ZMATRIX_PARALLEL_THRESHOLD) {
        return zmatrix_simd::max_f32_tree(p, N);  // ← NOVO
    }
    #endif
    
    return zmatrix_simd::max_f32(p, N);
}
```

---

### Step 1.4: Testes de Tree Reduction

**Criar arquivo:** `tests/test_tree_reduction.php`

```php
<?php
echo "Testing Tree Reduction...\n";

// Test 1: sum()
$a = new ZTensor([100000000]);  // 100M
$a->fill(2.0);
$result = $a->sum();
$expected = 100000000 * 2.0;
assert(abs($result - $expected) < 1.0, "sum() failed: got $result, expected $expected");
echo "✓ sum() passed\n";

// Test 2: max()
$a = new ZTensor([10000000]);
for ($i = 0; $i < 10000000; $i++) {
    $a[$i] = sin($i);
}
$result = $a->max();
assert($result <= 1.0 && $result >= 0.99, "max() failed: got $result");
echo "✓ max() passed\n";

echo "All tree reduction tests passed!\n";
```

**Executar:**
```bash
php tests/test_tree_reduction.php
```

**Checklist:**
- [ ] Tests passam
- [ ] Sem segfaults
- [ ] Resultados numericamente corretos

---

### Step 1.5: Benchmark Tree Reduction

**Criar arquivo:** `benchmarks/bench_tree_reduction.php`

```php
<?php
$sizes = [100000, 1000000, 10000000, 100000000];

foreach ($sizes as $size) {
    $a = new ZTensor([$size]);
    $a->fill(2.0);
    
    $start = microtime(true);
    for ($i = 0; $i < 10; $i++) {
        $sum = $a->sum();
    }
    $elapsed = (microtime(true) - $start) * 1000;  // ms
    $per_iter = $elapsed / 10;
    
    $gb_per_sec = ($size * 4 * 10) / ($elapsed / 1000) / 1e9;
    
    echo "sum($size): {$per_iter:.2f}ms, {$gb_per_sec:.1f} GB/s\n";
}
```

---

## 🔄 PHASE 2: KERNEL FUSION (Days 3-4)

### Por Que Depois de Tree Reduction?

```
✅ Beneficia de tree reduction (menos memory I/O)
✅ Padrão bem definido (add+scalar, mul+add+relu, etc)
✅ Fácil testar (resultado bitwise esperado)
✅ Alto ganho (2-5x)
```

### Step 2.1: Criar Funções Fused em simd_dispatch.h

**Arquivo:** `src/simd/simd_dispatch.h`

```cpp
// Fused multiply + add (escalar)
inline void fused_mul_add_f32(float* a, float scale, float offset, size_t n) {
    #if HAS_AVX2
    __m256 scale_vec = _mm256_set1_ps(scale);
    __m256 offset_vec = _mm256_set1_ps(offset);
    
    for (size_t i = 0; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(a + i);
        // FMA: multiply-add em 1 instrução!
        __m256 result = _mm256_fmadd_ps(x, scale_vec, offset_vec);
        _mm256_storeu_ps(a + i, result);
    }
    // Tail
    for (size_t i = (n / 8) * 8; i < n; ++i) {
        a[i] = a[i] * scale + offset;
    }
    #else
    for (size_t i = 0; i < n; ++i) {
        a[i] = a[i] * scale + offset;
    }
    #endif
}

// Fused multiply + add + relu (binário)
inline void fused_mul_add_relu_f32(float* a, const float* b, float bias, size_t n) {
    #if HAS_AVX2
    __m256 bias_vec = _mm256_set1_ps(bias);
    __m256 zero = _mm256_setzero_ps();
    
    for (size_t i = 0; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        
        __m256 result = _mm256_fmadd_ps(a_vec, b_vec, bias_vec);
        result = _mm256_max_ps(result, zero);  // ReLU
        
        _mm256_storeu_ps(a + i, result);
    }
    // Tail
    for (size_t i = (n / 8) * 8; i < n; ++i) {
        float temp = a[i] * b[i] + bias;
        a[i] = std::max(0.0f, temp);
    }
    #else
    for (size_t i = 0; i < n; ++i) {
        float temp = a[i] * b[i] + bias;
        a[i] = std::max(0.0f, temp);
    }
    #endif
}
```

---

### Step 2.2: Criar Métodos em zmatrix.cpp

**Arquivo:** `src/zmatrix.cpp`

```cpp
// Novo método: fused_multiply_add_scalar
void fused_multiply_add(float scale, float offset) {
    const size_t N = size();
    if (N == 0) return;
    
#ifdef HAVE_CUDA
    if (device_valid) {
        ensure_device();
        // gpu_fused_mul_add_device(d_data, scale, offset, N);
        // mark_device_modified();
        // return;  ← Skip GPU for now
    }
    ensure_host();
#endif
    
    float * __restrict__ a = data.data();
    
    #if HAS_OPENMP
    if (N > ZMATRIX_PARALLEL_THRESHOLD) {
        #pragma omp parallel for simd schedule(static)
        for (size_t i = 0; i < N; ++i) {
            a[i] = a[i] * scale + offset;
        }
    } else {
        zmatrix_simd::fused_mul_add_f32(a, scale, offset, N);
    }
    #else
    zmatrix_simd::fused_mul_add_f32(a, scale, offset, N);
    #endif
    
#ifdef HAVE_CUDA
    mark_host_modified();
#endif
}

// Novo método: fused_mul_add_relu
void fused_mul_add_relu(const ZTensor& other, float bias) {
    if (!same_shape(other)) {
        throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);
    }
    
    const size_t N = size();
    if (N == 0) return;
    
    float * __restrict__ a = data.data();
    const float * __restrict__ b = other.data.data();
    
    #if HAS_OPENMP
    if (N > ZMATRIX_PARALLEL_THRESHOLD) {
        #pragma omp parallel for simd schedule(static)
        for (size_t i = 0; i < N; ++i) {
            float temp = a[i] * b[i] + bias;
            a[i] = std::max(0.0f, temp);
        }
    } else {
        zmatrix_simd::fused_mul_add_relu_f32(a, b, bias, N);
    }
    #else
    zmatrix_simd::fused_mul_add_relu_f32(a, b, bias, N);
    #endif
}
```

---

### Step 2.3: Testar Fusion

**Criar arquivo:** `tests/test_fusion.php`

```php
<?php
echo "Testing Kernel Fusion...\n";

// Test: fused_mul_add
$a = new ZTensor([10000000]);
$a->fill(2.0);

// Reference: a = a * 3 + 5
$a_ref = clone $a;
$a_ref->multiply_scalar(3);
$a_ref->scalar_add(5);

// Fused version
$a->fused_multiply_add(3, 5);

// Compare
$diff = 0;
for ($i = 0; $i < 10000000; $i += 1000) {
    $diff = max($diff, abs($a[$i] - $a_ref[$i]));
}

assert($diff < 1e-5, "Fusion mismatch: max diff = $diff");
echo "✓ fused_mul_add passed\n";

echo "All fusion tests passed!\n";
```

---

## 🎯 PHASE 3: AUTO-DISPATCH (Day 5)

### Por Que Por Último?

```
✅ Completa as outras técnicas (não dependência)
✅ Calibration automática no startup
✅ Refina decisões GPU vs CPU
```

### Step 3.1: Criar DispatchMetrics em zmatrix.cpp

**Arquivo:** `src/zmatrix.cpp` (adicionar antes do struct ZTensor)

```cpp
struct DispatchMetrics {
    double simd_throughput_gb_per_sec = 50.0;  // Default
    double openmp_overhead_us = 5.0;            // Default
    int num_cores = 1;
    
    size_t adaptive_parallel_threshold = 40000;  // Default
    size_t adaptive_gpu_threshold = 200000;      // Default
    
    static DispatchMetrics& instance() {
        static DispatchMetrics m;
        return m;
    }
    
    void calibrate() {
        // Detect cores
        #ifdef _OPENMP
        num_cores = omp_get_max_threads();
        #endif
        
        // Adjust thresholds based on cores
        if (num_cores >= 8) {
            adaptive_parallel_threshold = 25000;
        } else if (num_cores >= 4) {
            adaptive_parallel_threshold = 40000;
        } else {
            adaptive_parallel_threshold = 100000;
        }
        
        php_printf("[zmatrix] Detected %d cores\n", num_cores);
        php_printf("[zmatrix] Parallel threshold: %zu\n", adaptive_parallel_threshold);
    }
};
```

---

### Step 3.2: Hook em MINIT

**Arquivo:** `src/zmatrix.cpp` (procure por `PHP_MINIT_FUNCTION(zmatrix)`)

```cpp
PHP_MINIT_FUNCTION(zmatrix) {
    // ... código existente ...
    
    // Calibrate dispatch metrics
    DispatchMetrics::instance().calibrate();
    
    // ... resto do código ...
    return SUCCESS;
}
```

---

### Step 3.3: Usar Thresholds Adaptativos

**Arquivo:** `src/zmatrix.cpp` (substituir ZMATRIX_PARALLEL_THRESHOLD)

```cpp
// ANTES:
if (N > ZMATRIX_PARALLEL_THRESHOLD) {

// DEPOIS:
if (N > DispatchMetrics::instance().adaptive_parallel_threshold) {
```

**Aplicar em todos os métodos críticos:**
- add()
- mul()
- sum()
- relu()
- etc.

---

## 📊 ORDEM PRÁTICA DIA-A-DIA

### DAY 1 (Tree Reduction - Part 1)

```
Morning:
├─ 9:00: Ler este roadmap
├─ 9:30: Code review de sum() + max() atuais
└─ 10:00: Implementar sum_f32_tree() + max_f32_tree()

Afternoon:
├─ 14:00: Integrar em zmatrix.cpp
├─ 15:00: Testes básicos
└─ 16:00: Commit: "Add tree reduction for sum/max"

Evening:
├─ 17:00: Criar benchmark_tree_reduction.php
└─ 18:00: Medir ganho (2.5x target)
```

### DAY 2 (Tree Reduction - Part 2)

```
Morning:
├─ 9:00: mean() com tree reduction
├─ 10:00: std() com tree reduction
└─ 11:00: Testes

Afternoon:
├─ 14:00: Benchmarks completos
├─ 15:00: Verificar scaling (14x em 16 cores)
└─ 16:00: Commit: "Complete tree reduction implementation"

Evening:
├─ 17:00: Code review próprio
└─ 18:00: Preparar para fusão
```

### DAY 3 (Kernel Fusion - Part 1)

```
Morning:
├─ 9:00: Implementar fused_mul_add_f32() em SIMD
├─ 10:00: Integrar em novo método fused_multiply_add()
└─ 11:00: Testes básicos

Afternoon:
├─ 14:00: Implementar fused_mul_add_relu_f32()
├─ 15:00: Integrar em novo método fused_mul_add_relu()
└─ 16:00: Testes de correctness

Evening:
├─ 17:00: Benchmarks fusion
└─ 18:00: Verificar 2-3x ganho
```

### DAY 4 (Kernel Fusion - Part 2 + GPU optional)

```
Morning:
├─ 9:00: SIMD fused_add_relu() (matrix ops)
├─ 10:00: Testar com diferentes tamanhos
└─ 11:00: Commit fusion

Afternoon (opcional - GPU):
├─ 14:00: Gpu kernels equivalentes
├─ 15:00: Testes GPU vs CPU
└─ 16:00: Benchmarks

Evening:
├─ 17:00: Documento performance gains
└─ 18:00: Preparar para auto-dispatch
```

### DAY 5 (Auto-Dispatch)

```
Morning:
├─ 9:00: Implementar DispatchMetrics
├─ 10:00: Hook em MINIT
└─ 11:00: Testar calibration

Afternoon:
├─ 14:00: Substituir thresholds hardcoded
├─ 15:00: Testar em CPU multicore
└─ 16:00: Testar GPU dispatch

Evening:
├─ 17:00: Final benchmarks
└─ 18:00: Commit: "Add adaptive dispatch"
```

### DAY 6 (Testing + Docs)

```
Morning:
├─ 9:00: Run full test suite
├─ 10:00: Regression tests
└─ 11:00: Benchmark comparação antes/depois

Afternoon:
├─ 14:00: Documentation
├─ 15:00: Update README
└─ 16:00: Create blog post

Evening:
├─ 17:00: Code review final
└─ 18:00: Preparar PR
```

---

## 🛠️ ARQUIVOS QUE VOCÊ VAI MODIFICAR

### Arquivos Críticos

```
1. src/simd/simd_dispatch.h
   ├─ Adicionar: sum_f32_tree()
   ├─ Adicionar: max_f32_tree()
   ├─ Adicionar: fused_mul_add_f32()
   └─ Adicionar: fused_mul_add_relu_f32()

2. src/zmatrix.cpp
   ├─ Modificar: sum() const
   ├─ Modificar: max() const
   ├─ Adicionar: fused_multiply_add()
   ├─ Adicionar: fused_mul_add_relu()
   ├─ Adicionar: DispatchMetrics struct
   ├─ Modificar: PHP_MINIT_FUNCTION
   └─ Substituir: ZMATRIX_PARALLEL_THRESHOLD → adaptive

3. tests/
   ├─ Adicionar: test_tree_reduction.php
   └─ Adicionar: test_fusion.php

4. benchmarks/
   ├─ Adicionar: bench_tree_reduction.php
   └─ Adicionar: bench_fusion.php
```

### Arquivos NÃO Modificar (Deixar Intactos)

```
❌ src/gpu_wrapper.cu      (CUDA já funciona)
❌ src/zmatrix_methods.h   (PHP binding - não mexer)
❌ config.m4               (Build config - não mexer)
```

---

## ⚠️ CUIDADOS IMPORTANTES

### 1. Não Quebrar OpenMP Existente

```cpp
// ✅ CORRETO: Verificar HAS_OPENMP
#if HAS_OPENMP
if (N > threshold) {
    #pragma omp parallel for simd
    ...
}
#endif

// ❌ ERRADO: Remover OpenMP
#pragma omp parallel for simd  // Sem verificação
```

### 2. Manter Fallbacks

```cpp
// ✅ CORRETO: Fallback sempre existe
if (large_N) zmatrix_simd::tree_reduction();
else simd_simple::add_f32();  // Fallback

// ❌ ERRADO: Sem fallback
return zmatrix_simd::tree_reduction();  // Pode crash se falhar
```

### 3. Não Mudar CUDA Existente

```cpp
// ✅ CORRETO: Preservar CUDA checks
#ifdef HAVE_CUDA
if (device_valid) { ... gpu code ... }
ensure_host();
#endif

// ❌ ERRADO: Remover CUDA code
// Comentar CUDA checks - vai quebrar para users com GPU
```

### 4. Precisão Numérica

```cpp
// ✅ CORRETO: Double accumulator
double total = 0.0;  // Não float!
for (...) total += (double)a[i];

// ❌ ERRADO: Float accumulator
float total = 0.0f;  // Underflow em grandes somas
```

---

## ✅ CHECKLIST POR DAY

### After DAY 1
- [ ] sum_f32_tree() compila
- [ ] max_f32_tree() compila
- [ ] sum() uses tree reduction
- [ ] Testes passam (test_tree_reduction.php)
- [ ] Benchmark mostra 2x+ ganho

### After DAY 2
- [ ] mean() refatorizado
- [ ] std() refatorizado
- [ ] Scaling linear com cores (14x em 16 cores)
- [ ] 3.0-3.4x ganho em std()
- [ ] Git commit feito

### After DAY 3
- [ ] fused_mul_add() implementada
- [ ] fused_mul_add_relu() implementada
- [ ] Resultados bitwise-exact vs não-fused
- [ ] Ganho 2-3x medido
- [ ] Git commit feito

### After DAY 4
- [ ] Fusion operações matriciais funciona
- [ ] GPU kernels (opcional) testados
- [ ] Sem regressões (run-tests.php)
- [ ] Documentação draft

### After DAY 5
- [ ] DispatchMetrics calibra corretamente
- [ ] Thresholds adaptativos ativo
- [ ] Teste em CPU multicore
- [ ] Teste com GPU disabled

### After DAY 6
- [ ] 100% de testes passam
- [ ] Benchmarks mostram 3.6-12.5x ganho combinado
- [ ] Documentação completa
- [ ] PR ready

---

## 🚀 Comandos Essenciais

### Build & Test

```bash
# Clean rebuild
make clean && ./configure --enable-cuda --enable-openmp && make

# Run unit tests
php run-tests.php

# Run custom tests
php tests/test_tree_reduction.php
php tests/test_fusion.php

# Benchmark
php benchmarks/bench_tree_reduction.php
php benchmarks/bench_fusion.php

# Git flow
git add src/
git commit -m "feat: add tree reduction and kernel fusion"
git push origin feature/advanced-optimizations
```

---

## 💡 Dicas Práticas

### Problema: "Não sei qual threshold usar"

**Solução:** Use os defaults, depois perfil com seu hardware:
```bash
php -r "
    \$a = new ZTensor([100000]);
    \$a->fill(1.0);
    for (\$i=0; \$i<1000; \$i++) \$s = \$a->sum();
" | time php -r "..."
```

### Problema: "Teste está muito lento"

**Solução:** Use tamanhos menores para dev:
```php
$a = new ZTensor([1000000]);  // 1M em vez de 100M
```

### Problema: "Resultado está errado"

**Solução:** Verificar tolerância:
```php
$diff = abs($result - $expected);
assert($diff < 1e-4, "Diff: $diff");  // Float tolerance
```

### Problema: "OpenMP não está usando cores"

**Solução:** Verificar:
```bash
export OMP_NUM_THREADS=4
export OMP_DYNAMIC=FALSE
php benchmark.php
```

---

## 📊 Ganho Esperado (Referência)

```
Operação        DAY 1    DAY 2    DAY 3-4  DAY 5   Final
────────────────────────────────────────────────────────
sum(100M)       +2.5x    +2.5x    +2.5x    +1.1x   = 3.0x
mean(100M)      +1.0x    +2.5x    +2.5x    +1.1x   = 2.8x
relu(100M)      +1.0x    +1.0x    +3.0x    +1.1x   = 3.3x
add+mul+relu    +1.0x    +1.0x    +3.5x    +1.1x   = 3.9x
────────────────────────────────────────────────────────
COMBINADO       +1.6x    +1.8x    +3.0x    +1.1x   = 5.7x
```

---

## 🎯 Meta Final

```
✅ Score Otimização: 8.5 → 9.5
✅ Performance: 3.6-12.5x mais rápido
✅ Tempo: 5-7 dias (realistic timeline)
✅ Risco: Baixo (técnicas proven)
✅ Teste: 100% passing
✅ Pronto: Production deployment
```

---

*Roadmap Prático - 17 de Janeiro de 2026*  
**Status: PRONTO PARA IMPLEMENTAÇÃO IMEDIATA** 🚀
