# 📋 **ROADMAP: DIA 4-5 (PRÓXIMAS ETAPAS)**

## 🎯 **Objetivos DIA 4-5**

Depois da implementação bem-sucedida de OpenMP + SIMD AVX2 (DIA 1-3), as próximas etapas focam em:

1. **Extensão SIMD** para mais operações
2. **Otimizações finais** baseadas em profiling
3. **Validação completa** com benchmarks
4. **Documentação** e preparação para produção

---

## 🔧 **DIA 4: Extended SIMD Optimization**

### **4.1 - Implementar SIMD para mais operações**

```cpp
// Candidatos para SIMD:
// - abs() : _mm256_set1_ps(0.0) para max(x, -x)
// - sqrt() : _mm256_sqrt_ps() 
// - min/max (element-wise) : _mm256_min_ps(), _mm256_max_ps()
// - clamp(x, min, max)
```

**Ganhos Esperados**:
- `abs`: ~3x (operação simples)
- `sqrt`: ~2x (instrução SIMD nativa)
- `min/max`: ~8x (semelhante a add)

### **4.2 - Otimizar redução (sum, mean, min, max)**

Atualmente, reduções usam loop simples:
```cpp
float sum = 0.0f;
for (size_t i = 0; i < N; ++i) {
    sum += data[i];
}
```

**Melhoria**: Usar SIMD horizontal add:
```cpp
// Redução SIMD: 4 somas paralelas → 1
__m256 acc = _mm256_setzero_ps();
for (...) {
    acc = _mm256_add_ps(acc, _mm256_loadu_ps(...));
}
// Horizontal add dos 8 floats → resultado final
```

**Ganho Esperado**: ~4-6x para grandes arrays

---

## 🧪 **DIA 5: Final Validation & Documentation**

### **5.1 - Profiling com Linux `perf`**

```bash
# Gravar profile durante benchmark
perf record -F 99 php benchmark.php

# Analisar resultados
perf report

# Verificar instruções SIMD usadas
perf stat -e cycles,instructions,cache-misses php benchmark.php
```

### **5.2 - Memory Access Optimization**

Verificar:
- **Cache misses**: `perf stat -e LLC-loads,LLC-load-misses`
- **Memory bandwidth**: Garantir streaming eficiente
- **NUMA awareness**: Se em multi-socket

### **5.3 - Compile-time Optimization Check**

```bash
# Verificar flags:
make clean && make CXXFLAGS="-O3 -march=native -fopenmp -g"

# Confirmar AVX2:
nm -D /usr/lib/php/20240924/zmatrix.so | grep "addps"
```

### **5.4 - Comprehensive Test Suite**

```php
// Tests to implement:
- test_simd_vs_scalar() : Compare accuracy
- test_different_sizes() : 100 to 10M elements
- test_edge_cases() : NaN, Inf, denormalized
- test_memory_layout() : Aligned vs unaligned
- test_parallel_vs_serial() : Threshold validation
```

---

## 📊 **Expected Results Summary**

| Phase | Operations | Speedup | Status |
|-------|-----------|---------|--------|
| **DIA 1** | OpenMP | 1.5x | ✅ Done |
| **DIA 2** | SIMD Elementwise | 7.98x | ✅ Done |
| **DIA 3** | SIMD Activation | 3.61x (ReLU) | ✅ Done |
| **DIA 4** | Extended SIMD | 3-8x (per op) | 🔄 Pending |
| **DIA 5** | Final Validation | - | 🔄 Pending |
| **Total** | Combined | **~50-100x** | 🎯 Target |

---

## 🚀 **Implementation Checklist DIA 4**

- [ ] Implement `abs_simd_kernel()` with `_mm256_set1_ps()`
- [ ] Implement `sqrt_simd_kernel()` with `_mm256_sqrt_ps()`
- [ ] Implement `min_simd_kernel()` and `max_simd_kernel()`
- [ ] Refactor reduction loops with SIMD horizontal ops
- [ ] Benchmark each new optimization
- [ ] Update `DIA_1_3_RESUMO.md` with DIA 4 results

---

## 🧩 **Implementation Checklist DIA 5**

- [ ] Run `perf record` benchmark analysis
- [ ] Check cache efficiency (LLC-loads, misses)
- [ ] Verify SIMD instruction usage (AVX2 specific)
- [ ] Test accuracy vs scalar implementation
- [ ] Create comprehensive test suite
- [ ] Profile multi-threaded performance
- [ ] Document final performance gains
- [ ] Prepare production deployment guide

---

## 📝 **Code Template for DIA 4 (SIMD Kernels)**

```cpp
// Pattern para novas operações SIMD
static inline void op_simd_kernel(float* __restrict__ a, 
                                   const float* __restrict__ b, 
                                   size_t n) {
    #if HAS_AVX2
    const size_t vec_size = 8;
    const size_t aligned_n = (n / vec_size) * vec_size;

    // Vectorized loop
    for (size_t i = 0; i < aligned_n; i += vec_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_op_ps(va, vb);  // Replace: add_ps, mul_ps, etc
        _mm256_storeu_ps(&a[i], result);
    }

    // Tail loop
    for (size_t i = aligned_n; i < n; ++i) {
        a[i] op= b[i];
    }
    #else
    for (size_t i = 0; i < n; ++i) {
        a[i] op= b[i];
    }
    #endif
}
```

---

## 🔗 **References for DIA 4-5**

- **SIMD Intrinsics**: https://www.intel.com/content/dam/develop/external/us/en/documents/manual/64-ia-32-architectures-software-developers-intels-64-and-ia-32-architectures-optimization-reference-manual-325462.pdf
- **OpenMP Optimization**: https://www.openmp.org/
- **Linux Perf**: `man perf-record`, `man perf-stat`
- **AVX2 Cheatsheet**: https://software.intel.com/intrinsics/

---

## ⚠️ **Important Notes**

1. **Transcendentals**: `sin`, `cos`, `exp`, `log` não têm instruções SIMD nativas - usar aproximações polinomiais para melhor ganho
2. **Memory Alignment**: Crucial para `_mm256_loadu_ps` (32-byte alignment)
3. **Denormalized Numbers**: Considerar `_MM_SET_DENORMALS_ZERO_MODE` para performance
4. **Cross-platform**: Testar em diferentes CPUs (Ryzen, Intel, ARM NEON)

---

**Status Atual**: DIA 1-3 ✅ Complete | DIA 4-5 🔄 Ready to Start

**Next Action**: Após revisão desta documentação, implementar DIA 4 com foco em:
1. Operações simples (abs, sqrt, min, max)
2. Reduções eficientes (sum, mean)
3. Benchmarking contínuo
