# ✅ CHECKLIST DE IMPLEMENTAÇÃO - Técnicas Avançadas

## 🎯 Objetivo Final
Implementar **Kernel Fusion + Tree Reduction + Auto-Dispatch** em **5-7 dias**

---

## 📋 PHASE 1: TREE REDUCTION (Days 1-2)

### Day 1: Funções Base em simd/simd_dispatch.h

- [ ] **sum_f32_tree()**
  - [ ] Blocos de 256 elementos (L2 cache friendly)
  - [ ] SIMD horizontal reduction dentro do bloco
  - [ ] Acumulação final sequencial
  - [ ] Teste com 100M elementos
  - [ ] Benchmark vs versão simples
  
- [ ] **max_f32_tree()**
  - [ ] Mesmo padrão que sum
  - [ ] Usar `_mm256_max_ps()` para comparação
  - [ ] Teste unitário
  
- [ ] **min_f32_tree()** (bonus se time)
  - [ ] Similar a max

### Day 2: Integração em zmatrix.cpp

- [ ] **mean_f32_tree()**
  - [ ] Usar sum_f32_tree() + divisão
  - [ ] Integrar em `ZTensor::mean()`
  - [ ] Testes
  
- [ ] **std_f32_tree()**
  - [ ] Computar mean primeiro
  - [ ] Tree reduction da variance
  - [ ] Sqrt final
  - [ ] Integrar em `ZTensor::std()`
  - [ ] Testes com diferentes tamanhos
  
- [ ] **Benchmarking Completo**
  - [ ] Criar benchmark_tree_reduction.cpp
  - [ ] Comparar: simples vs tree vs GPU (se disponível)
  - [ ] Plotar scaling com número de threads
  - [ ] Documentar resultados

---

## 📋 PHASE 2: KERNEL FUSION (Days 3-4)

### Day 3: Operações Escalares Fused

- [ ] **fused_mul_add_f32()** em SIMD
  - [ ] Implementação AVX2 com FMA
  - [ ] Fallback scalar
  - [ ] Teste: `a = a * scale + offset`
  
- [ ] **ZTensor::fused_multiply_add(scale, offset)**
  - [ ] Integrar em zmatrix.cpp
  - [ ] OpenMP + SIMD + CUDA (se houver gpu_fused_*)
  - [ ] Testes
  
- [ ] **fused_mul_add_relu_f32()** em SIMD
  - [ ] AVX2: FMA + max com zero
  - [ ] Teste: `a = max(0, a * b + bias)`
  
- [ ] **Teste de Fusion vs Não-Fusion**
  - [ ] Confirmar ganho 2-3x
  - [ ] Validar resultados numéricos

### Day 4: Operações Matriciais Fused

- [ ] **fused_add_relu()** para matrizes
  - [ ] Row-wise broadcast do bias
  - [ ] Teste com diferentes tamanhos
  - [ ] Validação numérica vs versão separada
  
- [ ] **GPU Kernels** (cuda/gpu_wrapper.cu)
  - [ ] `gpu_fused_mul_add_relu()`
  - [ ] `gpu_fused_add_relu()`
  - [ ] Teste de correctness vs CPU
  
- [ ] **Benchmarking Fusion**
  - [ ] Criar benchmark_fusion.cpp
  - [ ] Comparar: fused vs non-fused
  - [ ] Tamanhos: 1K, 100K, 10M
  - [ ] CPU vs GPU
  
- [ ] **Documentação de Padrões**
  - [ ] Quando usar fusion
  - [ ] Exemplos em zmatrix.php

---

## 📋 PHASE 3: AUTO-DISPATCH (Days 5)

### Day 5: Sistema Automático

- [ ] **DispatchMetrics struct** em zmatrix.cpp
  - [ ] `calibrate_simd()`
  - [ ] `calibrate_openmp()`
  - [ ] `calibrate_gpu()`
  - [ ] `compute_thresholds()`
  
- [ ] **AutoDispatcher class**
  - [ ] Método `decide(size_t N, string operation)`
  - [ ] Retorna Target enum (GPU, OpenMP, SIMD, Sequential)
  - [ ] Métodos `apply_*` para cada operação
  
- [ ] **MINIT Hook**
  - [ ] Adicionar `DispatchMetrics::instance().calibrate()` em PHP_MINIT_FUNCTION
  - [ ] Print resultado no STDOUT/logs
  
- [ ] **Integração em Métodos Críticos**
  - [ ] `void add()`
  - [ ] `void mul()`
  - [ ] `double sum()`
  - [ ] `float max()`
  - [ ] Testar dispatch correto
  
- [ ] **Testes de Dispatch**
  - [ ] Verificar que GPU é usado para N grande
  - [ ] Verificar que OpenMP é usado para N médio
  - [ ] Verificar que SIMD é usado para N pequeno
  - [ ] Disabilitar GPU e verificar fallback

---

## 🧪 TESTING & VALIDATION (Day 6)

### Testes Unitários

- [ ] **Tree Reduction**
  - [ ] sum vs reference (std::accumulate)
  - [ ] mean vs manual calculation
  - [ ] std vs numpy/scipy
  - [ ] max/min correctness
  
- [ ] **Kernel Fusion**
  - [ ] Resultado numérico vs operações separadas (1e-5 tolerance)
  - [ ] Overflow/underflow cases
  - [ ] Edge cases (N=1, N=0, muito pequeno)
  
- [ ] **Auto-Dispatch**
  - [ ] Threshold selection lógico
  - [ ] Fallbacks funcionam
  - [ ] Sem crashes ao desabilitar GPU
  
- [ ] **Regression Tests**
  - [ ] Todos os testes antigos ainda passam
  - [ ] Resultados bit-exact (onde aplicável)

### Benchmarks

- [ ] **CPU Benchmarks**
  - [ ] sum/mean/std (10M elementos)
  - [ ] add/mul (100M elementos)
  - [ ] fused operations vs components
  - [ ] Verificar scaling com threads
  
- [ ] **GPU Benchmarks** (se GPU disponível)
  - [ ] Mesmas operações
  - [ ] Comparar CPU vs GPU
  - [ ] Verify dispatch decisions
  
- [ ] **End-to-End Benchmark**
  - [ ] Forward pass de rede neural 3-layer
  - [ ] Antes: 45ms/epoch
  - [ ] Depois: <10ms/epoch (4-5x)

---

## 📖 DOCUMENTATION (Day 7)

### Código

- [ ] **Comentários em SIMD**
  - [ ] Explicar cada _mm256_ operação
  - [ ] Documentar fallbacks
  
- [ ] **Documentar Novos Métodos**
  - [ ] `fused_mul_add()`
  - [ ] `fused_mul_add_relu()`
  - [ ] `fused_add_relu()`
  
- [ ] **Atualizar README**
  - [ ] Mencionar técnicas avançadas
  - [ ] Performance gains
  - [ ] Hardware requirements

### Markdown Docs

- [ ] **Criar IMPLEMENTACAO_TECNICAS_AVANCADAS.md**
  - [ ] Decisões de design
  - [ ] Trade-offs considerados
  - [ ] Lessons learned
  
- [ ] **Atualizar ANALISE_OTIMIZACOES.md**
  - [ ] Adicionar resultados reais
  - [ ] Benchmarks executados
  - [ ] Comparação antes/depois

---

## 🎯 Checkpoints Diários

### Day 1 Checkpoint
```
✅ sum_f32_tree implementada
✅ max_f32_tree implementada
✅ Benchmark mostra 2.5x ganho
✅ Testes passam
```

### Day 2 Checkpoint
```
✅ mean(), std() refatorizadas para tree
✅ Testes unitários passam
✅ 3.0-3.4x ganho em std/mean confirmado
✅ Scaling com threads verifies (14x em 16 cores)
```

### Day 3 Checkpoint
```
✅ fused_mul_add funcionando (CPU + SIMD)
✅ fused_mul_add_relu implementada
✅ Resultados bit-exact vs versão separada
✅ 2-3x ganho medido
```

### Day 4 Checkpoint
```
✅ GPU kernels compilam e executam
✅ Testes de fusion passam
✅ Auto-dispatch pronto para integração
✅ Documentação draft
```

### Day 5 Checkpoint
```
✅ DispatchMetrics calibra corretamente
✅ AutoDispatcher::decide() retorna Target correto
✅ Métodos kritikos integrados com auto-dispatch
✅ Fallbacks funcionam sem GPU
```

### Day 6 Checkpoint
```
✅ Todos os testes unitários passam
✅ Benchmarks mostram 3.6-12.5x ganho total
✅ Sem regressões vs versão anterior
✅ Scaling confirms (14x em 16 cores vs 8x antes)
```

### Day 7 Checkpoint
```
✅ Documentação completa
✅ Commit pronto com messages descritivas
✅ PR review-ready
✅ Pronto para produção
```

---

## 🔄 Build & Test Commands

### Compilação

```bash
# Clean rebuild
make clean && ./configure --enable-cuda --enable-openmp && make

# Apenas tree reduction
make && php -d enable_dl=Off -r "
  \$a = new ZTensor([10000000]);
  \$a->fill(2.0);
  echo \"Testing sum tree reduction...\n\";
"

# Com debug symbols
./configure CFLAGS="-O2 -g" --enable-cuda && make
gdb php -x gdb_commands.txt
```

### Testes

```bash
# Unit tests
php tests/test_tree_reduction.php
php tests/test_kernel_fusion.php
php tests/test_auto_dispatch.php

# Benchmarks
php benchmarks/bench_tree_reduction.php
php benchmarks/bench_fusion.php
php benchmarks/bench_dispatch.php

# Full regression
php run-tests.php
```

### Verificações

```bash
# Check GPU dispatch decision
ZMATRIX_GPU_DEBUG=1 php test_dispatch.php

# Profiling
perf record php benchmark.php
perf report

# Memory usage
valgrind --tool=massif php benchmark.php
```

---

## 📊 Success Criteria

| Métrica | Target | Acceptance |
|---------|--------|-----------|
| Tree Reduction Speedup | 2.5-3.5x | ≥ 2.0x |
| Fusion Speedup | 2-3x | ≥ 1.5x |
| Auto-Dispatch Overhead | < 1% | < 2% |
| Combined Speedup | 3.6-12x | ≥ 3.0x |
| Test Pass Rate | 100% | 100% |
| Zero Regressions | Yes | Yes |
| GPU Fallback Safe | Yes | Yes |
| Documentation | Complete | > 90% |

---

## 🚀 Launch Plan

### Pre-Launch (Day 0)
- [ ] Feature branch criado
- [ ] Plano compartilhado com team
- [ ] Ambiente de teste preparado

### Development (Days 1-5)
- [ ] Daily standup confirma progresso
- [ ] WIP commits em feature branch
- [ ] Tests rodam a cada checkpoint

### Testing (Day 6)
- [ ] QA testa em múltiplos hardware
- [ ] Benchmarks validados
- [ ] Documentação review

### Release (Day 7)
- [ ] PR criado com descrição detalhada
- [ ] Code review aprovado
- [ ] Merge para main branch
- [ ] Tag version criada
- [ ] Release notes publicadas

---

## 📝 Template de Commit

```
commit: Implement tree reduction for parallel operations

WHAT:
- Add sum_f32_tree(), mean_f32_tree(), std_f32_tree() with block-based reduction
- Integrate into ZTensor::sum(), mean(), std() methods
- SIMD horizontal reduction within each block (AVX2)

WHY:
- Eliminate OpenMP synchronization overhead
- Cache-friendly block processing
- 2.5-3.5x speedup on large tensors (>40K elements)

HOW:
- Each thread processes independent block (256 elements)
- SIMD reduction within block (no sync)
- Final reduction of block results (sequencial)
- Tested: bit-exact vs simple reduction, scaling on 16 cores

PERFORMANCE:
- sum(100M): 45ms → 15ms (3.0x)
- mean(100M): 50ms → 18ms (2.8x)
- std(100M): 85ms → 25ms (3.4x)

TESTS:
- Unit tests: test_tree_reduction.php ✓
- Benchmark: bench_tree_reduction.php ✓
- Regression: run-tests.php ✓
```

---

## 🎓 Lessons Learned Framework

Ao terminar cada fase, documentar:

```markdown
## PHASE 1: Tree Reduction

### What Worked Well
- Bloco-based reduction muito eficiente
- SIMD dentro do bloco = ideal
- Scaling linear com threads (14x em 16 cores)

### What Was Challenging
- Redução horizontal de __m256 requer permutação
- Balancing block size vs overhead

### What Would You Do Differently
- Maybe start with 512-element blocks
- Profile different block sizes

### Key Insights
- Memory bandwidth é o gargalo real
- Tree reduction elimina sync, não speedup computation
```

---

## 📞 Escalation Path

If blocked or uncertain:

```
Tree Reduction Issue?
  → Check: memory bandwidth saturation
  → Check: block size optimization
  → Check: SIMD horizontal reduction correctness

Fusion Issue?
  → Check: numerical accuracy (1e-5 tolerance)
  → Check: compiler optimizations (-O3 -march=native)
  → Check: register pressure

Auto-Dispatch Issue?
  → Check: threshold calibration on your hardware
  → Check: ZMATRIX_GPU_DEBUG environment variable
  → Check: fallback paths work
```

---

## ✨ Final Verification Checklist

Before declaring "done":

- [ ] Código compila sem warnings
- [ ] Testes passam (100%)
- [ ] Benchmarks executam sem crashes
- [ ] Documentação é clara e completa
- [ ] Exemplos em PHP funcionam
- [ ] GPU fallback funciona (export ZMATRIX_FORCE_CPU=1)
- [ ] Sem memory leaks (valgrind clean)
- [ ] Performance matches targets (±10%)
- [ ] Ready for production deployment

---

*Checklist de Implementação - 17 de Janeiro de 2026*  
**Status: PRONTO PARA COMEÇAR** 🚀
