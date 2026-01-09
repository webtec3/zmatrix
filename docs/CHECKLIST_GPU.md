# 🔧 CHECKLIST TÉCNICO: GPU Implementation para ZMatrix

**Objetivo**: Validar pré-requisitos e começar implementação GPU  
**Versão**: 1.0  
**Atualizado**: Janeiro 2026

---

## 1️⃣ VERIFICAÇÃO PRÉ-REQUISITOS

### Passo 1.1: Verificar CUDA

```bash
# 1. CUDA compiler
nvcc --version
# Esperado: cuda_11.0 ou superior

# 2. CUDA directory
ls -d /usr/local/cuda
echo $CUDA_HOME

# 3. Bibliotecas
ls /usr/local/cuda/lib64/libcudart.so
ls /usr/local/cuda/lib64/libcublas.so
ls /usr/local/cuda/lib64/libcurand.so

# 4. Headers
ls /usr/local/cuda/include/cuda_runtime.h
ls /usr/local/cuda/include/cublas_v2.h
ls /usr/local/cuda/include/curand.h
```

**Resultado esperado**:
```
✅ CUDA Compilation tools, release 11.0 (ou superior)
✅ /usr/local/cuda/
✅ Todos os .so encontrados
✅ Todos os headers encontrados
```

**Se falhar**:
- Instalar CUDA: `sudo apt-get install nvidia-cuda-toolkit`
- Ou download: https://developer.nvidia.com/cuda-downloads

### Passo 1.2: Verificar GPU

```bash
nvidia-smi
# Esperado: Tabela com GPU info
```

**Resultado esperado**:
```
+-----------------------+------------------------+
| NVIDIA-SMI 460.91     | Driver Version: 460.91 |
+-----------------------+------------------------+
| GPU Name     Compute SM | Memory Usage         |
|Tesla V100         7.0  | 0MiB / 32000MiB      |
+-----------------------+------------------------+
```

**Se falhar**:
- Sem GPU = código CUDA não pode rodar
- Mas PODE compilar com `-arch=sm_70` (sem testar)

### Passo 1.3: Verificar Compilador C++

```bash
g++ --version
# Esperado: g++ (GCC) 9.0 ou superior

clang++ --version  
# Ou: clang version 10.0 ou superior
```

**Se falhar**: `sudo apt-get install build-essential`

### Passo 1.4: Verificar PHP Dev

```bash
php-config --version
php-config --includes
php-config --libs

# Ou se não encontrado:
which php
/usr/bin/php --version
```

**Se falhar**: `sudo apt-get install php-dev`

### Passo 1.5: Verificar Build Tools

```bash
which phpize
which autoconf
which automake
which make
```

**Se falhar**: `sudo apt-get install autoconf automake libtool`

---

## 2️⃣ VALIDAÇÃO DO CÓDIGO EXISTENTE

### Passo 2.1: Verificar gpu_kernels.h

```bash
cd ~/php-projetos/php-extension/zmatrix
cat src/gpu_kernels.h | grep "extern"
```

**Esperado**: 30 linhas com `extern "C" void gpu_*(...)`

**Checklist**:
- [ ] Arquivo existe
- [ ] 30 funções declaradas
- [ ] Nomes estão em `extern "C"`

### Passo 2.2: Verificar gpu_kernels.cu

```bash
wc -l src/gpu_kernels.cu
cat src/gpu_kernels.cu | grep "__global__"
```

**Esperado**: 
- ~30 linhas total
- 1 kernel `__global__` (gpu_add)
- Resto são stubs ou comentários

### Passo 2.3: Verificar gpu_wrapper.h

```bash
cat src/gpu_wrapper.h
```

**Esperado**:
```cpp
#pragma once

#ifdef HAVE_CUDA
extern "C" void gpu_add(float* a, const float* b, size_t n);
#endif
```

### Passo 2.4: Verificar config.m4

```bash
grep -A 5 "HAVE_CUDA" config.m4
grep "nvcc" config.m4
```

**Esperado**: Detecção automática de CUDA

---

## 3️⃣ COMPILAÇÃO DE TESTE

### Passo 3.1: Compilar arquivo .cu isolado

```bash
cd src/
nvcc -c gpu_kernels.cu -o gpu_kernels.o
ls -la gpu_kernels.o
```

**Resultado esperado**: `gpu_kernels.o` criado (~50-200KB)

**Se falhar**:
```
error: cuda_runtime.h: No such file
→ CUDA headers não encontrados
→ Executar: export CUDA_PATH=/usr/local/cuda
→ E adicionar flags: nvcc -I/usr/local/cuda/include ...
```

### Passo 3.2: Compilar com PHP

```bash
cd ~/php-projetos/php-extension/zmatrix
./configure --enable-zmatrix --with-cuda-path=/usr/local/cuda
make clean
make -j$(nproc)
```

**Resultado esperado**:
```
Build complete.
Don't forget to run 'make test'.

Installing shared extensions: /usr/lib/php/extensions/...
```

**Se falhar com CUDA**:
```
conftest.c:1: error: 'cuda_runtime.h' file not found
→ config.m4 não detectou CUDA
→ Verificar: ls /usr/local/cuda/include/
→ Se vazio, CUDA não instalado
```

### Passo 3.3: Instalar

```bash
sudo make install
php -m | grep zmatrix
```

**Resultado esperado**: `zmatrix` na lista de módulos

---

## 4️⃣ TESTE FUNCIONAL BÁSICO

### Passo 4.1: Testar se CUDA foi detectado

```bash
php -r "
if (extension_loaded('zmatrix')) {
    echo 'ZMatrix carregado\n';
    echo phpversion('zmatrix') . '\n';
}
"
```

### Passo 4.2: Testar CPU fallback

```bash
php -r "
\$a = array_fill(0, 1000, 1.0);
\$b = array_fill(0, 1000, 2.0);
\$result = zmatrix_add(\$a, \$b);
var_dump(\$result[0]); // Esperado: 3.0
"
```

### Passo 4.3: Testar performance

```bash
php -c benchmark.php
# Deve mostrar tempos tanto para CPU quanto GPU (se GPU disponível)
```

---

## 5️⃣ IMPLEMENTAÇÃO PHASE 1: GPU_ADD

### Passo 5.1: Editar gpu_kernels.cu

Substituir:
```cuda-cpp
__global__ void kernel_add(float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] += b[i];
    }
}

extern "C" void gpu_add(float* a, const float* b, size_t n) {
    float *d_a = nullptr, *d_b = nullptr;

    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_add<<<blocks, threads>>>(d_a, d_b, n);

    cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
}
```

Por:

```cuda-cpp
// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            throw std::runtime_error("CUDA operation failed"); \
        } \
    } while(0)

__global__ void kernel_add(float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] += b[i];
    }
}

extern "C" void gpu_add(float* a, const float* b, size_t n) {
    if (n == 0) return;
    
    float *d_a = nullptr, *d_b = nullptr;

    // Alocate
    CUDA_CHECK(cudaMalloc((void**)&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, n * sizeof(float)));

    // Copy H2D
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

    // Execute kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_add<<<blocks, threads>>>(d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());

    // Sync
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy D2H
    CUDA_CHECK(cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}
```

### Passo 5.2: Compilar e testar

```bash
make clean
make -j$(nproc)
sudo make install

php -r "
\$a = array_fill(0, 1000000, 1.0);
\$b = array_fill(0, 1000000, 2.0);
\$start = microtime(true);
\$result = zmatrix_add(\$a, \$b);
echo 'GPU Add: ' . round((microtime(true) - \$start) * 1000, 2) . 'ms\n';
var_dump(\$result[0]); // 3.0
"
```

---

## 6️⃣ CHECKLIST DE IMPLEMENTAÇÃO

### Phase 1: Core (Semana 1)

- [ ] **gpu_add** (multiplicar)
  - [ ] Kernel CUDA escrito
  - [ ] CUDA_CHECK macro aplicado
  - [ ] Sincronização com cudaDeviceSynchronize()
  - [ ] Testado em PHP
  - [ ] Benchmark comparado com CPU

- [ ] **gpu_subtract**
  - [ ] Implementado (copiar de gpu_add, mudar operação)
  - [ ] Testado

- [ ] **gpu_multiply** (element-wise)
  - [ ] Implementado
  - [ ] Testado

- [ ] **gpu_transpose** (2D)
  - [ ] Kernel com shared memory (otimizado)
  - [ ] Testado com 4k×4k matrix

### Phase 2: Activation (Semana 2)

- [ ] **gpu_sigmoid**
  - [ ] Kernel com `expf()`
  - [ ] Testado em 1M elementos

- [ ] **gpu_relu**
  - [ ] Kernel com `fmaxf()`
  - [ ] Testado

- [ ] **gpu_tanh**
  - [ ] Kernel com `tanhf()`
  - [ ] Testado

- [ ] **gpu_softmax**
  - [ ] Kernel com reduce (max, exp, sum)
  - [ ] Testado com batches

### Phase 3: Advanced (Semana 3)

- [ ] **gpu_matmul** (usando cuBLAS)
  - [ ] `cublasSSgemm()` integrado
  - [ ] Testado com 1000×1000

- [ ] **gpu_sum_all, gpu_mean_all**
  - [ ] Redução com `__shared__`
  - [ ] Testado

---

## 7️⃣ TESTE FINAL

```bash
# Test suite completo
php run-tests.php

# Benchmark
php benchmark_comparative.php

# Memory leak check (se disponível)
valgrind --leak-check=full php test.php

# Performance check
php -d auto_prepend_file='' -r "
// Teste performance GPU vs CPU
for (\$size = 10000; \$size <= 1000000; \$size *= 10) {
    \$a = array_fill(0, \$size, 1.0);
    \$b = array_fill(0, \$size, 2.0);
    
    \$start = microtime(true);
    \$result = zmatrix_add(\$a, \$b);
    \$gpu_time = (microtime(true) - \$start) * 1000;
    
    printf(\"Size: %d, Time: %.2fms\n\", \$size, \$gpu_time);
}
"
```

---

## 8️⃣ OTIMIZAÇÕES PÓS-IMPLEMENTAÇÃO

### Otimização 1: Memory Pooling

```cpp
// Usar cuMemoryPool (CUDA 11.2+)
cudaMemoryPool_t mempool;
CUDA_CHECK(cudaDeviceGetMemPool(&mempool, device));
cudaMemoryPoolSetAttribute(mempool, 
    cudaMemPoolAttrReleaseThreshold, UINT64_MAX);
// Malloc/free muito mais rápido
```

### Otimização 2: Pinned Memory

```cpp
// Alocate host memory que GPU pode acessar diretamente
cudaMallocHost((void**)&host_data, size);
// H2D/D2H ~2x mais rápido
```

### Otimização 3: Streams

```cpp
// Pipelining: transferência + kernel + transferência
cudaStream_t stream;
CUDA_CHECK(cudaStreamCreate(&stream));

CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, size, H2D, stream));
kernel<<<blocks, threads, 0, stream>>>(d_a, d_b, d_c);
CUDA_CHECK(cudaMemcpyAsync(h_c, d_c, size, D2H, stream));
CUDA_CHECK(cudaStreamSynchronize(stream));
```

---

## ✅ GO/NO-GO DECISION

### Pré-requisitos Met?

- [ ] CUDA 11.0+ instalado
- [ ] GPU com compute capability ≥5.0
- [ ] nvcc compila código CUDA
- [ ] PHP dev tools disponíveis
- [ ] config.m4 detecta CUDA

### Código Pronto?

- [ ] gpu_kernels.h tem 30 assinaturas
- [ ] gpu_kernels.cu compila
- [ ] gpu_wrapper.h é simples
- [ ] ZTensor está em float

### Time Disponível?

- [ ] 20+ horas para Phase 1-2
- [ ] Pode fazer 2-3h/dia
- [ ] Semanas 1-3 livres

### Se TUDO ✅

**Comece com Passo 5.1: gpu_add com CUDA_CHECK**

### Se algum ❌

**Implemente CPU-only** (muito mais rápido):
1. Descomentar OpenMP (5 min)
2. SIMD AVX2 (2-3h)
3. Ganho: 15x
4. Nenhuma complexidade CUDA

---

## 📞 Status Report

Assim que completar cada seção, reporte:

```
✅ Seção 1: CUDA detectado (nvcc v11.2)
✅ Seção 2: gpu_kernels.h tem 30 funções
⏳ Seção 3: Compilação em progresso...
❌ Seção 4: Erro em CUDA headers
```

Então posso ajudar com troubleshooting específico!

