# Otimização dos hot paths CUDA — 2026-07-18

## Escopo e conclusão

Esta rodada otimizou `greater`, `tile`, `cumsum`, `dot`, `matvec` e a criação
de resultados device-only sem alterar API, versão ou despacho. CPU continua
sendo o padrão; GPU continua dependendo de `toGpu()` explícito.

O gargalo medido dos operadores simples não era o kernel. Em 1M elementos,
os kernels/scan steady-state ficaram entre 0,028 e 0,061 ms, enquanto a
alocação device custou tipicamente 0,3–0,5 ms e a antiga inicialização do
vetor host custava 0,4–5 ms. A correção mantida elimina o zero-fill host para
resultados que serão totalmente escritos no device e remove sincronizações
globais dos resultados residentes.

## Alterações por arquivo

| Arquivo | Função/área | Problema medido | Correção |
|---|---|---|---|
| `src/gpu_kernels.cu` | `gpu_greater_device` | branch tensor/escalar por elemento e sync global | kernels especializados e launch check sem sync |
| `src/gpu_kernels.cu` | `gpu_tile_device` | módulo por elemento e sync global | expansão contígua com D2D assíncrono em progressão |
| `src/gpu_kernels.cu` | `gpu_cumsum_device` | consulta repetida do storage CUB e sync global | capacidade do workspace cacheada e launch check |
| `src/gpu_kernels.cu` | `gpu_dot_device` | sync redundante após `cublasSdot` host-pointer | removido; retorno escalar continua síncrono pela cuBLAS |
| `src/gpu_kernels.cu` | `gpu_matvec_device` | sync global após SGEMV | removido; resultado permanece device-resident |
| `src/gpu_kernels.cu` | `CudaProfileEvents` | custos internos não observáveis | eventos/log opcionais com `ZMATRIX_CUDA_PROFILE=1` |
| `src/zmatrix.cpp` | construtor interno/`ensure_host` | vetor host grande era zerado antes de ser sobrescrito no device | host storage diferido e materializado apenas no D2H |
| `src/zmatrix.cpp` | invariantes de estado | estado transitório device-write não era modelado | `device_write_pending`, limpo antes de o tensor escapar |
| `src/zmatrix.cpp` | wrappers de `greater`, `broadcast`, `tile`, `cumsum` | criação/alocação misturadas no tempo GPU | resultado device-only e tempos internos opcionais |
| `src/zmatrix_methods.h` | matvec em `dot` | resultado host alocado apesar de SGEMV sobrescrevê-lo | construção device-only e profiling do wrapper |
| `tests/cuda/extended_ops_correctness.php` | stress | faltavam repetição, alternância e recuperação | 200 iterações, crescimento CUB, aliasing e pós-exceção |

## Política de sincronização

| Ponto | Classificação | Política final |
|---|---|---|
| kernel produz tensor residente | desnecessária | `cudaPeekAtLastError`; erro assíncrono aparece no próximo sync real |
| D2D de `tile` na stream padrão | desnecessária | cópias enfileiradas, sem sync entre blocos |
| `cublasSgemv` com saída tensor | desnecessária | retorno residente, sem D2H |
| `cublasSdot` com pointer mode host | obrigatória por retorno PHP | a própria chamada entrega o escalar host; sync extra removido |
| `toCpu()`/`toArray()` | obrigatória por retorno host | `cudaMemcpy` D2H síncrono |
| validação de domínio de `sqrt` | obrigatória pela exceção síncrona atual | mantida |
| eventos da instrumentação | necessária apenas para benchmark | criados/sincronizados somente com profiling habilitado |

## Perfil interno final

Com `ZMATRIX_CUDA_PROFILE=1`, três warmups e sete amostras:

| Operação | Tamanho | Cold wrapper (ms) | Kernel/cuBLAS mediano (ms) | Wrapper mediano (ms) | Host output típico (ms) | Device alloc típico (ms) |
|---|---:|---:|---:|---:|---:|---:|
| greater | 1M | 8,867 | 0,028 | 1,787 | 0,001 | 0,30–0,45 |
| tile | 1024²×2 | 2,360 | 0,060 D2D | 2,277 | 0,001 | 0,35–0,50 |
| cumsum | 1M | 3,509 | 0,033 | 1,898 | 0,001 | 0,35–0,53 |
| matvec | 2048² | 23,118 | 0,060 | 0,253 | <0,001 | 0,005–0,007 |

Cold start inclui contexto CUDA/cuBLAS e primeira alocação. O evento mede só o
trabalho device; o wrapper PHP também inclui criação/destruição do objeto.

## Contadores antes e depois

| Operação | Antes | Depois |
|---|---|---|
| greater 1M | alloc 1; launch 1; device sync 1; H2D/D2H/D2D 0 | alloc 1; launch 1; device sync 0; H2D/D2H/D2D 0 |
| tile 1024²×2 | alloc 1; launch 1; device sync 1; D2D 0 | alloc 1; launch 0; device sync 0; D2D 2 |
| cumsum 1M | alloc 1; launch 1; device sync 1; workspace query repetida | alloc 1; launch 1; device sync 0; workspace query cacheada |
| matvec 2048² | alloc 1; cuBLAS 1; device sync 1 | alloc 1; cuBLAS 1; device sync 0 |

Frees continuam ocorrendo no destrutor do resultado; não foi introduzido pool.

## Antes e depois

Baseline: `extended_ops_20260718_221129.json`. Depois: execução final
`extended_ops_20260718_231126.json`, gerada pelo binário instalado e com sete
amostras. Valores em ms; a variação é da mediana final contra a baseline.

| Operação | Tamanho | Métrica | Antes | Depois p25/mediana/p75 | Variação |
|---|---:|---|---:|---:|---:|
| greater | 1M | kernel | não isolado | 0,028 | — |
| greater | 1M | residente | 3,396 | 1,147 / 1,162 / 1,307 | -65,8% |
| greater | 1M | E2E | 5,360 | 3,618 / 3,761 / 4,182 | -29,8% |
| tile | 1024²×2 | kernel/D2D | não isolado | 0,060 | — |
| tile | 1024²×2 | residente | 12,073 | 3,516 / 5,691 / 6,568 | -52,9% |
| tile | 1024²×2 | E2E | 16,261 | 10,652 / 13,040 / 14,462 | -19,8% |
| cumsum | 1M | scan | não isolado | 0,033 | — |
| cumsum | 1M | residente | 4,994 | 3,481 / 4,751 / 6,953 | -4,9% |
| cumsum | 1M | E2E | 8,580 | 6,899 / 7,189 / 7,833 | -16,2% |
| matvec | 2048² | cuBLAS | não isolado | 0,060 | — |
| matvec | 2048² | residente | 1,624 | 1,831 / 1,860 / 4,587 | +14,5% |
| matvec | 2048² | E2E | 7,515 | 7,773 / 8,168 / 10,364 | +8,7% |

O baseline não continha eventos, portanto não é correto inventar um “kernel
antes”. O tempo residente baseline inclui criação, alocação, launch e sync.

`dot` e o wrapper de `matvec` são classificados como inconclusivos: rodadas
anteriores melhoraram, mas a rodada final regrediu (`dot` residente +65,2%,
E2E +38,7%). O microbenchmark isolado continua mostrando cuBLAS em 0,060 ms e
matvec residente em 0,105 ms; a divergência confirma que pressão de
alocação/destruição e interferência entre casos dominam o benchmark misto.

## Matriz de tamanhos

Arquivo: `hotpath_matrix_20260718_231128.json/csv`. Três warmups, sete
amostras, cold separado e validação numérica antes de aceitar o caso.

* `greater`: 1K, 64K, 1M e 16M; medianas 0,050, 0,045, 1,084 e 5,535 ms.
* `tile`: 256², 1024² e 2048², fatores 2/4/8/16. Em 1024²: 1,339, 1,788,
  3,051 e 5,522 ms, respectivamente; a matriz atinge até ~13,7 GB/s.
* `cumsum` 1D: 1K, 64K, 1M e 8M; medianas 0,082, 0,087, 1,319 e 2,875 ms.
* `cumsum` 2D eixos 0/1: em 2048², 2,189 e 2,211 ms.
* `matvec` 2048²: ambos residentes 0,105 ms; matriz residente + vetor CPU
  0,213 ms; matriz CPU + vetor residente 28,745 ms. O último caso inclui a
  criação/cópia do operando de 16 MiB e confirma que residência importa.

## Cadeias residentes

Resultados da rodada final; p25/mediana/p75 ms.

| Cadeia | CPU | H2D | GPU residente | D2H | E2E | Classificação |
|---|---:|---:|---:|---:|---:|---|
| greater→cumsum→clip | 5,280/6,134/6,483 | 1,635/1,736/1,880 | 10,998/14,993/23,677 | 1,650/1,912/2,077 | 15,877/16,395/27,410 | regrediu/alta dispersão |
| broadcast→greater→cumsum | 22,290/23,100/23,386 | 1,868/1,926/2,044 | 12,940/13,590/15,178 | 1,927/2,066/2,117 | 17,433/18,415/20,640 | melhorou residente e E2E |
| matvec→clip→softmax | 5,383/5,454/5,696 | 5,266/5,614/6,891 | 2,056/2,213/3,097 | 0,068/0,077/0,102 | 7,586/8,129/8,950 | residente melhora; E2E perde |
| tile→sqrt→sum | 5,439/6,306/6,508 | 1,395/1,524/2,392 | 5,346/5,934/8,885 | 0,007/0,007/0,007 | 7,537/8,680/10,881 | residente melhora; E2E perde |

`tile→sqrt→sum` não é totalmente livre de sincronização interna: `sqrt`
preserva a validação de domínio síncrona exigida pela semântica de exceção.

## Tentativas rejeitadas

* Kernel `output[i] = input[i % input_size]` para `tile`: evento típico
  0,054–0,057 ms contra 0,055–0,060 ms no D2D; resultado neutro e wrapper D2D
  ligeiramente melhor. O seletor experimental foi removido.
* Reuso do output público/in-place: rejeitado por aliasing e imutabilidade.
* `cudaMallocAsync`: não mantido; sem executor/streams e com dispersão ainda
  dominada pelo lifecycle, criaria dependência de runtime sem evidência isolada.
* Remover validação de domínio de `sqrt`: rejeitado por alterar exceções.
* Primeira matriz `230225`: cenário matvec inválido porque o operando CPU ficou
  residente após a primeira chamada. Não entrou nos dados auditáveis; `230300`
  recria o operando CPU em cada amostra.

## Validação e ambiente

* PHP 8.4.16; extensão 0.4.0-float.
* GCC/G++ 13.3.0; C++17, `-O3 -march=native`, OpenMP.
* CUDA toolkit 12.0.140; driver 576.02; RTX 3060 12 GiB; WSL2.
* CUDA normal: clean/configure/build/install aprovado.
* PHPT: `basic.phpt` e `complete.phpt`, 2/2 PASS.
* CUDA: SGEMM, elementwise, reduções, sqrt, clip, softmax, residência,
  lifecycle, extended ops e cadeias PASS.
* CUDA debug invariants: PASS após modelar o estado device-write transitório.
* CPU-only: lifecycle e extended ops PASS; `ldd` do build não contém CUDA.
* ASan/UBSan CPU-only: PASS, leak detection ativa.
* Compatibilidade: carga conjunta `zmatrix`, `RubixNumPower` (`ndarray.so`) e
  `ort` PASS.
* Compute Sanitizer 2022.4.1: bloqueado no WSL com `Unable to find injection
  library libsanitizer-collection.so`, mesmo a biblioteca existindo em
  `/usr/lib/nvidia-cuda-toolkit/compute-sanitizer` e com `LD_LIBRARY_PATH`
  explícito. Nenhuma investigação ambiental adicional foi feita.

Módulo produzido e instalado:

```text
/home/omgaalfa/php-projetos/php-extension/zmatrix/modules/zmatrix.so
/usr/lib/php/20240924/zmatrix.so
SHA-256: a77c7f8ed5da03ecf48949d8d18ec5892a166ebc7314067a75ac0a31983d3b03
```

## Pendências

`cudaMalloc/cudaFree` e destruição do resultado continuam dominando caminhos
pequenos e explicam parte da dispersão. Não foi criado pool, executor, evento
persistente ou múltiplas streams. O ganho de cadeias depende do volume de
trabalho e da transferência; não há despacho automático. Nsight importável
continua necessário antes de qualquer evolução arquitetural.
