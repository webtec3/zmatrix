# Lifecycle e alocação CUDA — 2026-07-18

## Resultado

Esta rodada reduziu o custo estrutural dos resultados CUDA usando o pool nativo
do runtime (`cudaMallocAsync`/`cudaFreeAsync`) quando o device declara
`cudaDevAttrMemoryPoolsSupported`. O modo `auto` é o padrão interno, com
fallback para `cudaMalloc`/`cudaFree`; `ZMATRIX_CUDA_ALLOCATOR=legacy` força a
rota antiga. API PHP, `toGpu()`, dtype e versão não mudaram.

Também foi corrigida uma falha real: o handler Zend não implementava clone e
`clone $tensor` produzia um objeto com ponteiro C++ nulo. O clone agora é uma
cópia profunda, inclusive D2D quando o original está apenas no device.

## Decomposição do wrapper

Fonte: `wrapper_decomposition_20260718_235325.json`. Eventos CUDA medem device;
relógio monotônico mede host. Valores são medianas em ms.

| Operação | criação | alocação | operando | submissão | estado | retorno Zend | device |
|---|---:|---:|---:|---:|---:|---:|---:|
| greater 1M | 0,00022 | 0,01133 | 0 | 0,09548 | 0,00022 | 0,02189 | 0,06045 |
| tile 1024²×2 | 0,00022 | 0,01089 | 0 | 0,10692 | 0,00022 | — | 0,06451 |
| cumsum 1M | 0,00022 | 0,01111 | 0 | 0,11264 | 0,00055 | — | 0,15872 |
| matvec 2048² | 0,00022 | 0,00924 | 0,00781 | 0,17292 | 0,00033 | 0,02596 | 0,10240 |

Parsing Zend ficou abaixo da resolução do relógio nas medianas. O profiling
cria/sincroniza eventos e aumenta `submissão`; permanece desativado em produção.

Lifecycle steady para buffers de 4 MiB no modo auto:

```text
cudaMallocAsync host: 0,00308 ms
cudaFreeAsync host:   0,00154 ms
destrutor completo:   0,00495 ms
```

## Antes e depois

Baseline: `wrapper_lifecycle_20260718_232735.json`. Final instalado:
`wrapper_lifecycle_20260718_235311.json`, com 3 warmups e 15 repetições.

| Operação | Componente | Antes | Depois auto | Variação |
|---|---|---:|---:|---:|
| greater 1M | wrapper | 1,27732 | 0,15356 | -88,0% |
| greater 1M | destruição | 0,18986 | 0,00165 | -99,1% |
| tile 1024²×2 | wrapper | 1,57817 | 0,21384 | -86,5% |
| tile 1024²×2 | destruição | 0,26059 | 0,00165 | -99,4% |
| cumsum 1M | wrapper | 1,30548 | 0,03861 | -97,0% |
| cumsum 1M | destruição | 0,16995 | 0,00132 | -99,2% |
| dot 1M | wrapper misto | 0,11770 | 0,30019 | regrediu/inconclusivo |
| matvec 2048² | wrapper misto | 0,12991 | 0,13629 | neutro |

O async enfileira trabalho; tempos de wrapper não são E2E. Os benchmarks
isolados abaixo materializam o resultado quando aplicável.

## Alocação por tamanho

Fontes: `allocation_lifecycle_20260718_235324/235325.json`. Valores representam
tempo host de operação/free.

| Buffer | legacy operação/free | auto operação/free |
|---:|---:|---:|
| 1 KiB | 0,04400 / 0,00198 | 0,09933 / 0,00187 |
| 1 MiB | 0,50688 / 0,18612 | 0,02838 / 0,00121 |
| 4 MiB | 1,06073 / 0,15400 | 0,02640 / 0,00121 |
| 16 MiB | 2,08890 / 0,36575 | 0,04125 / 0,00132 |
| 64 MiB | 5,79513 / 0,24398 | 0,12628 / 0,00165 |

Para 1 KiB o auto não compensa na criação, mas a diferença é ~0,055 ms. Para
buffers maiores, remove a sincronização estrutural do free.

## Dot e matvec isolados

Fonte: `isolated_linalg_20260718_235403.json`. Cada cenário inicia processo PHP
novo, faz 3 warmups e 15 repetições. Operandos CPU são recriados em cada amostra.

| Operação | Cenário | operação | materialização | destruição | E2E |
|---|---|---:|---:|---:|---:|
| dot 1M | CPU+CPU | 1,75549 | 0,00088 | 0,00033 | 1,75934 |
| dot 1M | GPU+GPU | 0,09273 | 0,00011 | 0,00011 | 0,09482 |
| dot 1M | GPU+CPU | 2,18526 | 0,00077 | 0,00022 | 2,18933 |
| dot 1M | CPU+GPU | 3,23488 | 0,00088 | 0,00022 | 3,23807 |
| matvec 2048² | CPU+CPU | 6,00072 | 0,00187 | 0,00121 | 6,00666 |
| matvec 2048² | GPU+GPU | 0,09405 | 0,15037 | 0,00429 | 0,24651 |
| matvec 2048² | GPU+CPU | 0,18953 | 0,13398 | 0,00451 | 0,32703 |
| matvec 2048² | CPU+GPU | 32,07622 | 0,13904 | 0,00924 | 32,15718 |
| matvec 4096² | GPU+GPU | 0,04950 | 0,26675 | 0,00330 | 0,32285 |

O contrato foi preservado: se um operando está no device, `ensure_device`
promove o outro objeto original e essa residência persiste. Transferir a matriz
é o pior cenário; transferir apenas o vetor custa pouco.

## Lifecycle e invariantes

| Estado | host válido | device válido | write pending | buffers |
|---|---:|---:|---:|---|
| host-only | sim | não | não | host |
| sincronizado | sim | sim | não | host+device |
| device-only concluído | não | sim | não | device |
| device write pendente | não | não | sim | device obrigatório/capacidade suficiente |
| vazio | sim | opcional | não | nenhum obrigatório |
| movido | sim/vazio | não | não | ponteiro transferido e origem limpa |

`device_write_pending` só é marcado depois da alocação. A factory
`create_device_result()` cria shape/strides sem storage de elementos host,
aloca o device e devolve resultado não publicado. `mark_device_modified()`
resolve o pending. Move transfere ponteiro, capacidade, modo e pending;
copy/clone alocam buffer independente e fazem D2D; destrutor não lança e libera
uma vez. Operações na stream padrão consomem dependências sem sync host.

## Memória e stress

O stress executou 1.000 operações alternando greater/tile/cumsum/matvec/dot,
clone device-only, shapes diferentes, exceção e GC:

```text
auto: RSS 262.131.712 -> 282.365.952 bytes (+20.234.240), depois platô
PHP:  2.097.152 bytes; pico 2.097.152 bytes
```

O crescimento inicial é contexto/runtime/cache intencional. Dez amostras
subsequentes ficaram estáveis. O pool nativo pode reter VRAM durante o processo
e devolve no shutdown; a extensão não criou pool próprio. O workspace CUB
continua limitado pela maior capacidade solicitada e é liberado no shutdown.
Shutdown com 128 tensores device vivos passou. Não foi exposta API de VRAM,
portanto memória global do driver não foi atribuída a um operador específico.

## Código, testes e builds

* `src/gpu_kernels.cu`/`gpu_wrapper.h`: capability, malloc/free async e fallback.
* `src/zmatrix.cpp`: factory, modo por buffer, telemetria, invariantes,
  copy/move/destrutor e clone Zend profundo.
* `src/zmatrix_methods.h`: decomposição Zend de greater/dot/matvec.
* `allocator_lifecycle_stress.php`: 1.000 operações, clone, exceção, RSS e GC.
* `shutdown_live_tensors.php`: shutdown com 128 buffers vivos.
* Benchmarks lifecycle/decomposition/isolated: JSON bruto versionado.

Validações:

* CUDA clean/configure/build/install: PASS.
* PHPT basic/complete: 2/2 PASS.
* SGEMM, elementwise, reduções, sqrt, clip, softmax, greater, broadcast, tile,
  cumsum, dot, matvec, residência e exception safety: PASS.
* CUDA debug invariants, auto+legacy e stress: PASS.
* CPU-only sem dependência CUDA; ASan/UBSan: PASS.
* ZMatrix + RubixNumPower/NDArray + ORT: PASS.
* Compute Sanitizer 2022.4.1: bloqueado com `Unable to find injection library
  libsanitizer-collection.so`, inclusive com `LD_LIBRARY_PATH` explícito.

```text
versão: 0.4.0-float
build:   /home/omgaalfa/php-projetos/php-extension/zmatrix/modules/zmatrix.so
install: /usr/lib/php/20240924/zmatrix.so
SHA-256: ad3ed77ef41feb8d8dee127668067e7676ed268f744f8f0c45a89b16dd2530bb
allocator: auto (async quando suportado, fallback legacy)
```

## Tentativas rejeitadas e pendências

* Reuso de buffer público: rejeitado por aliasing/imutabilidade.
* Pool global próprio: rejeitado; o pool nativo produziu o ganho.
* Redução do lock CUB: não mantida; workspace compartilhado exige proteção.
* Cache de metadata: criação medida em ~0,00022 ms, sem ganho justificável.
* Dot GPU+CPU/CPU+GPU continua dominado por H2D.
* Buffers minúsculos podem ser neutros ou ligeiramente piores no modo auto.

Não foram criados executor, scheduler, streams por tensor, CUDA Graphs, API
nova, tag, release ou alteração de versão.
