# Evolucao pratica C++/CUDA da ZMatrix

Data: 2026-07-18

Escopo: endurecimento interno, reducoes CUDA e migracao de `sqrt`, `clip`,
`softmax` e `softmaxDerivative` para CUDA, sem alterar a API PHP, a versao
publica ou a regra de residencia explicita por `toGpu()`.

## Alteracoes por arquivo

| Arquivo e funcao | Problema | Correcao |
|---|---|---|
| `src/zmatrix.cpp`, lifecycle e transicoes | flags host/device eram manipuladas em varios pontos; `freeDevice()` podia descartar o unico valor valido | transicoes centralizadas, invariantes opcionais de desenvolvimento e download antes da liberacao publica do device |
| `src/zmatrix.cpp`, construcao/`size()` | multiplicacoes de shape nao tinham uma unica politica de overflow | `checked_element_count()` valida a multiplicacao antes de alocar ou indexar |
| `src/zmatrix.cpp`, `sqrt`, `clip_values`, `softmax`, `softmax_derivative` | operacoes baixavam para CPU mesmo com residencia explicita | despacho CUDA somente se `device_valid`; CPU continua sendo o padrao |
| `src/zmatrix_methods.h`, `reshape` | excecao C++ de shape/overflow podia atravessar a ABI C da Zend e encerrar o PHP | conversao para excecao PHP no ponto de entrada |
| `src/zmatrix_methods.h`, `clip` | implementacao PHP manipulava host diretamente | delegacao ao metodo interno que preserva residencia e contrato |
| `src/gpu_kernels.cu`, temporarios | divide/log e reducoes tinham cleanup manual ou `cudaMalloc`/`cudaFree` repetido | RAII para temporarios de validacao e cache serializado simples para reducao global |
| `src/gpu_kernels.cu`, `kernel_softmax_rows` | nao existia softmax CUDA; durante os testes, uma barreira insuficiente permitiu sobrescrever o maximo compartilhado | algoritmo estavel por linha, maximo compartilhado separado e barreira antes da etapa de soma |
| `src/gpu_kernels.cu`, novos kernels | `sqrt`, `clip` e derivada de softmax eram CPU-only | kernels e wrappers sincronizados, com validacao anterior a mutacao quando aplicavel |
| `src/gpu_kernels.h`, `src/gpu_wrapper.h` | wrappers novos nao estavam expostos ao core | declaracoes internas adicionadas, sem API PHP nova |

## Invariantes e exception safety

`ZMATRIX_ENABLE_DEBUG_INVARIANTS` habilita verificacoes que lancam
`std::logic_error`; builds de producao nao executam `abort()`. Para tensores nao
vazios, pelo menos uma representacao deve ser valida. Um device valido exige
buffer e capacidade suficientes, e um host valido exige storage suficiente.

As flags somente mudam depois de copia ou kernel bem-sucedido. As validacoes de
dominio de `sqrt`, divide e log ocorrem antes do kernel mutador. `ScopedCudaBuffer`
garante cleanup em excecoes. O cache de reducao e deliberadamente pequeno,
sincrono e protegido por mutex; nao constitui pool geral, executor ou scheduler.

## Semantica preservada

- CPU permanece o backend padrao; nenhum caminho chama GPU sem `toGpu()`.
- `sqrt` rejeita negativos antes de alterar o tensor, como a CPU; NaN e `+Inf`
  sao preservados.
- `clip` preserva a ordem exata de comparacao do backend CPU e rejeita limites
  invertidos ou NaN.
- `softmax` subtrai o maximo antes de `exp`; os comportamentos existentes de 1D,
  2D, NaN e infinitos foram reproduzidos.
- reducoes continuam usando CUB somente para soma global. Min, max, argmin e
  argmax mantem a implementacao propria e o primeiro indice em empates.

## Testes adicionados ou ampliados

| Teste | Cobertura | Resultado |
|---|---|---|
| `tests/core_validation.php` | overflow, shape negativo, reshape com excecao PHP, limites de clip, CPU default e lifecycle | passou |
| `tests/cuda/new_kernels_correctness.php` | sqrt, clip, softmax, derivada, vazios, NaN, infinitos, negativos, residencia, cadeia e softmax grande | passou |
| `tests/cuda/reduction_strategies.php` | serial, hierarquica, CUB e 200 somas repetidas com cache | passou |
| `tests/cuda/residency_coherence.php` | coerencia atualizada sem tratar softmax como CPU-only | passou |
| `tests/build_cuda_debug_invariants.sh` | build CUDA limpo com invariantes e testes de estado | passou |

Regressoes executadas: PHPT, SGEMM (11 grupos), elementwise/transpose/reducoes,
coerencia, lifecycle, build CPU-only, ASan/UBSan CPU e novos kernels. O wrapper
geral de `make test` sob LeakSanitizer observou 2 bytes em `/usr/bin/sed`; o
executavel da extensao e o stress de lifecycle passaram, portanto esse achado
externo nao foi atribuido a ZMatrix.

## Benchmark validado

Ambiente: RTX 3060 12 GiB, WSL2, CUDA 12.0, PHP 8.4, sete repeticoes. Todas as
amostras abaixo passaram pela referencia CPU. Medianas em milissegundos.

| Operacao | CPU | H2D | Kernel/operacao GPU | D2H | End-to-end | Resultado residente |
|---|---:|---:|---:|---:|---:|---:|
| sqrt | 1.086 | 1.812 | 1.235 | 0.816 | 4.013 | sim |
| clip | 1.292 | 1.387 | 3.003 | 0.844 | 5.185 | sim |
| softmax | 10.792 | 1.469 | 0.562 | 0.852 | 2.733 | sim |

Uma operacao isolada de sqrt ou clip nao amortizou as transferencias neste
hardware. Softmax teve aproximadamente 3,66x de ganho end-to-end e 16,35x na
fase de operacao contra o caminho GPU anterior, que na realidade baixava e
executava na CPU. A comparacao anterior distribui custos entre colunas de forma
diferente e por isso nao deve ser interpretada como tempo de kernel antigo.

O cache simples de temporarios reduziu a mediana da soma global validada de
0,23287 para 0,19767 ms em `1024^2` (15,1%) e de 0,29920 para 0,23958 ms em
`2048^2` (19,9%). Dados brutos estao em `benchmarks/v050/results/`.

## Pendencias conscientes

Continuam CPU-only: broadcast, tile, cumsum, comparacoes, dot vetorial/matvec,
slice, ordenacao, estatistica adicional e varios helpers de ML/autograd. Nao
foram introduzidos executor, streams, eventos, pool complexo, graphs, fusao
generica ou despacho automatico. Profiling Nsight importavel permanece requisito
antes dessas mudancas arquiteturais.

Nenhuma tag ou release foi criada e a versao publica permanece
`0.4.0-float` conforme a autorizacao desta rodada.
