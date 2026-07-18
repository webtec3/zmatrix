# Auditoria e correção do backend CUDA da ZMatrix

Data: 2026-07-18

## 1. Resumo executivo

O backend partia de um SGEMM cuBLAS com mapeamento row-major incorreto, despacho automático por tamanho, carregamento de `libcuda` que podia aceitar stubs de distribuição no WSL, erros CUDA ignorados e diversos acessos diretos a buffers host potencialmente obsoletos.

O SGEMM foi corrigido e validado em matrizes quadradas, retangulares, vetores linha/coluna, identidade, zeros, valores negativos/fracionários, escalas mistas e dados pseudoaleatórios reproduzíveis. A seleção da GPU voltou a ser estritamente explícita por `toGpu()`. O resultado permanece no device, e o buffer `C` não sofre H2D quando `beta = 0`.

O estado host/device passou a respeitar as transições documentadas abaixo. Chamadas CUDA/cuBLAS relevantes verificam seus retornos e chegam ao PHP como exceções C++, sem `abort()`. Os kernels existentes e os novos caminhos de divide, pow, log, fill, transpose e reduções foram comparados com a CPU.

Limitações verificadas: Compute Sanitizer 12.0 não conseguiu injetar sua biblioteca no processo PHP/WSL desta máquina; Linux nativo com GPU e multi-GPU não estavam disponíveis; reduções CUDA priorizam corretude e usam um laço serial por saída, ainda sem redução hierárquica otimizada.

## 2. Problemas encontrados e corrigidos

| Prioridade | Arquivo/função | Causa raiz | Impacto | Correção |
|---|---|---|---|---|
| P0 | `src/gpu_kernels.cu::gpu_matmul_device` | `m/n`, ordem A/B e leading dimensions descreviam outra operação column-major | Resultado GPU incorreto (`76` em vez de `58`) | cuBLAS agora calcula `C_col[N,M] = B_col[N,K] * A_col[K,M]` |
| P0 | `src/zmatrix.cpp::matmul` | Heurística por tamanho chamava GPU sem `toGpu()` | Violação do contrato público | GPU somente quando ao menos um operando já está residente |
| P0 | `src/zmatrix.cpp::matmul` | `result.ensure_device()` enviava zeros de C antes de SGEMM | H2D inútil com `beta=0` | `allocate_device_for_write()` aloca sem upload e só valida após sucesso |
| P0 | `src/gpu_kernels.cu::load_cuda_driver` | Ordem favorecia loader genérico antes do proxy WSL | Stub/biblioteca errada podia ser aceita | `/usr/lib/wsl/lib` tem prioridade e bibliotecas fora desse diretório são recusadas no WSL |
| P0 | `src/gpu_kernels.cu::gpu_available` | `dlopen()` era tratado como disponibilidade | Falso positivo de CUDA | Validação por símbolos Driver API, `cuInit`, `cuDeviceGetCount`, runtime e `cudaFree(nullptr)` |
| P0 | kernels e SGEMM | Erros de launch, execução, cópia e cuBLAS eram ignorados | Corrupção silenciosa/flags inválidas | Helpers contextualizados verificam CUDA, launch, sync e cuBLAS |
| P0 | buffers host em vários métodos | Leitura direta de `data` sem D2H | Resultado CPU baseado em bytes obsoletos | `ensure_host()` antes das leituras e `mark_host_modified()` após escritas |
| P1 | cache host-I/O CUDA | Cache global sem mutex e falhas de alocação silenciosas | Corrida e retorno incorreto | Mutex e exceções em allocation/free |
| P1 | copy/move/destructor/handle | Resultado/handle sem lifecycle completo | Vazamento ou estado ambíguo | saída device-only, cópias D2D, cleanup cuBLAS e driver registrados no shutdown |
| P1 | build sem CUDA | `freeDevice()` chamava membro inexistente | Build CPU-only quebrado | chamada protegida por `HAVE_CUDA` |
| P2 | divide/pow/log/fill | Caminhos sempre baixavam para host | Perda de residência | kernels CUDA, incluindo validação prévia de zero e domínio do log |
| P2 | transpose | D2H + transpose CPU | Transferência e perda de residência | kernel tiled `32x32`, padding para evitar bank conflict |
| P2 | sum/min/max/argmin/argmax | Reduções sempre liam tensor inteiro no host | D2H completo | redução no device; somente escalares públicos são baixados |
| P2 | zeros/ones | Criação estática deve continuar host-only | Não havia geração device explícita | após `toGpu()`, buffers uniformes 0/1 são preenchidos no device sem H2D |

Métodos de coerência revisados/corrigidos incluem `clip`, `sum(axis)`, `dot`, `minimum`, `maximum`, `broadcast`, `tile`, `zeroGrad`, `findIndicesWhere`, `calculateSplitGini`, `softmax` 1D e closures de autograd. A revisão também cobriu slice, argsort, mode, cumsum, stack, concat, histogram e demais leituras diretas; os últimos já possuíam sincronização adequada.

## 3. SGEMM row-major

Para buffers row-major:

- `A_row` tem shape `M x K`, mas seus mesmos bytes representam `A_col` com shape `K x M`;
- `B_row` tem shape `K x N`, mas representam `B_col` com shape `N x K`;
- `C_row` tem shape `M x N`, mas representam `C_col` com shape `N x M`.

Logo, `(A_row * B_row)^T = B_col * A_col`. A chamada final usa:

- operandos: `d_b`, depois `d_a`;
- operações: `CUBLAS_OP_N`, `CUBLAS_OP_N`;
- dimensões cuBLAS: `m=N`, `n=M`, `k=K`;
- `lda=N`, `ldb=K`, `ldc=N`;
- `alpha=1`, `beta=0`.

As dimensões são validadas contra o intervalo de `int` do cuBLAS. `device_valid` do resultado só é marcado após retorno cuBLAS bem-sucedido e sincronização CUDA bem-sucedida.

## 4. Máquina de estados host/device

| Evento | `host_valid` | `device_valid` |
|---|---:|---:|
| criação/escrita host | true | false |
| H2D concluído | true | true |
| kernel/escrita device concluído | false | true |
| D2H concluído | true | true |
| `toCpu()` explícito | true | false |
| falha antes de escrita device | estado anterior preservado | estado anterior preservado |
| falha ao criar/copiar destino | destino nunca é marcado válido | false |

`allocate_device_for_write()` mantém o destino inválido enquanto apenas existe armazenamento alocado. `mark_device_modified()` é chamado somente depois de sucesso confirmado. As validações de divide/log ocorrem antes do kernel mutador, tornando essas falhas atômicas para os dados.

## 5. Tratamento de erros e sincronização

`CUDA_CHECK_CONTEXT`, `CUDA_KERNEL_CHECK` e `CUBLAS_CHECK_CONTEXT` incluem operação, expressão, arquivo, linha, descrição e código. Erros de launch são observados por `cudaPeekAtLastError()`; erros assíncronos são observados por `cudaDeviceSynchronize()`.

As sincronizações por operação foram mantidas deliberadamente. O contrato PHP é síncrono e a extensão precisa atribuir uma falha CUDA à chamada PHP correta antes de marcar o buffer como válido. D2H também sincroniza implicitamente. Remover essas barreiras exigirá um modelo futuro de erro assíncrono/stream por tensor.

## 6. Lifecycle

- cada `ZTensor` é dono de seu `d_data` e `d_capacity`;
- cópia device-resident usa D2D e preserva quais versões são válidas;
- movimentos transferem ownership e anulam a origem;
- destrutores não lançam;
- caches host-I/O são serializados;
- handle cuBLAS é serializado, criado sob demanda e destruído no shutdown;
- driver CUDA é carregado sob demanda, portanto uma requisição CPU não inicializa contexto via código da ZMatrix;
- build CPU-only foi compilado e testado separadamente.

## 7. Testes e ferramentas

Arquivos adicionados:

- `tests/cuda/sgemm_correctness.php`;
- `tests/cuda/elementwise_correctness.php`;
- `tests/cuda/residency_coherence.php`;
- `tests/cpu_lifecycle.php`;
- `tests/cpu_no_cuda.php`;
- `tests/build_cpu_sanitized.sh`;
- `benchmarks/cuda_backend_benchmark.php`.

Resultados finais neste ambiente:

- PHPT: 2/2 passaram, incluindo os 60 métodos existentes;
- SGEMM CUDA: 11 grupos passaram;
- resolução WSL: a suíte SGEMM passou com `LD_LIBRARY_PATH` removido; o módulo registra `libcuda.so.1` em `DT_NEEDED` e resolve `/usr/lib/wsl/lib/libcuda.so.1` pelo RUNPATH;
- integração do PHP CLI: como `ndarray.so` era carregada diretamente no `php.ini` antes de `conf.d` e abria primeiro o stub da distribuição, a entrada de ZMatrix foi posicionada antes dela; NDArray, ZMatrix e ORT agora coexistem usando o proxy `/usr/lib/wsl/lib/libcuda.so.1`;
- elementwise/transpose/reduções: 40 grupos passaram;
- coerência/lifecycle/falhas: 7 grupos passaram;
- build sem CUDA: compilou; PHPT 2/2; `toGpu()` recusado sem corromper estado;
- ASan + UBSan, build CPU-only: passou;
- Valgrind, build CPU-only: 0 erros, 0 bytes definitivamente/possivelmente perdidos;
- Compute Sanitizer: indisponível na prática; falhou antes da primeira API instrumentada por incompatibilidade de injeção no WSL, mesmo com `--injection-path` e `--target-processes all`.

## 8. Benchmark validado

Hardware/software: RTX 3060 12 GiB, driver 576.02, WSL2, CUDA/nvcc 12.0, PHP 8.4.16, GCC 13.3.0. Shape `512x512`, 3 warmups e 15 repetições. Todos os resultados foram validados com tolerância absoluta e relativa antes da publicação.

| Medição | Mediana (ms) | Média (ms) | Desvio (ms) |
|---|---:|---:|---:|
| CPU SGEMM | 3.523 | 8.719 | 12.589 |
| H2D dos dois operandos | 0.851 | 1.457 | 1.673 |
| SGEMM GPU residente | 0.681 | 1.157 | 1.640 |
| D2H do resultado | 0.264 | 0.271 | 0.026 |
| GPU end-to-end | 1.919 | 1.939 | 0.241 |
| cadeia elementwise residente | 0.400 | 0.521 | 0.406 |

Memória observada ao final: PHP 26 MiB, RSS 299.1 MiB e 0 MiB atribuídos ao PID pelo `nvidia-smi` após a sincronização final. A memória PHP não inclui todas as alocações nativas/CUDA, por isso as três métricas são mantidas separadas.

## 9. Cobertura atual

| Operação | CPU | CUDA | Teste CPU/GPU | Mantém residência | Observações |
|---|---:|---:|---:|---:|---|
| add/sub/mul | sim | sim | sim | sim | tensor e escalar |
| scalar div | sim | sim | sim | sim | semântica IEEE existente |
| divide tensor | sim | sim | sim | sim | valida zero antes de mutar |
| relu/leakyRelu/sigmoid/tanh/exp/abs | sim | sim | sim | sim | NaN e infinitos cobertos onde aplicável |
| pow/log/fill | sim | sim | sim | sim | log valida domínio |
| matmul | OpenBLAS | cuBLAS | sim | sim | GPU somente explícita |
| transpose | sim | sim | sim | sim | tiled; retangulares/não múltiplos cobertos |
| sum/mean/min/max global | sim | sim | sim | entrada sim | baixa apenas escalar |
| sum(axis) | sim | sim | sim | resultado sim | qualquer eixo contíguo row-major |
| argmin/argmax global e eixo | sim | sim | sim | resultado de eixo sim | primeiro índice em empates |
| zeros/ones após `toGpu()` | sim | sim | sim | sim | device fill explícito, sem H2D |

## 10. Operações ainda CPU-only

Permanecem CPU-only operações fora do conjunto P2 solicitado ou sem kernel nesta etapa: sigmoid/relu/tanh derivatives (exceto leaky derivative existente), sqrt, softmax, softmax derivative, clip, broadcast, tile, dot 1D/matvec, slice, reshape, argsort, sort, mode, variance/std, percentile, histogram, cumsum, unique, stack, concatenate, comparisons, filtros/ML helpers e caminhos de autograd. Ao receber um tensor device-only, esses métodos sincronizam corretamente com o host e invalidam o device quando escrevem.

## 11. Recomendações

1. substituir reduções seriais por reduções hierárquicas/CUB e medir ordem numérica;
2. adotar stream por contexto/tensor e eventos CUDA para erro assíncrono e benchmark de kernel puro;
3. introduzir allocator/pool para temporários de validação e reduções;
4. avaliar operações fusionadas para cadeias elementwise;
5. adicionar views/strides e transpose lazy;
6. avaliar cuBLASLt para layouts explícitos e tuning;
7. executar Compute Sanitizer/Nsight em Linux nativo com toolkit compatível;
8. criar CI CPU-only e runner GPU, incluindo fault injection de malloc/H2D/D2H;
9. adicionar testes multi-GPU e de concorrência real entre requests.
