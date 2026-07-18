# Operacoes CUDA estendidas da ZMatrix

Data: 2026-07-18

Esta rodada implementou os contratos PHP existentes de `greater`, `broadcast`,
`tile`, `cumsum` e `dot` (vetorial, matrix-vector e matrix-matrix ja existente),
sem API nova, despacho automatico, alteracao de versao, tag ou release.

## Contratos encontrados

- A unica comparacao publica da ZMatrix e `greater`. A saida continua sendo
  `float`, com valores `0.0f` e `1.0f`. NaN segue a comparacao C++ `>` e produz
  falso. Escalar, tensor do mesmo shape, vetor unitario e broadcast 2D x 1D
  continuam aceitos.
- `broadcast` materializa o tensor passado no shape de `self`, alinhando shapes
  a direita e repetindo dimensoes de tamanho 1. Ele nao soma `self`; o comentario
  antigo da stub era impreciso. Rank de entrada maior e fonte vazia para saida
  nao vazia agora sao rejeitados antes da indexacao.
- `tile(tensor, times)` repete o bloco row-major completo no eixo zero e exige
  `times >= 1`. O novo calculo valida overflow antes de construir a saida.
- `cumsum` aceita 1D ou 2D. Em 2D, eixo ausente/negativo usa eixo 1; eixos 0 e 1
  sao aceitos. A API existente nao faz flatten de matriz quando o eixo e nulo.
- `dot` 1D x 1D retorna imediatamente um escalar PHP; 2D x 1D retorna tensor 1D;
  2D x 2D continua delegando para SGEMM.

## Implementacao

- `kernel_greater` cobre escalar, mesmo shape e vetor de broadcast.
- `kernel_broadcast_materialize` usa metadados pequenos copiados ao device e
  strides pre-computados; nenhum ponteiro host e dereferenciado pelo kernel.
- `kernel_tile` mapeia `output[i]` para `input[i % input_size]`.
- Vetores 1D usam CUB `DeviceScan::InclusiveSum`. Eixos 2D usam um segmento por
  thread com acumulador `double`, reproduzindo a ordem CPU dentro do segmento.
- `cublasSdot` implementa dot vetorial. Como a API retorna `float`, a copia do
  escalar faz parte do tempo da operacao GPU.
- `cublasSgemv` usa `CUBLAS_OP_T`: os bytes de `A_row[M,K]` representam
  `A_col[K,M]`, logo a transposicao logica calcula `A_row * x` sem materializar A.
- O workspace CUB serializado existente passou a ser compartilhado entre soma
  global e scan. Isso remove alocacao/liberacao repetida sem criar pool geral.
- `launch_1d` calcula a grade sem overflow e valida a conversao para `int`.

Resultados device somente sao marcados validos depois de kernel/cuBLAS e
sincronizacao bem-sucedidos. Tensores CPU nao inicializam CUDA. Nas operacoes
binarias foi mantida a regra existente: se qualquer operando ja estiver no
device, o outro e enviado e a operacao ocorre na GPU.

## Validacao

- PHPT: 2/2.
- SGEMM: 11 grupos.
- elementwise/transpose/reducoes: todos passaram.
- reducoes serial, hierarquica, CUB e cache repetido: passaram.
- sqrt, clip e softmax, incluindo matriz grande: passaram.
- coerencia/lifecycle: 7 grupos.
- operacoes desta rodada: 10 grupos, incluindo 65.537 elementos nao multiplos
  de bloco, NaN, infinitos, zeros com sinal, vazios e falhas de shape.
- cadeias `greater->cumsum`, `broadcast->sqrt->softmax` e
  `matvec->softmax`: permaneceram no device sem D2H intermediario no log debug.
- build CPU-only com ASan/UBSan e build CUDA com invariantes: passaram.
- Compute Sanitizer no WSL falhou antes de instrumentar o PHP porque nao
  encontrou a biblioteca de injecao `libsanitizer-collection.so`; nenhum
  resultado de memcheck foi inferido dessa tentativa.

## Benchmark validado

RTX 3060 12 GiB, CUDA 12.0, driver 576.02, PHP 8.4.16, 2 warmups e 7
repeticoes. Medianas em ms; p25, p75 e amostras completas permanecem no JSON.

| Operacao | Tamanho | CPU | H2D | GPU | D2H | End-to-end | Speedup residente | Speedup E2E |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| greater | 1.048.576 | 1,668 | 1,397 | 3,396 | 0,962 | 5,360 | 0,49x | 0,31x |
| broadcast | 1024x1024 <- 1x1024 | 22,510 | 1,542 | 3,197 | 0,892 | 9,582 | 7,04x | 2,35x |
| tile | 1024x1024 x2 | 8,951 | 1,443 | 12,073 | 1,925 | 16,261 | 0,74x | 0,55x |
| cumsum | 1.048.576 | 2,566 | 1,538 | 4,994 | 1,330 | 8,580 | 0,51x | 0,30x |
| dot | 1.048.576 | 1,699 | 3,175 | 1,122 | 0,000 | 4,658 | 1,51x | 0,36x |
| matvec | 2048x2048 @ 2048 | 6,640 | 4,377 | 1,624 | 0,093 | 7,515 | 4,09x | 0,88x |

A cadeia residente `greater->cumsum` levou 9,961 ms. O benchmark nao mascara
resultados ruins: greater, tile e cumsum isolados perderam para CPU neste shape;
broadcast venceu inclusive end-to-end; dot e matvec venceram residentes, mas nao
amortizaram totalmente as transferencias.

Antes de compartilhar o cache CUB com o scan, a mediana residente do cumsum era
6,086 ms; depois ficou em 4,994 ms, melhora de aproximadamente 17,9%. O ganho nao
foi suficiente para superar a CPU nesse shape, portanto o gargalo residual
permanece registrado.

O dot de 1M elementos apresentou diferenca relativa de aproximadamente 3,7e-4
entre a soma sequencial float da CPU e a arvore cuBLAS. O limite de 5e-4 foi
registrado explicitamente; NaN e infinitos sao verificados semanticamente.

Memoria ao fim: PHP 70 MiB, pico PHP 102 MiB e RSS 390,8 MiB. O hash do binario
medido foi `c7eb949230a6a4ee52ad15e609e309c71c36f1d2798403b9f4c5901a88e90f6f`.

## Pendencias

Equal/notEqual/greaterEqual/less/lessEqual nao foram implementados porque nao
existem na API publica atual. Permanecem CPU-only comparacoes adicionais que
venham a ser adicionadas futuramente, sort/argsort, estatisticas adicionais e
varios helpers de ML/autograd. Tile, greater e cumsum mostram overhead relevante
de alocacao/sincronizacao; pool, streams, executor e fusao continuam adiados ate
profiling Nsight analisavel.
