# Benchmark CPU/GPU reproduzível

Este runner substitui a medição histórica baseada em `microtime(true)` e médias
de lotes. Ele usa `hrtime(true)` (relógio monotônico), 10 warm-ups, no mínimo 50
amostras independentes, estatísticas por amostra e filtragem de outliers por MAD
(com IQR como fallback quando MAD é zero).

## Execução

```bash
php test_gpu_comparison.php
```

Uma matriz menor pode ser selecionada para verificações rápidas, mas os mínimos
de aquecimento e amostragem continuam obrigatórios:

```bash
php test_gpu_comparison.php \
  --sizes=256,512,1024 \
  --warmups=10 \
  --iterations=50 \
  --json=benchmarks/reliable_gpu/results/local.json \
  --csv=benchmarks/reliable_gpu/results/local.csv
```

Se Xdebug estiver carregado, o wrapper reinicia o benchmark com `php -n`,
carregando somente `zmatrix.so`. A execução fixa uma thread OpenMP/BLAS para
reduzir interferência da política de threads do host.

## Contrato das medições

- `cpu_operation`: exclui o clone da entrada; inclui operação e alocação da saída.
- `gpu_resident`: operandos já estão no device; exclui preparação D2D; inclui
  alocação da saída, kernel/cuBLAS e sincronização.
- `gpu_end_to_end`: inclui alocação device, H2D, operação, sincronização e D2H;
  exclui criação do contexto CUDA, inicialização inicial de handles e clone host.
- `Sum`: por contrato, inclui workspace da redução e a leitura D2H do escalar.

Os kernels expostos ao PHP são síncronos: `CUDA_KERNEL_CHECK` executa
`cudaPeekAtLastError()` e `cudaDeviceSynchronize()`, e SGEMM sincroniza depois de
`cublasSgemm()`. O projeto não possui backends OpenCL ou Vulkan.

Cada operação CPU e GPU recebe clones do mesmo tensor base determinístico. O
resultado é consumido depois da janela cronometrada, impedindo benchmark de uma
operação cujo resultado nunca é observado. Antes da coleta, uma amostra do
resultado GPU é comparada elemento a elemento com CPU; o checksum global também
é preservado como diagnóstico.

## Transpose

`transpose()` em CPU é deliberadamente uma strided view O(1). Portanto um tempo
de poucos microssegundos mede somente a criação da view e é válido, mas não pode
ser comparado com uma transposição física CUDA. O runner publica duas linhas:

- `TransposeView`: custo O(1) de criar a view, com consumo posterior do resultado;
- `TransposePhysicalProxy`: força materialização contígua em CPU por meio de um
  `clip(-INF, INF)` e mede o kernel físico em GPU.

O proxy CPU inclui a passagem de clamp e é explicitamente marcado como não
comparável; ele existe para demonstrar a ordem de grandeza de percorrer e
materializar todos os elementos, sem alterar a API pública.

Consulte [REPORT_20260719.md](REPORT_20260719.md) para a auditoria e os resultados
da rodada completa.
