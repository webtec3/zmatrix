# ZMatrix v0.5.0-float — Release Candidate

Data da revisão candidata: 2026-07-18  
Status: tecnicamente concluída, ainda não publicada  
Tag `v0.5.0`: não criada

## Resumo

A versão 0.5.0 torna o backend CUDA matematicamente auditável, preserva a coerência host/device e adiciona uma estratégia de reduções baseada em medições. A CPU continua sendo o backend padrão; CUDA somente é usada após `toGpu()` explícito.

Não há alteração incompatível na API pública.

## Mudanças do backend

- SGEMM cuBLAS row-major corrigido e validado para matrizes quadradas, retangulares, linha e coluna;
- resultado SGEMM permanece residente e não realiza H2D de C com `beta = 0`;
- máquina de estados host/device centralizada e auditada;
- erros CUDA/cuBLAS propagados como exceções contextualizadas;
- resolução do driver WSL prioriza `/usr/lib/wsl/lib/libcuda.so.1`;
- kernels CUDA adicionados para divide, pow, log, fill, transpose e reduções;
- transpose tiled `32x32` com shared memory;
- estratégia híbrida de redução:
  - CUB para soma global;
  - kernel hierárquico próprio para min, max, argmin, argmax e eixos;
  - semântica CPU preservada para NaN, infinitos e primeiro índice em empates;
- build e execução CPU-only preservados.

## Validação

- PHPT cobre os 60 métodos públicos;
- suítes específicas cobrem SGEMM, elementwise, transpose e reduções;
- testes de coerência cobrem CPU/H2D/kernel/D2H, cópia e falhas;
- ASan, UBSan e Valgrind foram executados no caminho CPU-only durante a auditoria;
- benchmarks rejeitam medições sem comparação com a referência CPU;
- diferenças numéricas registram erro absoluto, relativo e ULP quando aplicável.

## Resultados no hardware de referência

Ambiente: RTX 3060 12 GiB, driver 576.02, WSL2, CUDA 12.0, PHP 8.4.16.

| Cenário | Resultado observado |
|---|---:|
| SGEMM `4096²`, GPU residente | 6,32× sobre CPU |
| SGEMM `4096²`, end-to-end | 3,91× sobre CPU |
| Crossover elementwise `512²+` | aproximadamente 5 operações |
| Soma global `1024²`, híbrida/serial válida | 111,66× |
| Soma global `2048²`, híbrida/hierárquica válida | 20,35× |

Esses resultados são referências experimentais específicas desse hardware e não representam garantia geral de desempenho.

## Integridade e reprodução

- metodologia: `benchmarks/v050/README.md`;
- dados brutos: `benchmarks/v050/results/baseline_*.json` e `.csv`;
- relatório: `benchmarks/v050/results/PERFORMANCE_DECISION_MAP.md`;
- mapa estruturado: `decision_map.json` e `decision_map.csv`;
- trace: `benchmarks/v050/profiles/zmatrix_systems.qdstrm.gz`.

O corpus completo registra `0.4.0-float` porque foi coletado antes do bump administrativo. Os dados não foram alterados; o backend funcional medido corresponde ao RC. A revisão candidata é novamente compilada e submetida a um benchmark mínimo após o bump.

## Limitações conhecidas

- o trace Nsight Systems foi coletado, porém não importado por ausência do importer compatível;
- Nsight Compute não oferece profiling para essa configuração WSL;
- não há ainda executor CUDA, grafo de eventos, múltiplas streams ou pool `cudaMallocAsync`;
- sincronizações globais continuam preservando o contrato síncrono PHP;
- operações CUDA ainda não cobrem toda a API CPU.

## Publicação

Este documento descreve um Release Candidate. A tag `v0.5.0` e a release somente podem ser criadas após aprovação explícita das evidências do build limpo.
