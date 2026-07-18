# ZMatrix 0.5.0 — mapa experimental CPU × GPU

Gerado em: 2026-07-18T20:26:26+00:00

Este documento orienta o uso explícito de `toGpu()`; não é uma política de despacho automático.

Ambiente: NVIDIA GeForce RTX 3060, 576.02, 12288 MiB; Linux DESKTOP-0748TMR 6.18.33.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun 18 21:54:43 UTC 2026 x86_64; PHP 8.4.16; ZMatrix 0.4.0-float.
Metodologia: 3 warmups, 7 repetições; mediana como estatística principal; toda medição comparada com CPU.

## SGEMM

| Shape | CPU (ms) | GPU residente (ms) | GPU E2E (ms) | Speedup residente | Speedup E2E | Orientação |
|---|---:|---:|---:|---:|---:|---|
| 512x512_512x512 | 10.023 | 1.229 | 2.099 | 8.16x | 4.77x | GPU compensa incluindo transferências |
| 1024x1024_1024x1024 | 14.483 | 3.088 | 5.752 | 4.69x | 2.52x | GPU compensa incluindo transferências |
| 2048x2048_2048x2048 | 61.416 | 8.908 | 20.828 | 6.89x | 2.95x | GPU compensa incluindo transferências |
| 4096x4096_4096x4096 | 743.704 | 117.760 | 190.249 | 6.32x | 3.91x | GPU compensa incluindo transferências |
| 256x4096_4096x128 | 1.045 | 0.868 | 3.432 | 1.20x | 0.30x | GPU somente com dados residentes |
| 4096x256_256x4096 | 122.703 | 110.168 | 165.453 | 1.11x | 0.74x | GPU somente com dados residentes |
| 4096x128_128x256 | 1.592 | 6.465 | 9.204 | 0.25x | 0.17x | CPU preferível |
| 128x4096_4096x256 | 1.035 | 1.576 | 4.628 | 0.66x | 0.22x | CPU preferível |
| 1x4096_4096x256 | 12.718 | 1.188 | 3.410 | 10.71x | 3.73x | GPU compensa incluindo transferências |
| 4096x1_1x1 | 0.013 | 0.152 | 0.276 | 0.09x | 0.05x | CPU preferível |
| 1x4096_4096x1 | 0.009 | 0.157 | 0.281 | 0.06x | 0.03x | CPU preferível |

### Dispersão SGEMM

| Shape | Cenário | Mediana | Média | Desvio | P05 | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|
| 512x512_512x512 | cpu | 10.023 | 12.757 | 7.008 | 7.023 | 24.723 | 27.834 |
| 512x512_512x512 | h2d | 6.844 | 6.308 | 4.024 | 0.747 | 11.636 | 12.492 |
| 512x512_512x512 | gpu_resident | 1.229 | 1.407 | 0.709 | 0.773 | 2.547 | 2.821 |
| 512x512_512x512 | d2h | 0.317 | 0.320 | 0.034 | 0.285 | 0.375 | 0.390 |
| 512x512_512x512 | gpu_end_to_end | 2.099 | 2.545 | 1.149 | 1.781 | 4.478 | 5.132 |
| 1024x1024_1024x1024 | cpu | 14.483 | 21.606 | 12.917 | 10.984 | 43.282 | 47.652 |
| 1024x1024_1024x1024 | h2d | 2.604 | 5.376 | 6.127 | 2.006 | 15.543 | 19.298 |
| 1024x1024_1024x1024 | gpu_resident | 3.088 | 3.229 | 0.693 | 2.579 | 4.388 | 4.637 |
| 1024x1024_1024x1024 | d2h | 0.907 | 0.911 | 0.093 | 0.802 | 1.051 | 1.061 |
| 1024x1024_1024x1024 | gpu_end_to_end | 5.752 | 6.245 | 1.114 | 5.229 | 8.070 | 8.457 |
| 2048x2048_2048x2048 | cpu | 61.416 | 61.497 | 5.696 | 55.548 | 69.526 | 70.467 |
| 2048x2048_2048x2048 | h2d | 8.809 | 11.198 | 6.344 | 7.617 | 21.523 | 25.631 |
| 2048x2048_2048x2048 | gpu_resident | 8.908 | 10.094 | 2.898 | 8.542 | 14.826 | 16.689 |
| 2048x2048_2048x2048 | d2h | 2.926 | 3.199 | 0.557 | 2.791 | 4.162 | 4.317 |
| 2048x2048_2048x2048 | gpu_end_to_end | 20.828 | 23.263 | 5.326 | 20.594 | 32.114 | 35.388 |
| 4096x4096_4096x4096 | cpu | 743.704 | 742.197 | 112.741 | 590.571 | 904.206 | 938.061 |
| 4096x4096_4096x4096 | h2d | 59.763 | 60.013 | 0.706 | 59.155 | 61.090 | 61.231 |
| 4096x4096_4096x4096 | gpu_resident | 117.760 | 160.701 | 67.507 | 106.179 | 265.248 | 266.728 |
| 4096x4096_4096x4096 | d2h | 15.341 | 18.420 | 11.416 | 11.545 | 37.266 | 44.216 |
| 4096x4096_4096x4096 | gpu_end_to_end | 190.249 | 202.522 | 39.027 | 175.359 | 267.878 | 289.820 |
| 256x4096_4096x128 | cpu | 1.045 | 3.534 | 6.138 | 0.935 | 13.342 | 17.523 |
| 256x4096_4096x128 | h2d | 6.032 | 5.727 | 2.811 | 2.228 | 8.902 | 8.929 |
| 256x4096_4096x128 | gpu_resident | 0.868 | 2.565 | 2.933 | 0.610 | 7.635 | 8.399 |
| 256x4096_4096x128 | d2h | 0.133 | 0.146 | 0.024 | 0.127 | 0.186 | 0.194 |
| 256x4096_4096x128 | gpu_end_to_end | 3.432 | 3.494 | 0.264 | 3.246 | 3.919 | 4.059 |
| 4096x256_256x4096 | cpu | 122.703 | 129.131 | 19.887 | 112.037 | 162.141 | 167.386 |
| 4096x256_256x4096 | h2d | 2.642 | 2.596 | 0.232 | 2.323 | 2.877 | 2.888 |
| 4096x256_256x4096 | gpu_resident | 110.168 | 113.301 | 7.900 | 104.642 | 125.293 | 125.431 |
| 4096x256_256x4096 | d2h | 13.420 | 13.761 | 1.307 | 12.343 | 15.768 | 15.844 |
| 4096x256_256x4096 | gpu_end_to_end | 165.453 | 171.334 | 28.216 | 137.558 | 215.448 | 225.033 |
| 4096x128_128x256 | cpu | 1.592 | 1.761 | 0.328 | 1.521 | 2.301 | 2.334 |
| 4096x128_128x256 | h2d | 1.455 | 2.306 | 2.016 | 1.404 | 5.565 | 6.905 |
| 4096x128_128x256 | gpu_resident | 6.465 | 6.730 | 1.865 | 5.076 | 9.850 | 10.542 |
| 4096x128_128x256 | d2h | 1.632 | 1.864 | 0.481 | 1.615 | 2.658 | 2.957 |
| 4096x128_128x256 | gpu_end_to_end | 9.204 | 10.423 | 3.311 | 7.973 | 16.142 | 17.600 |
| 128x4096_4096x256 | cpu | 1.035 | 5.901 | 11.913 | 0.958 | 24.910 | 33.048 |
| 128x4096_4096x256 | h2d | 2.685 | 2.963 | 0.617 | 2.589 | 3.978 | 4.356 |
| 128x4096_4096x256 | gpu_resident | 1.576 | 1.692 | 0.512 | 1.171 | 2.540 | 2.719 |
| 128x4096_4096x256 | d2h | 0.187 | 0.194 | 0.030 | 0.168 | 0.245 | 0.258 |
| 128x4096_4096x256 | gpu_end_to_end | 4.628 | 4.631 | 0.310 | 4.259 | 5.014 | 5.015 |
| 1x4096_4096x256 | cpu | 12.718 | 22.691 | 22.955 | 0.190 | 53.008 | 55.041 |
| 1x4096_4096x256 | h2d | 3.758 | 6.256 | 6.033 | 1.719 | 16.482 | 19.276 |
| 1x4096_4096x256 | gpu_resident | 1.188 | 1.476 | 0.942 | 0.801 | 2.981 | 3.586 |
| 1x4096_4096x256 | d2h | 0.105 | 0.111 | 0.035 | 0.064 | 0.160 | 0.169 |
| 1x4096_4096x256 | gpu_end_to_end | 3.410 | 5.258 | 3.537 | 3.295 | 11.127 | 13.246 |
| 4096x1_1x1 | cpu | 0.013 | 0.017 | 0.009 | 0.013 | 0.032 | 0.038 |
| 4096x1_1x1 | h2d | 0.070 | 0.138 | 0.165 | 0.068 | 0.403 | 0.515 |
| 4096x1_1x1 | gpu_resident | 0.152 | 0.394 | 0.601 | 0.141 | 1.354 | 1.764 |
| 4096x1_1x1 | d2h | 0.071 | 0.070 | 0.003 | 0.066 | 0.074 | 0.074 |
| 4096x1_1x1 | gpu_end_to_end | 0.276 | 0.294 | 0.036 | 0.265 | 0.352 | 0.354 |
| 1x4096_4096x1 | cpu | 0.009 | 0.010 | 0.003 | 0.009 | 0.016 | 0.018 |
| 1x4096_4096x1 | h2d | 0.071 | 0.138 | 0.161 | 0.067 | 0.396 | 0.504 |
| 1x4096_4096x1 | gpu_resident | 0.157 | 0.201 | 0.103 | 0.143 | 0.376 | 0.435 |
| 1x4096_4096x1 | d2h | 0.060 | 0.062 | 0.004 | 0.060 | 0.069 | 0.070 |
| 1x4096_4096x1 | gpu_end_to_end | 0.281 | 0.285 | 0.009 | 0.277 | 0.300 | 0.303 |

## Crossover de cadeias elementwise

| Shape | 1 operação | 2 operações | 5 operações | 10 operações | Mínimo residente | Mínimo E2E |
|---|---:|---:|---:|---:|---:|---:|
| 256x256 | 0.04x / 0.01x | 0.06x / 0.03x | 1.95x / 0.88x | 1.06x / 0.93x | 5 | não medido |
| 512x512 | 0.05x / 0.02x | 0.09x / 0.04x | 3.58x / 1.71x | 2.56x / 2.15x | 5 | 5 |
| 1024x1024 | 0.19x / 0.08x | 0.23x / 0.11x | 2.31x / 1.59x | 2.24x / 1.73x | 5 | 5 |
| 2048x2048 | 0.34x / 0.09x | 0.92x / 0.28x | 9.00x / 4.42x | 6.20x / 4.06x | 5 | 5 |

Cada célula mostra `speedup residente / speedup end-to-end`.

### Dispersão das cadeias

| Shape/cadeia | Cenário | Mediana | Média | Desvio | P05 | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|
| 256x256_chain_1 | cpu | 0.008 | 0.009 | 0.002 | 0.008 | 0.012 | 0.013 |
| 256x256_chain_1 | gpu_resident | 0.189 | 0.257 | 0.161 | 0.186 | 0.518 | 0.624 |
| 256x256_chain_1 | gpu_end_to_end | 0.916 | 0.862 | 0.189 | 0.555 | 0.984 | 0.985 |
| 256x256_chain_2 | cpu | 0.014 | 0.015 | 0.002 | 0.014 | 0.018 | 0.020 |
| 256x256_chain_2 | gpu_resident | 0.249 | 0.325 | 0.176 | 0.248 | 0.608 | 0.726 |
| 256x256_chain_2 | gpu_end_to_end | 0.490 | 0.501 | 0.027 | 0.467 | 0.534 | 0.534 |
| 256x256_chain_5 | cpu | 0.950 | 0.950 | 0.019 | 0.930 | 0.980 | 0.987 |
| 256x256_chain_5 | gpu_resident | 0.488 | 0.599 | 0.214 | 0.463 | 0.973 | 1.061 |
| 256x256_chain_5 | gpu_end_to_end | 1.077 | 1.052 | 0.098 | 0.943 | 1.192 | 1.213 |
| 256x256_chain_10 | cpu | 1.379 | 1.384 | 0.037 | 1.347 | 1.445 | 1.459 |
| 256x256_chain_10 | gpu_resident | 1.306 | 1.259 | 0.167 | 0.990 | 1.431 | 1.435 |
| 256x256_chain_10 | gpu_end_to_end | 1.477 | 1.498 | 0.039 | 1.465 | 1.563 | 1.578 |
| 512x512_chain_1 | cpu | 0.032 | 0.033 | 0.003 | 0.031 | 0.038 | 0.039 |
| 512x512_chain_1 | gpu_resident | 0.668 | 0.673 | 0.161 | 0.510 | 0.933 | 0.982 |
| 512x512_chain_1 | gpu_end_to_end | 1.629 | 1.685 | 0.318 | 1.390 | 2.175 | 2.189 |
| 512x512_chain_2 | cpu | 0.069 | 0.080 | 0.021 | 0.063 | 0.115 | 0.122 |
| 512x512_chain_2 | gpu_resident | 0.732 | 0.797 | 0.156 | 0.681 | 1.056 | 1.139 |
| 512x512_chain_2 | gpu_end_to_end | 1.821 | 1.750 | 0.156 | 1.529 | 1.910 | 1.920 |
| 512x512_chain_5 | cpu | 3.837 | 5.112 | 3.015 | 3.802 | 9.995 | 11.991 |
| 512x512_chain_5 | gpu_resident | 1.072 | 1.231 | 0.273 | 1.004 | 1.671 | 1.688 |
| 512x512_chain_5 | gpu_end_to_end | 2.237 | 2.173 | 0.207 | 1.838 | 2.406 | 2.448 |
| 512x512_chain_10 | cpu | 5.588 | 6.875 | 3.189 | 5.506 | 11.969 | 14.142 |
| 512x512_chain_10 | gpu_resident | 2.185 | 2.157 | 0.307 | 1.670 | 2.486 | 2.517 |
| 512x512_chain_10 | gpu_end_to_end | 2.604 | 2.703 | 0.206 | 2.505 | 3.020 | 3.080 |
| 1024x1024_chain_1 | cpu | 0.412 | 0.426 | 0.047 | 0.387 | 0.506 | 0.527 |
| 1024x1024_chain_1 | gpu_resident | 2.149 | 2.477 | 0.650 | 1.885 | 3.538 | 3.773 |
| 1024x1024_chain_1 | gpu_end_to_end | 5.259 | 5.327 | 0.179 | 5.110 | 5.604 | 5.647 |
| 1024x1024_chain_2 | cpu | 0.702 | 0.729 | 0.070 | 0.683 | 0.845 | 0.886 |
| 1024x1024_chain_2 | gpu_resident | 3.038 | 3.355 | 0.899 | 2.783 | 4.856 | 5.387 |
| 1024x1024_chain_2 | gpu_end_to_end | 6.495 | 6.661 | 0.515 | 6.120 | 7.510 | 7.710 |
| 1024x1024_chain_5 | cpu | 15.833 | 15.949 | 0.195 | 15.733 | 16.210 | 16.236 |
| 1024x1024_chain_5 | gpu_resident | 6.867 | 6.740 | 0.701 | 5.855 | 7.725 | 7.927 |
| 1024x1024_chain_5 | gpu_end_to_end | 9.942 | 9.854 | 0.357 | 9.308 | 10.280 | 10.319 |
| 1024x1024_chain_10 | cpu | 24.519 | 24.508 | 0.351 | 24.009 | 24.983 | 25.010 |
| 1024x1024_chain_10 | gpu_resident | 10.969 | 11.023 | 0.592 | 10.415 | 11.950 | 12.202 |
| 1024x1024_chain_10 | gpu_end_to_end | 14.171 | 14.052 | 0.224 | 13.691 | 14.276 | 14.298 |
| 2048x2048_chain_1 | cpu | 1.686 | 1.677 | 0.044 | 1.618 | 1.739 | 1.756 |
| 2048x2048_chain_1 | gpu_resident | 4.925 | 5.517 | 1.683 | 4.147 | 8.391 | 9.146 |
| 2048x2048_chain_1 | gpu_end_to_end | 18.180 | 19.619 | 3.402 | 17.730 | 25.294 | 27.346 |
| 2048x2048_chain_2 | cpu | 3.681 | 3.753 | 0.262 | 3.499 | 4.193 | 4.283 |
| 2048x2048_chain_2 | gpu_resident | 4.016 | 4.683 | 1.324 | 3.591 | 6.955 | 7.451 |
| 2048x2048_chain_2 | gpu_end_to_end | 13.311 | 13.350 | 0.522 | 12.591 | 14.025 | 14.054 |
| 2048x2048_chain_5 | cpu | 74.173 | 73.409 | 4.971 | 67.102 | 80.662 | 82.567 |
| 2048x2048_chain_5 | gpu_resident | 8.241 | 8.764 | 0.920 | 7.855 | 10.210 | 10.301 |
| 2048x2048_chain_5 | gpu_end_to_end | 16.789 | 17.390 | 1.079 | 16.713 | 19.203 | 19.794 |
| 2048x2048_chain_10 | cpu | 99.280 | 100.213 | 4.295 | 96.259 | 107.329 | 109.355 |
| 2048x2048_chain_10 | gpu_resident | 16.003 | 17.941 | 3.470 | 15.769 | 23.725 | 25.670 |
| 2048x2048_chain_10 | gpu_end_to_end | 24.444 | 22.617 | 4.489 | 15.342 | 25.054 | 25.257 |

## Reduções

| Caso | Serial (ms) | Hierárquica (ms) | CUB (ms) | Híbrida (ms) | Híb./serial | Híb./hierárquica | Validação S/H/C/Híb |
|---|---:|---:|---:|---:|---:|---:|---|
| 1024x1024_sum_axis_all | 26.003 | 1.656 | 0.263 | 0.233 | 111.66x | 7.11x | ok/ok/ok/ok |
| 1024x1024_min_axis_all | 29.736 | 1.843 | 1.848 | 1.868 | 15.92x | 0.99x | ok/ok/ok/ok |
| 1024x1024_max_axis_all | 29.854 | 1.906 | 1.894 | 1.866 | 16.00x | 1.02x | ok/ok/ok/ok |
| 1024x1024_argmin_axis_all | 29.743 | 1.936 | 1.873 | 1.912 | 15.56x | 1.01x | ok/ok/ok/ok |
| 1024x1024_argmax_axis_all | 29.693 | 1.932 | 1.919 | 1.905 | 15.59x | 1.01x | ok/ok/ok/ok |
| 2048x2048_sum_axis_all | 98.926 | 6.084 | 0.300 | 0.299 | n/a | 20.34x | inválido/ok/ok/ok |
| 2048x2048_min_axis_all | 117.946 | 7.021 | 6.941 | 7.079 | 16.66x | 0.99x | ok/ok/ok/ok |
| 2048x2048_max_axis_all | 117.895 | 7.007 | 6.991 | 6.948 | 16.97x | 1.01x | ok/ok/ok/ok |
| 2048x2048_argmin_axis_all | 117.896 | 6.853 | 6.885 | 6.925 | 17.03x | 0.99x | ok/ok/ok/ok |
| 2048x2048_argmax_axis_all | 117.895 | 6.999 | 6.967 | 6.999 | 16.84x | 1.00x | ok/ok/ok/ok |
| 4096x256_sum_axis_0 | 0.314 | 0.249 | 0.362 | 0.246 | 1.28x | 1.01x | ok/ok/ok/ok |
| 4096x256_argmin_axis_0 | 0.301 | 0.189 | 0.353 | 0.251 | 1.20x | 0.75x | ok/ok/ok/ok |
| 4096x256_argmax_axis_0 | 0.343 | 0.247 | 0.277 | 0.258 | 1.33x | 0.96x | ok/ok/ok/ok |
| 256x4096_sum_axis_1 | 0.817 | 0.188 | 0.180 | 0.324 | 2.52x | 0.58x | ok/ok/ok/ok |
| 256x4096_argmin_axis_1 | 0.788 | 0.248 | 0.197 | 0.237 | 3.32x | 1.05x | ok/ok/ok/ok |
| 256x4096_argmax_axis_1 | 2.505 | 0.244 | 0.201 | 0.259 | 9.68x | 0.94x | ok/ok/ok/ok |

A estratégia híbrida usa CUB para soma global e a redução hierárquica própria para min/max/argmin/argmax e eixos. Os primitivos CUB de mínimo/máximo/índice foram rejeitados porque não preservam a semântica pública de NaN na posição zero.

### Validade numérica das somas globais

| Caso | Estratégia | Válida | Erro absoluto | Erro relativo | ULP |
|---|---|---|---:|---:|---:|
| 1024x1024_sum_axis_all | serial | sim | 12.453125 | 2.32868565e-5 | 199 |
| 1024x1024_sum_axis_all | hierarchical | sim | 2.515625 | 4.70412033e-6 | 40 |
| 1024x1024_sum_axis_all | cub | sim | 2.828125 | 5.2884831e-6 | 45 |
| 1024x1024_sum_axis_all | hybrid | sim | 2.828125 | 5.2884831e-6 | 45 |
| 2048x2048_sum_axis_all | serial | não | 3116.9375 | 0.00145726459 | 12468 |
| 2048x2048_sum_axis_all | hierarchical | sim | 198.1875 | 9.26587801e-5 | 793 |
| 2048x2048_sum_axis_all | cub | sim | 198.1875 | 9.26587801e-5 | 793 |
| 2048x2048_sum_axis_all | hybrid | sim | 198.1875 | 9.26587801e-5 | 793 |

NaN inicial, NaN posterior, infinitos e empates de índice são cobertos por `tests/cuda/reduction_strategies.php` para os três caminhos. A soma serial de `2048²` foi mantida no corpus como medição inválida, sem ampliar a tolerância.

## Ranking baseado nas medições

1. Reduções globais: otimização implementada e comparada diretamente com o caminho serial.
2. Sincronização por kernel: cadeias curtas ainda mostram overhead fixo relevante; requer timeline Nsight em ambiente suportado antes de alterar o contrato.
3. Alocação temporária: CUB ainda aloca temporários por chamada; avaliar `cudaMallocAsync`/pool com profiling nativo.
4. Fusão elementwise: considerar apenas para sequências dominantes observadas em workloads reais.

## Limitação de profiling

O Nsight Systems 2022.4.2 executou e validou o alvo e preservou o trace bruto `.qdstrm`, mas o pacote local não contém o importer necessário para gerar/consultar o relatório. O Nsight Compute 2022.4.1 recusou métricas por a GPU estar exposta via WSL. Logs, códigos de saída e comandos foram preservados; portanto não são feitas afirmações sobre ocupação, coalescência ou pressão de registradores nesta máquina.
