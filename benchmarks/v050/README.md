# ZMatrix 0.5.0 performance investigation

All GPU scenarios call `toGpu()` explicitly. The benchmark never controls backend dispatch and rejects a measurement when its result differs from the CPU reference.

Quick baseline:

```bash
php benchmarks/v050/benchmark_matrix.php --suite=quick --repetitions=5
```

Full shape/crossover matrix:

```bash
php benchmarks/v050/benchmark_matrix.php --suite=full --repetitions=9
```

Full matrix plus isolated serial/hierarchical/CUB reduction variants:

```bash
bash benchmarks/v050/run_matrix.sh
```

The command writes both JSON (including raw samples and environment) and flat CSV. Median is the primary statistic; mean, standard deviation and p05/p25/p75/p95/p99 remain available.

Profiles:

```bash
bash benchmarks/v050/run_profiles.sh
```

The profile target performs warmups, uses explicit device residency and consumes every result through a validated checksum. Nsight Systems must be run before Nsight Compute. On environments where WSL profiling is unsupported, keep the command, tool output and failure reason as part of the report rather than substituting estimated metrics.

## Delivered artifacts

1. `run_profiles.sh`, `profile_target.php` and `profiles/`: reproducible commands, versions, logs, exit codes and compressed raw Nsight Systems trace.
2. `results/baseline_*.json` and `results/baseline_*.csv`: raw samples, complete statistics, validation and memory observations.
3. `results/PERFORMANCE_DECISION_MAP.md`: statistical report and crossovers.
4. Serial, hierarchical, CUB and hybrid result sets plus `tests/cuda/reduction_strategies.php`: reduction study.
5. `results/decision_map.json` and `results/decision_map.csv`: machine-readable CPU/GPU guidance.
6. The final report ranking: optimization order based only on measured results.

The environment used for the checked-in corpus is RTX 3060 12 GiB, driver 576.02, WSL2, CUDA 12.0 and PHP 8.4.16. The full corpus uses three warmups and seven repetitions. Raw samples are retained; median is primary and mean, standard deviation, p05, p25, p75, p95 and p99 are also reported.
