<?php

declare(strict_types=1);

function cliOption(string $name, string $default): string
{
    foreach (array_slice($GLOBALS['argv'], 1) as $argument) {
        if (str_starts_with($argument, "--{$name}=")) return substr($argument, strlen($name) + 3);
    }
    return $default;
}

function loadLatest(string $directory, string $strategy, array $requiredCategories = []): array
{
    $files = glob("{$directory}/baseline_full_{$strategy}_*.json") ?: [];
    if (!$files) throw new RuntimeException("missing full result for {$strategy}");
    usort($files, static fn(string $a, string $b): int => filemtime($a) <=> filemtime($b));
    foreach (array_reverse($files) as $file) {
        $data = json_decode((string) file_get_contents($file), true, 512, JSON_THROW_ON_ERROR);
        $present = array_values(array_unique(array_column($data['records'], 'category')));
        if (!array_diff($requiredCategories, $present)) return $data;
    }
    throw new RuntimeException("no {$strategy} result contains categories: " . implode(',', $requiredCategories));
}

function byScenario(array $records, string $category): array
{
    $result = [];
    foreach ($records as $record) {
        if ($record['category'] !== $category) continue;
        $result[$record['case']][$record['scenario']] = $record;
    }
    return $result;
}

function ratio(float $a, float $b): float
{
    return $b > 0.0 ? $a / $b : INF;
}

$input = cliOption('input', __DIR__ . '/results');
$output = cliOption('output', $input);
if (!is_dir($output) && !mkdir($output, 0775, true) && !is_dir($output)) throw new RuntimeException('cannot create output directory');

$hybrid = loadLatest($input, 'hybrid', ['sgemm', 'elementwise_chain']);
$hybridReduction = loadLatest($input, 'hybrid', ['reduction']);
$serial = loadLatest($input, 'serial');
$hierarchical = loadLatest($input, 'hierarchical');
$cub = loadLatest($input, 'cub');
$sgemm = byScenario($hybrid['records'], 'sgemm');
$chains = byScenario($hybrid['records'], 'elementwise_chain');
$reductionSets = [
    'serial' => byScenario($serial['records'], 'reduction'),
    'hierarchical' => byScenario($hierarchical['records'], 'reduction'),
    'cub' => byScenario($cub['records'], 'reduction'),
    'hybrid' => byScenario($hybridReduction['records'], 'reduction'),
];

$decision = ['schema_version' => 1, 'generated_utc' => gmdate(DATE_ATOM), 'sgemm' => [], 'chains' => [], 'reductions' => []];
foreach ($sgemm as $case => $scenarios) {
    $cpu = $scenarios['cpu']['timing']['median_ms'];
    $resident = $scenarios['gpu_resident']['timing']['median_ms'];
    $end = $scenarios['gpu_end_to_end']['timing']['median_ms'];
    $decision['sgemm'][] = [
        'case' => $case,
        'cpu_ms' => $cpu,
        'gpu_resident_ms' => $resident,
        'gpu_end_to_end_ms' => $end,
        'resident_speedup' => ratio($cpu, $resident),
        'end_to_end_speedup' => ratio($cpu, $end),
        'guidance' => $end < $cpu ? 'GPU compensa incluindo transferências' : ($resident < $cpu ? 'GPU somente com dados residentes' : 'CPU preferível'),
    ];
}

$chainByShape = [];
foreach ($chains as $case => $scenarios) {
    $shape = implode('x', $scenarios['cpu']['shape']);
    $length = $scenarios['cpu']['chain_length'];
    $cpu = $scenarios['cpu']['timing']['median_ms'];
    $resident = $scenarios['gpu_resident']['timing']['median_ms'];
    $end = $scenarios['gpu_end_to_end']['timing']['median_ms'];
    $entry = [
        'shape' => $shape, 'chain_length' => $length, 'cpu_ms' => $cpu,
        'gpu_resident_ms' => $resident, 'gpu_end_to_end_ms' => $end,
        'resident_speedup' => ratio($cpu, $resident), 'end_to_end_speedup' => ratio($cpu, $end),
    ];
    $decision['chains'][] = $entry;
    $chainByShape[$shape][] = $entry;
}

foreach ($reductionSets['serial'] as $case => $unused) {
    $entry = ['case' => $case];
    foreach ($reductionSets as $strategy => $set) {
        $scenario = 'gpu_' . $strategy;
        $record = $set[$case][$scenario] ?? null;
        if (!$record) continue;
        $entry[$strategy . '_ms'] = $record['timing']['median_ms'];
        $entry[$strategy . '_max_abs_error'] = $record['validation']['max_abs_error'];
        $entry[$strategy . '_max_relative_error'] = $record['validation']['max_relative_error'];
        $entry[$strategy . '_max_ulp_error'] = $record['validation']['max_ulp_error'];
        $entry[$strategy . '_valid'] = $record['validation']['valid'];
    }
    if (($entry['serial_valid'] ?? false) && isset($entry['serial_ms'], $entry['hybrid_ms'])) {
        $entry['hybrid_speedup_vs_serial'] = ratio($entry['serial_ms'], $entry['hybrid_ms']);
    }
    if (($entry['hierarchical_valid'] ?? false) && isset($entry['hierarchical_ms'], $entry['hybrid_ms'])) {
        $entry['hybrid_speedup_vs_hierarchical'] = ratio($entry['hierarchical_ms'], $entry['hybrid_ms']);
    }
    $decision['reductions'][] = $entry;
}

file_put_contents("{$output}/decision_map.json", json_encode($decision, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR) . PHP_EOL);
$csv = fopen("{$output}/decision_map.csv", 'wb');
fputcsv($csv, ['category', 'case_or_shape', 'chain_length', 'cpu_ms', 'gpu_resident_ms', 'gpu_end_to_end_ms', 'resident_speedup', 'end_to_end_speedup', 'guidance'], ',', '"', '');
foreach ($decision['sgemm'] as $row) fputcsv($csv, ['sgemm', $row['case'], '', $row['cpu_ms'], $row['gpu_resident_ms'], $row['gpu_end_to_end_ms'], $row['resident_speedup'], $row['end_to_end_speedup'], $row['guidance']], ',', '"', '');
foreach ($decision['chains'] as $row) fputcsv($csv, ['chain', $row['shape'], $row['chain_length'], $row['cpu_ms'], $row['gpu_resident_ms'], $row['gpu_end_to_end_ms'], $row['resident_speedup'], $row['end_to_end_speedup'], ''], ',', '"', '');
fclose($csv);

$lines = [
    '# ZMatrix 0.5.0 — mapa experimental CPU × GPU', '',
    'Gerado em: ' . gmdate(DATE_ATOM), '',
    'Este documento orienta o uso explícito de `toGpu()`; não é uma política de despacho automático.', '',
    sprintf('Ambiente: %s; %s; PHP %s; ZMatrix %s.', $hybrid['environment']['gpu'], $hybrid['environment']['os'], $hybrid['environment']['php'], $hybrid['environment']['zmatrix']),
    sprintf('Metodologia: %d warmups, %d repetições; mediana como estatística principal; toda medição comparada com CPU.', $hybrid['methodology']['warmups'], $hybrid['methodology']['repetitions']), '',
    '## SGEMM', '',
    '| Shape | CPU (ms) | GPU residente (ms) | GPU E2E (ms) | Speedup residente | Speedup E2E | Orientação |',
    '|---|---:|---:|---:|---:|---:|---|',
];
foreach ($decision['sgemm'] as $row) {
    $lines[] = sprintf('| %s | %.3f | %.3f | %.3f | %.2fx | %.2fx | %s |', $row['case'], $row['cpu_ms'], $row['gpu_resident_ms'], $row['gpu_end_to_end_ms'], $row['resident_speedup'], $row['end_to_end_speedup'], $row['guidance']);
}
$lines[] = '';
$lines[] = '### Dispersão SGEMM';
$lines[] = '';
$lines[] = '| Shape | Cenário | Mediana | Média | Desvio | P05 | P95 | P99 |';
$lines[] = '|---|---|---:|---:|---:|---:|---:|---:|';
foreach ($sgemm as $case => $scenarios) {
    foreach ($scenarios as $scenario => $record) {
        $t = $record['timing'];
        $lines[] = sprintf('| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |', $case, $scenario, $t['median_ms'], $t['mean_ms'], $t['stddev_ms'], $t['p05_ms'], $t['p95_ms'], $t['p99_ms']);
    }
}
$lines[] = '';
$lines[] = '## Crossover de cadeias elementwise';
$lines[] = '';
$lines[] = '| Shape | 1 operação | 2 operações | 5 operações | 10 operações | Mínimo residente | Mínimo E2E |';
$lines[] = '|---|---:|---:|---:|---:|---:|---:|';
foreach ($chainByShape as $shape => $rows) {
    usort($rows, static fn(array $a, array $b): int => $a['chain_length'] <=> $b['chain_length']);
    $cells = [];
    $minimumResident = null;
    $minimumEnd = null;
    foreach ($rows as $row) {
        $cells[$row['chain_length']] = sprintf('%.2fx / %.2fx', $row['resident_speedup'], $row['end_to_end_speedup']);
        if ($minimumResident === null && $row['resident_speedup'] > 1.0) $minimumResident = $row['chain_length'];
        if ($minimumEnd === null && $row['end_to_end_speedup'] > 1.0) $minimumEnd = $row['chain_length'];
    }
    $lines[] = sprintf('| %s | %s | %s | %s | %s | %s | %s |', $shape, $cells[1] ?? '-', $cells[2] ?? '-', $cells[5] ?? '-', $cells[10] ?? '-', $minimumResident ?? 'não medido', $minimumEnd ?? 'não medido');
}
$lines[] = '';
$lines[] = 'Cada célula mostra `speedup residente / speedup end-to-end`.';
$lines[] = '';
$lines[] = '### Dispersão das cadeias';
$lines[] = '';
$lines[] = '| Shape/cadeia | Cenário | Mediana | Média | Desvio | P05 | P95 | P99 |';
$lines[] = '|---|---|---:|---:|---:|---:|---:|---:|';
foreach ($chains as $case => $scenarios) {
    foreach ($scenarios as $scenario => $record) {
        $t = $record['timing'];
        $lines[] = sprintf('| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |', $case, $scenario, $t['median_ms'], $t['mean_ms'], $t['stddev_ms'], $t['p05_ms'], $t['p95_ms'], $t['p99_ms']);
    }
}
$lines[] = '';
$lines[] = '## Reduções';
$lines[] = '';
$lines[] = '| Caso | Serial (ms) | Hierárquica (ms) | CUB (ms) | Híbrida (ms) | Híb./serial | Híb./hierárquica | Validação S/H/C/Híb |';
$lines[] = '|---|---:|---:|---:|---:|---:|---:|---|';
foreach ($decision['reductions'] as $row) {
    $validity = implode('/', array_map(static fn(string $key): string => ($row[$key . '_valid'] ?? false) ? 'ok' : 'inválido', ['serial', 'hierarchical', 'cub', 'hybrid']));
    $serialSpeedup = isset($row['hybrid_speedup_vs_serial']) ? sprintf('%.2fx', $row['hybrid_speedup_vs_serial']) : 'n/a';
    $hierarchicalSpeedup = isset($row['hybrid_speedup_vs_hierarchical']) ? sprintf('%.2fx', $row['hybrid_speedup_vs_hierarchical']) : 'n/a';
    $lines[] = sprintf('| %s | %.3f | %.3f | %.3f | %.3f | %s | %s | %s |', $row['case'], $row['serial_ms'] ?? NAN, $row['hierarchical_ms'] ?? NAN, $row['cub_ms'] ?? NAN, $row['hybrid_ms'] ?? NAN, $serialSpeedup, $hierarchicalSpeedup, $validity);
}
$lines[] = '';
$lines[] = 'A estratégia híbrida usa CUB para soma global e a redução hierárquica própria para min/max/argmin/argmax e eixos. Os primitivos CUB de mínimo/máximo/índice foram rejeitados porque não preservam a semântica pública de NaN na posição zero.';
$lines[] = '';
$lines[] = '### Validade numérica das somas globais';
$lines[] = '';
$lines[] = '| Caso | Estratégia | Válida | Erro absoluto | Erro relativo | ULP |';
$lines[] = '|---|---|---|---:|---:|---:|';
foreach ($decision['reductions'] as $row) {
    if (!str_contains($row['case'], '_sum_axis_all')) continue;
    foreach (['serial', 'hierarchical', 'cub', 'hybrid'] as $strategy) {
        $lines[] = sprintf('| %s | %s | %s | %.9g | %.9g | %s |', $row['case'], $strategy,
            ($row[$strategy . '_valid'] ?? false) ? 'sim' : 'não',
            $row[$strategy . '_max_abs_error'] ?? NAN, $row[$strategy . '_max_relative_error'] ?? NAN,
            $row[$strategy . '_max_ulp_error'] === null ? 'n/a' : (string) $row[$strategy . '_max_ulp_error']);
    }
}
$lines[] = '';
$lines[] = 'NaN inicial, NaN posterior, infinitos e empates de índice são cobertos por `tests/cuda/reduction_strategies.php` para os três caminhos. A soma serial de `2048²` foi mantida no corpus como medição inválida, sem ampliar a tolerância.';
$lines[] = '';
$lines[] = '## Ranking baseado nas medições';
$lines[] = '';
$lines[] = '1. Reduções globais: otimização implementada e comparada diretamente com o caminho serial.';
$lines[] = '2. Sincronização por kernel: cadeias curtas ainda mostram overhead fixo relevante; requer timeline Nsight em ambiente suportado antes de alterar o contrato.';
$lines[] = '3. Alocação temporária: CUB ainda aloca temporários por chamada; avaliar `cudaMallocAsync`/pool com profiling nativo.';
$lines[] = '4. Fusão elementwise: considerar apenas para sequências dominantes observadas em workloads reais.';
$lines[] = '';
$lines[] = '## Limitação de profiling';
$lines[] = '';
$lines[] = 'O Nsight Systems 2022.4.2 executou e validou o alvo e preservou o trace bruto `.qdstrm`, mas o pacote local não contém o importer necessário para gerar/consultar o relatório. O Nsight Compute 2022.4.1 recusou métricas por a GPU estar exposta via WSL. Logs, códigos de saída e comandos foram preservados; portanto não são feitas afirmações sobre ocupação, coalescência ou pressão de registradores nesta máquina.';
file_put_contents("{$output}/PERFORMANCE_DECISION_MAP.md", implode(PHP_EOL, $lines) . PHP_EOL);

echo json_encode(['decision_json' => "{$output}/decision_map.json", 'decision_csv' => "{$output}/decision_map.csv", 'report' => "{$output}/PERFORMANCE_DECISION_MAP.md"], JSON_PRETTY_PRINT), PHP_EOL;
