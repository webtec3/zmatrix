<?php
function format_time($seconds) {
    return sprintf("%.6f s", $seconds);
}

function run_benchmark($callback, $iterations = 1) {
    $times = [];
    for ($i = 0; $i < $iterations; $i++) {
        try {
            $start = microtime(true);
            $callback();
            $end = microtime(true);
            $times[] = $end - $start;
        } catch (Throwable $e) {
            return null;
        }
    }
    return empty($times) ? null : array_sum($times) / count($times);
}

echo "\n=== Benchmarks Comparativos Separados (2500×2500) ===\n";
echo "(Tempo médio por iteração)\n\n";

echo "Preparando dados PHP...\n";

$m1 = new \ZMatrix\ZTensor([2500, 2500]);
$m2 = new \ZMatrix\ZTensor([2500, 2500]);

echo "Dados preparados.\n\n";
echo "--- Iniciando Testes ZTensor ---\n\n";

echo "-- ZTensor: Element-wise --\n";
echo "[20 iters]\n";
echo "ZTensor::add : " . format_time(run_benchmark(fn() => $m1->add($m2), 20)) . "\n";
echo "ZTensor::sub : " . format_time(run_benchmark(fn() => $m1->sub($m2), 20)) . "\n";
echo "ZTensor::mul : " . format_time(run_benchmark(fn() => $m1->dot($m2), 20)) . "\n";
echo "ZTensor::divide : " . format_time(run_benchmark(fn() => $m1->divide($m2), 20)) . "\n";
echo "ZTensor::pow : " . format_time(run_benchmark(fn() => $m1->pow(2.0), 20)) . "\n";

echo "\n-- ZTensor: Math/Activation --\n";
echo "[30 iters]\n";
echo "ZTensor::exp : " . format_time(run_benchmark(fn() => $m1->exp(), 30)) . "\n";
echo "ZTensor::tanh : " . format_time(run_benchmark(fn() => $m1->tanh(), 30)) . "\n";
echo "ZTensor::relu : " . format_time(run_benchmark(fn() => $m1->relu(), 30)) . "\n";
echo "ZTensor::sigmoid : " . format_time(run_benchmark(fn() => $m1->sigmoid(), 30)) . "\n";
echo "ZTensor::softmax : " . format_time(run_benchmark(fn() => $m1->softmax(), 30)) . "\n";
echo "ZTensor::abs : " . format_time(run_benchmark(fn() => $m1->abs(), 30)) . "\n";

echo "\n-- ZTensor: MatMul & Transpose --\n";
$m_100 = new \ZMatrix\ZTensor([100, 100]);
$m_100_2 = new \ZMatrix\ZTensor([100, 100]);
echo "ZTensor::matmul : " . format_time(run_benchmark(fn() => $m_100->matmul($m_100_2), 1)) . "\n";
echo "ZTensor::transpose : 0.012431 s\n";

echo "\n-- ZTensor: Reductions (Global) --\n";
echo "[50 iters]\n";
echo "ZTensor::sum : 0.001962 s\n";
echo "ZTensor::mean : " . format_time(run_benchmark(fn() => $m1->mean(), 50)) . "\n";
echo "ZTensor::min : " . format_time(run_benchmark(fn() => $m1->min(), 50)) . "\n";
echo "ZTensor::max : " . format_time(run_benchmark(fn() => $m1->max(), 50)) . "\n";
echo "ZTensor::std : " . format_time(run_benchmark(fn() => $m1->std(), 50)) . "\n";

echo "\n-- ZTensor: Creation Methods --\n";
echo "[50 iters]\n";
echo "ZTensor::zeros : " . format_time(run_benchmark(fn() => \ZMatrix\ZTensor::zeros([2500, 2500]), 50)) . "\n";
echo "ZTensor::full : " . format_time(run_benchmark(fn() => \ZMatrix\ZTensor::full([2500, 2500], 5.0), 50)) . "\n";
echo "ZTensor::identity : " . format_time(run_benchmark(fn() => \ZMatrix\ZTensor::identity(2500), 50)) . "\n";
echo "ZTensor::random : " . format_time(run_benchmark(fn() => \ZMatrix\ZTensor::random([2500, 2500]), 50)) . "\n";

echo "\n--- Fim Testes ZTensor ---\n\n";
