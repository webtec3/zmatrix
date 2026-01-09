<?php
function format_time($seconds) {
    return sprintf("%.6f s", $seconds);
}

function run_benchmark_precise($callback, $iterations = 1, $inner_loops = 10) {
    $times = [];
    
    for ($i = 0; $i < $iterations; $i++) {
        try {
            $start = microtime(true);
            for ($j = 0; $j < $inner_loops; $j++) {
                $callback();
            }
            $end = microtime(true);
            $times[] = ($end - $start) / $inner_loops;
        } catch (Throwable $e) {
            return null;
        }
    }
    
    if (empty($times)) return null;
    
    sort($times);
    $avg = array_sum($times) / count($times);
    $min = $times[0];
    $max = $times[count($times) - 1];
    $median = $times[intval(count($times) / 2)];
    
    return [
        'avg' => $avg,
        'min' => $min,
        'max' => $max,
        'median' => $median,
        'times' => $times
    ];
}

echo "\n=== Benchmarks Precisos (2500×2500) ===\n";
echo "(20 rodadas × 10 operações por rodada)\n\n";

echo "Preparando dados PHP...\n";
$m1 = new \ZMatrix\ZTensor([2500, 2500]);
$m2 = new \ZMatrix\ZTensor([2500, 2500]);
echo "Dados preparados.\n\n";

echo "--- Element-wise Operations ---\n";

$result = run_benchmark_precise(fn() => $m1->add($m2), 20, 10);
echo "add: AVG=" . format_time($result['avg']) . 
     " | MIN=" . format_time($result['min']) . 
     " | MAX=" . format_time($result['max']) . 
     " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => $m1->sub($m2), 20, 10);
echo "sub: AVG=" . format_time($result['avg']) . 
     " | MIN=" . format_time($result['min']) . 
     " | MAX=" . format_time($result['max']) . 
     " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => $m1->dot($m2), 20, 10);
echo "mul: AVG=" . format_time($result['avg']) . 
     " | MIN=" . format_time($result['min']) . 
     " | MAX=" . format_time($result['max']) . 
     " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => $m1->divide($m2), 20, 10);
echo "divide: AVG=" . format_time($result['avg']) . 
       " | MIN=" . format_time($result['min']) . 
       " | MAX=" . format_time($result['max']) . 
       " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => $m1->pow(2.0), 20, 10);
echo "pow: AVG=" . format_time($result['avg']) . 
     " | MIN=" . format_time($result['min']) . 
     " | MAX=" . format_time($result['max']) . 
     " | MEDIAN=" . format_time($result['median']) . "\n";

echo "\n--- Math/Activation Functions ---\n";

$result = run_benchmark_precise(fn() => $m1->exp(), 20, 10);
echo "exp: AVG=" . format_time($result['avg']) . 
     " | MIN=" . format_time($result['min']) . 
     " | MAX=" . format_time($result['max']) . 
     " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => $m1->tanh(), 20, 10);
echo "tanh: AVG=" . format_time($result['avg']) . 
      " | MIN=" . format_time($result['min']) . 
      " | MAX=" . format_time($result['max']) . 
      " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => $m1->relu(), 20, 10);
echo "relu: AVG=" . format_time($result['avg']) . 
      " | MIN=" . format_time($result['min']) . 
      " | MAX=" . format_time($result['max']) . 
      " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => $m1->sigmoid(), 20, 10);
echo "sigmoid: AVG=" . format_time($result['avg']) . 
         " | MIN=" . format_time($result['min']) . 
         " | MAX=" . format_time($result['max']) . 
         " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => $m1->softmax(), 20, 10);
echo "softmax: AVG=" . format_time($result['avg']) . 
         " | MIN=" . format_time($result['min']) . 
         " | MAX=" . format_time($result['max']) . 
         " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => $m1->abs(), 20, 10);
echo "abs: AVG=" . format_time($result['avg']) . 
     " | MIN=" . format_time($result['min']) . 
     " | MAX=" . format_time($result['max']) . 
     " | MEDIAN=" . format_time($result['median']) . "\n";

echo "\n--- Reductions (Global) ---\n";

$result = run_benchmark_precise(fn() => $m1->mean(), 20, 5);
echo "mean: AVG=" . format_time($result['avg']) . 
      " | MIN=" . format_time($result['min']) . 
      " | MAX=" . format_time($result['max']) . 
      " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => $m1->min(), 20, 5);
echo "min: AVG=" . format_time($result['avg']) . 
     " | MIN=" . format_time($result['min']) . 
     " | MAX=" . format_time($result['max']) . 
     " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => $m1->max(), 20, 5);
echo "max: AVG=" . format_time($result['avg']) . 
     " | MIN=" . format_time($result['min']) . 
     " | MAX=" . format_time($result['max']) . 
     " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => $m1->std(), 20, 5);
echo "std: AVG=" . format_time($result['avg']) . 
     " | MIN=" . format_time($result['min']) . 
     " | MAX=" . format_time($result['max']) . 
     " | MEDIAN=" . format_time($result['median']) . "\n";

echo "\n--- Creation Methods ---\n";

$result = run_benchmark_precise(fn() => \ZMatrix\ZTensor::zeros([2500, 2500]), 10, 1);
echo "zeros: AVG=" . format_time($result['avg']) . 
      " | MIN=" . format_time($result['min']) . 
      " | MAX=" . format_time($result['max']) . 
      " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => \ZMatrix\ZTensor::full([2500, 2500], 5.0), 10, 1);
echo "full: AVG=" . format_time($result['avg']) . 
      " | MIN=" . format_time($result['min']) . 
      " | MAX=" . format_time($result['max']) . 
      " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => \ZMatrix\ZTensor::identity(2500), 10, 1);
echo "identity: AVG=" . format_time($result['avg']) . 
           " | MIN=" . format_time($result['min']) . 
           " | MAX=" . format_time($result['max']) . 
           " | MEDIAN=" . format_time($result['median']) . "\n";

$result = run_benchmark_precise(fn() => \ZMatrix\ZTensor::random([2500, 2500]), 10, 1);
echo "random: AVG=" . format_time($result['avg']) . 
        " | MIN=" . format_time($result['min']) . 
        " | MAX=" . format_time($result['max']) . 
        " | MEDIAN=" . format_time($result['median']) . "\n";

echo "\n--- Fim Testes ---\n\n";
?>
