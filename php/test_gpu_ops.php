<?php
// test_gpu_ops.php
// Ajuste o tamanho para ficar >= 200000 elementos
$rows = 1024;
$cols = 1024;
$n = $rows * $cols;
$warmup = 2;
$iters = 5;
$do_cpu_compare = true;

echo "N = {$n}\n";
echo "warmup={$warmup} iters={$iters}\n";

use ZMatrix\ZTensor;

// cria dois tensores grandes
$a = ZTensor::random([$rows, $cols], -1.0, 1.0);
$b = ZTensor::random([$rows, $cols], -1.0, 1.0);
$baseA = ZTensor::arr($a);
$baseB = ZTensor::arr($b);

function bench($label, $fn, $iters) {
    $times = [];
    for ($i = 0; $i < $iters; $i++) {
        $t0 = microtime(true);
        $fn();
        $t1 = microtime(true);
        $times[] = ($t1 - $t0) * 1000.0;
    }
    $sum = array_sum($times);
    $avg = $sum / $iters;
    $min = min($times);
    $max = max($times);
    return [$avg, $min, $max];
}

function print_line($label, $avg, $min, $max) {
    echo sprintf("%-20s avg %8.2f ms | min %8.2f | max %8.2f\n", $label, $avg, $min, $max);
}

function print_compare($label, $gpu_avg, $cpu_avg, $gpu_min, $gpu_max, $cpu_min, $cpu_max) {
    $speedup = ($gpu_avg > 0.0) ? ($cpu_avg / $gpu_avg) : 0.0;
    echo sprintf(
        "%-20s gpu %8.2f ms | cpu %8.2f ms | speedup %5.2fx\n",
        $label, $gpu_avg, $cpu_avg, $speedup
    );
    echo sprintf(
        "%-20s min/max gpu %6.2f/%6.2f | cpu %6.2f/%6.2f\n",
        "", $gpu_min, $gpu_max, $cpu_min, $cpu_max
    );
}

function set_force_cpu($on) {
    if ($on) {
        putenv("ZMATRIX_FORCE_CPU=1");
    } else {
        putenv("ZMATRIX_FORCE_CPU=0");
    }
}

// warmup
for ($i = 0; $i < $warmup; $i++) {
    $a->add($b);
    $a->sub($b);
    $a->mul($b);
    $a->relu();
    $a->leakyRelu(0.01);
    $a->sigmoid();
    $a->tanh();
    $a->exp();
    $a->abs();
    $a->add(0.5);
    $a->sub(0.25);
    $a->mul(1.5);
    $a->scalarDivide(1.1);
}

// elementwise
[$avg, $min, $max] = bench("add", function() use ($a, $b) { $a->add($b); }, $iters);
print_line("add", $avg, $min, $max);
[$avg, $min, $max] = bench("sub", function() use ($a, $b) { $a->sub($b); }, $iters);
print_line("sub", $avg, $min, $max);
[$avg, $min, $max] = bench("mul", function() use ($a, $b) { $a->mul($b); }, $iters);
print_line("mul", $avg, $min, $max);

// unarias
[$avg, $min, $max] = bench("relu", function() use ($a) { $a->relu(); }, $iters);
print_line("relu", $avg, $min, $max);
[$avg, $min, $max] = bench("leakyRelu", function() use ($a) { $a->leakyRelu(0.01); }, $iters);
print_line("leakyRelu", $avg, $min, $max);
[$avg, $min, $max] = bench("sigmoid", function() use ($a) { $a->sigmoid(); }, $iters);
print_line("sigmoid", $avg, $min, $max);
[$avg, $min, $max] = bench("tanh", function() use ($a) { $a->tanh(); }, $iters);
print_line("tanh", $avg, $min, $max);
[$avg, $min, $max] = bench("exp", function() use ($a) { $a->exp(); }, $iters);
print_line("exp", $avg, $min, $max);
[$avg, $min, $max] = bench("abs", function() use ($a) { $a->abs(); }, $iters);
print_line("abs", $avg, $min, $max);

// escalares
[$avg, $min, $max] = bench("scalar add", function() use ($a) { $a->add(0.5); }, $iters);
print_line("scalar add", $avg, $min, $max);
[$avg, $min, $max] = bench("scalar sub", function() use ($a) { $a->sub(0.25); }, $iters);
print_line("scalar sub", $avg, $min, $max);
[$avg, $min, $max] = bench("scalar mul", function() use ($a) { $a->mul(1.5); }, $iters);
print_line("scalar mul", $avg, $min, $max);
[$avg, $min, $max] = bench("scalar div", function() use ($a) { $a->scalarDivide(1.1); }, $iters);
print_line("scalar div", $avg, $min, $max);

if ($do_cpu_compare) {
    echo "\nGPU vs CPU compare (fresh copies)\n";

    // elementwise
    set_force_cpu(false);
    $a_gpu = ZTensor::arr($baseA);
    $b_gpu = ZTensor::arr($baseB);
    [$g_avg, $g_min, $g_max] = bench("add", function() use ($a_gpu, $b_gpu) { $a_gpu->add($b_gpu); }, $iters);
    set_force_cpu(true);
    $a_cpu = ZTensor::arr($baseA); $b_cpu = ZTensor::arr($baseB);
    [$c_avg, $c_min, $c_max] = bench("add", function() use ($a_cpu, $b_cpu) { $a_cpu->add($b_cpu); }, $iters);
    print_compare("add", $g_avg, $c_avg, $g_min, $g_max, $c_min, $c_max);

    set_force_cpu(false);
    $a_gpu = ZTensor::arr($baseA); $b_gpu = ZTensor::arr($baseB);
    [$g_avg, $g_min, $g_max] = bench("sub", function() use ($a_gpu, $b_gpu) { $a_gpu->sub($b_gpu); }, $iters);
    set_force_cpu(true);
    $a_cpu = ZTensor::arr($baseA); $b_cpu = ZTensor::arr($baseB);
    [$c_avg, $c_min, $c_max] = bench("sub", function() use ($a_cpu, $b_cpu) { $a_cpu->sub($b_cpu); }, $iters);
    print_compare("sub", $g_avg, $c_avg, $g_min, $g_max, $c_min, $c_max);

    set_force_cpu(false);
    $a_gpu = ZTensor::arr($baseA); $b_gpu = ZTensor::arr($baseB);
    [$g_avg, $g_min, $g_max] = bench("mul", function() use ($a_gpu, $b_gpu) { $a_gpu->mul($b_gpu); }, $iters);
    set_force_cpu(true);
    $a_cpu = ZTensor::arr($baseA); $b_cpu = ZTensor::arr($baseB);
    [$c_avg, $c_min, $c_max] = bench("mul", function() use ($a_cpu, $b_cpu) { $a_cpu->mul($b_cpu); }, $iters);
    print_compare("mul", $g_avg, $c_avg, $g_min, $g_max, $c_min, $c_max);

    // unarias
    set_force_cpu(false);
    $a_gpu = ZTensor::arr($baseA);
    [$g_avg, $g_min, $g_max] = bench("relu", function() use ($a_gpu) { $a_gpu->relu(); }, $iters);
    set_force_cpu(true);
    $a_cpu = ZTensor::arr($baseA);
    [$c_avg, $c_min, $c_max] = bench("relu", function() use ($a_cpu) { $a_cpu->relu(); }, $iters);
    print_compare("relu", $g_avg, $c_avg, $g_min, $g_max, $c_min, $c_max);

    set_force_cpu(false);
    $a_gpu = ZTensor::arr($baseA);
    [$g_avg, $g_min, $g_max] = bench("leakyRelu", function() use ($a_gpu) { $a_gpu->leakyRelu(0.01); }, $iters);
    set_force_cpu(true);
    $a_cpu = ZTensor::arr($baseA);
    [$c_avg, $c_min, $c_max] = bench("leakyRelu", function() use ($a_cpu) { $a_cpu->leakyRelu(0.01); }, $iters);
    print_compare("leakyRelu", $g_avg, $c_avg, $g_min, $g_max, $c_min, $c_max);

    set_force_cpu(false);
    $a_gpu = ZTensor::arr($baseA);
    [$g_avg, $g_min, $g_max] = bench("sigmoid", function() use ($a_gpu) { $a_gpu->sigmoid(); }, $iters);
    set_force_cpu(true);
    $a_cpu = ZTensor::arr($baseA);
    [$c_avg, $c_min, $c_max] = bench("sigmoid", function() use ($a_cpu) { $a_cpu->sigmoid(); }, $iters);
    print_compare("sigmoid", $g_avg, $c_avg, $g_min, $g_max, $c_min, $c_max);

    set_force_cpu(false);
    $a_gpu = ZTensor::arr($baseA);
    [$g_avg, $g_min, $g_max] = bench("tanh", function() use ($a_gpu) { $a_gpu->tanh(); }, $iters);
    set_force_cpu(true);
    $a_cpu = ZTensor::arr($baseA);
    [$c_avg, $c_min, $c_max] = bench("tanh", function() use ($a_cpu) { $a_cpu->tanh(); }, $iters);
    print_compare("tanh", $g_avg, $c_avg, $g_min, $g_max, $c_min, $c_max);

    set_force_cpu(false);
    $a_gpu = ZTensor::arr($baseA);
    [$g_avg, $g_min, $g_max] = bench("exp", function() use ($a_gpu) { $a_gpu->exp(); }, $iters);
    set_force_cpu(true);
    $a_cpu = ZTensor::arr($baseA);
    [$c_avg, $c_min, $c_max] = bench("exp", function() use ($a_cpu) { $a_cpu->exp(); }, $iters);
    print_compare("exp", $g_avg, $c_avg, $g_min, $g_max, $c_min, $c_max);

    set_force_cpu(false);
    $a_gpu = ZTensor::arr($baseA);
    [$g_avg, $g_min, $g_max] = bench("abs", function() use ($a_gpu) { $a_gpu->abs(); }, $iters);
    set_force_cpu(true);
    $a_cpu = ZTensor::arr($baseA);
    [$c_avg, $c_min, $c_max] = bench("abs", function() use ($a_cpu) { $a_cpu->abs(); }, $iters);
    print_compare("abs", $g_avg, $c_avg, $g_min, $g_max, $c_min, $c_max);

    // escalares
    set_force_cpu(false);
    $a_gpu = ZTensor::arr($baseA);
    [$g_avg, $g_min, $g_max] = bench("scalar add", function() use ($a_gpu) { $a_gpu->add(0.5); }, $iters);
    set_force_cpu(true);
    $a_cpu = ZTensor::arr($baseA);
    [$c_avg, $c_min, $c_max] = bench("scalar add", function() use ($a_cpu) { $a_cpu->add(0.5); }, $iters);
    print_compare("scalar add", $g_avg, $c_avg, $g_min, $g_max, $c_min, $c_max);

    set_force_cpu(false);
    $a_gpu = ZTensor::arr($baseA);
    [$g_avg, $g_min, $g_max] = bench("scalar sub", function() use ($a_gpu) { $a_gpu->sub(0.25); }, $iters);
    set_force_cpu(true);
    $a_cpu = ZTensor::arr($baseA);
    [$c_avg, $c_min, $c_max] = bench("scalar sub", function() use ($a_cpu) { $a_cpu->sub(0.25); }, $iters);
    print_compare("scalar sub", $g_avg, $c_avg, $g_min, $g_max, $c_min, $c_max);

    set_force_cpu(false);
    $a_gpu = ZTensor::arr($baseA);
    [$g_avg, $g_min, $g_max] = bench("scalar mul", function() use ($a_gpu) { $a_gpu->mul(1.5); }, $iters);
    set_force_cpu(true);
    $a_cpu = ZTensor::arr($baseA);
    [$c_avg, $c_min, $c_max] = bench("scalar mul", function() use ($a_cpu) { $a_cpu->mul(1.5); }, $iters);
    print_compare("scalar mul", $g_avg, $c_avg, $g_min, $g_max, $c_min, $c_max);

    set_force_cpu(false);
    $a_gpu = ZTensor::arr($baseA);
    [$g_avg, $g_min, $g_max] = bench("scalar div", function() use ($a_gpu) { $a_gpu->scalarDivide(1.1); }, $iters);
    set_force_cpu(true);
    $a_cpu = ZTensor::arr($baseA);
    [$c_avg, $c_min, $c_max] = bench("scalar div", function() use ($a_cpu) { $a_cpu->scalarDivide(1.1); }, $iters);
    print_compare("scalar div", $g_avg, $c_avg, $g_min, $g_max, $c_min, $c_max);

    set_force_cpu(false);
}

echo "done\n";
