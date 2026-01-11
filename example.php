<?php

use ZMatrix\ZTensor;

function assertEquals(array $a, array $b): void
{
    if (count($a) !== count($b) || array_diff_recursive($a, $b) !== []) {
        throw new RuntimeException("Arrays differ:\n" . print_r($a, true) . "\nvs\n" . print_r($b, true));
    }
}

function array_diff_recursive(array $a, array $b): array
{
    $diff = [];

    foreach ($a as $k => $v) {
        if (!array_key_exists($k, $b)) {
            $diff[$k] = $v;
        } elseif (is_array($v) || is_array($b[$k])) {
            $d = array_diff_recursive($v, $b[$k]);
            if (!empty($d)) {
                $diff[$k] = $d;
            }
        } elseif ($v != $b[$k]) {
            $diff[$k] = $v;
        }
    }

    return $diff;
}

$tests = [
    'greater elementwise' => function() {
        $a = ZTensor::arr([1, 5, 3, 8, 2]);
        $b = ZTensor::arr([2, 4, 3, 6, 9]);
        $out = $a->greater($b);
        assertEquals([0.0,1.0,0.0,1.0,0.0], $out->toArray());
    },

    'greater shape-mismatch' => function() {
        $a = ZTensor::arr([1,2,3]);
        try {
            $a->greater(ZTensor::arr([[1,2],[3,4]]));
            throw new RuntimeException("Esperava exceção de shape mismatch");       
        } catch (Exception $e) {
            // OK - captura qualquer exceção (RuntimeException ou Exception)
            if (strpos($e->getMessage(), 'mismatch') === false || strpos($e->getMessage(), 'Shape') === false) {
                throw new RuntimeException("Erro esperado diferente: " . $e->getMessage());
            }
        }
    },

    'minimum static' => function() {
        $t    = ZTensor::arr([1,5,3,-2,0]);
        $mini = ZTensor::minimum($t, 2.5);
        assertEquals([1.0,2.5,2.5,-2.0,0.0], $mini->toArray());
    },

    'maximum static' => function() {
        $t    = ZTensor::arr([1,5,3,-2,0]);
        $maxi = ZTensor::maximum($t, 2.5);
        assertEquals([2.5,5.0,3.0,2.5,2.5], $maxi->toArray());
    },

    'chained softmax→sigmoid→relu→greater' => function() {
        $t   = ZTensor::arr([0.1,0.9]);
        $out = $t->softmax()
            ->sigmoid()
            ->relu()
            ->greater(ZTensor::arr([0.2,0.8]));
        $arr = $out->toArray();
        if (count($arr) !== 2) {
            throw new RuntimeException("Resultado deve ter 2 elementos");
        }
        foreach ($arr as $v) {
            if (!in_array($v, [0.0,1.0], true)) {
                throw new RuntimeException("Valor inválido após chain: $v");
            }
        }
    },

    'divide elementwise' => function() {
        $a = ZTensor::arr([[10, 20], [30, 40]]);
        $b = ZTensor::arr([[2, 4], [5, 10]]);
        $a->divide($b);
        assertEquals([[5.0, 5.0], [6.0, 4.0]], $a->toArray());
    },

    'divide by scalar' => function() {
        $a = ZTensor::arr([[6, 9]]);
        $a->divide(3);
        assertEquals([[2.0, 3.0]], $a->toArray());
    },

    'divide by vector broadcast' => function() {
        $a = ZTensor::arr([[10, 20, 30], [40, 50, 60]]);
        $b = ZTensor::arr([10, 10, 10]);
        $a->divide($b);
        assertEquals([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], $a->toArray());
    },

    'divide by singleton tensor (1D)' => function() {
        $a = ZTensor::arr([[5, 10]]);
        $b = ZTensor::arr([5]);
        $a->divide($b);
        assertEquals([[1.0, 2.0]], $a->toArray());
    },
];

echo "Iniciando testes ZTensor:\n\n";
$passed = $failed = 0;

foreach ($tests as $name => $testFn) {
    echo "[ TEST ] $name ... ";
    try {
        $testFn();
        echo "OK\n";
        $passed++;
    } catch (Throwable $e) {
        echo "FAIL\n";
        echo "        ↳ " . get_class($e) . ": " . $e->getMessage() . "\n";
        $failed++;
    }
}

$a = ZTensor::arr([[4,8],[12,24]]);
$b = ZTensor::arr([[5,6],[7,8]]);
//echo "add=" . $a->add($b) . "\n";
echo "divide=" . $a->mul(2) . "\n";
echo "\nResumo: $passed pass, $failed fail.\n";
exit($failed > 0 ? 1 : 0);
