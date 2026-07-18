<?php

declare(strict_types=1);

use ZMatrix\ZTensor;

function profileCase(string $name, callable $operation): void {
    echo "PROFILE_BEGIN $name cold\n"; fflush(STDOUT);
    $coldStart = hrtime(true);
    $cold = $operation();
    $coldMs = (hrtime(true) - $coldStart) / 1.0e6;
    unset($cold);
    echo "PROFILE_COLD $name wrapper_ms=$coldMs\n"; fflush(STDOUT);

    for ($warmup = 0; $warmup < 3; ++$warmup) {
        $result = $operation();
        unset($result);
    }
    echo "PROFILE_BEGIN $name steady\n"; fflush(STDOUT);
    for ($iteration = 0; $iteration < 7; ++$iteration) {
        $start = hrtime(true);
        $result = $operation();
        $elapsed = (hrtime(true) - $start) / 1.0e6;
        unset($result);
        echo "PROFILE_SAMPLE $name iteration=$iteration wrapper_ms=$elapsed\n";
    }
    fflush(STDOUT);
}

$greater = ZTensor::linspace(-2, 2, 1048576)->toGpu();
profileCase('greater_1m', fn() => $greater->greater(0.125));

$tile = ZTensor::linspace(-1, 1, 1048576)->reshape([1024, 1024])->toGpu();
profileCase('tile_1024sq_x2', fn() => ZTensor::tile($tile, 2));

$scan = ZTensor::linspace(-0.001, 0.001, 1048576)->toGpu();
profileCase('cumsum_1m', fn() => $scan->cumsum());

$matrix = ZTensor::ones([2048, 2048])->toGpu();
$vector = ZTensor::linspace(-1, 1, 2048)->toGpu();
profileCase('matvec_2048sq', fn() => $matrix->dot($vector));
