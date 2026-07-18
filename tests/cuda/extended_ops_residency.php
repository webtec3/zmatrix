<?php

use ZMatrix\ZTensor;

$comparisonScan = ZTensor::arr([-2, 1, 3, -1])->toGpu()->greater(0)->cumsum();
$broadcastChain = ZTensor::zeros([64, 64])->toGpu()
    ->broadcast(ZTensor::ones([1, 64]))
    ->sqrt()
    ->softmax();
$matvecChain = ZTensor::ones([64, 64])->toGpu()->dot(ZTensor::ones([64]))->softmax();
$tileReduction = ZTensor::tile(ZTensor::ones([17, 19])->toGpu(), 3)->sum();

if (!$comparisonScan->isOnGpu() || !$broadcastChain->isOnGpu() || !$matvecChain->isOnGpu()) {
    throw new RuntimeException('resident chain left the device');
}
if (abs($tileReduction->toArray()[0] - 969.0) > 1.0e-5) throw new RuntimeException('tile reduction mismatch');

echo "Resident chains completed without intermediate host reads.\n";
