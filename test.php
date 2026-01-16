<?php


use ZMatrix\ZTensor;


$data = ZTensor::arr([-2, -1, 0, 1, 2]);
$leaky_relu = $data->leakyRelu(0.1);
print_r($leaky_relu->toArray());


$tensor = ZTensor::arr([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);
$sum_result = ZTensor::zeros([3]);
$tensor->sum($sum_result, 1); // sum along axis 1
print_r($sum_result->toArray());