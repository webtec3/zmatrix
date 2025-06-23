--TEST--
Testa operações básicas e utilitários do ZTensor
--SKIPIF--
<?php if (!extension_loaded("zmatrix")) die("skip"); ?>
--FILE--
<?php
use ZMatrix\ZTensor;

// Operações element-wise (cada uma com tensor fresco)
echo "add="     . ZTensor::arr([[1,2],[3,4]])->add(ZTensor::arr([[5,6],[7,8]]))          . "\n";
echo "sub="     . ZTensor::arr([[1,2],[3,4]])->sub(ZTensor::arr([[5,6],[7,8]]))          . "\n";
echo "mul="     . ZTensor::arr([[1,2],[3,4]])->mul(ZTensor::arr([[5,6],[7,8]]))          . "\n";
echo "divide="  . ZTensor::arr([[5,6],[7,8]])->divide(ZTensor::arr([[1,2],[3,4]]))       . "\n";

// Comparação
echo "greater=" . ZTensor::arr([[5,6],[7,8]])->greater(ZTensor::arr([[1,2],[3,4]]))      . "\n";

// Reduções e estatísticas
echo "sumtotal=" . ZTensor::arr([[1,2],[3,4]])->sumtotal()   . "\n";
echo "mean="     . ZTensor::arr([[1,2],[3,5]])->mean()       . "\n";
echo "min="      . ZTensor::arr([1,2,3,4])->min()            . "\n";
echo "max="      . ZTensor::arr([1,2,3,4])->max()            . "\n";
echo "std="      . round(ZTensor::arr([1,2,3,4])->std(),4)   . "\n";
?>
--EXPECT--
add=[[6,8],[10,12]]
sub=[[-4,-4],[-4,-4]]
mul=[[5,12],[21,32]]
divide=[[5,3],[2.33333,2]]
greater=[[1,1],[1,1]]
sumtotal=10
mean=2.75
min=1
max=4
std=1.291
