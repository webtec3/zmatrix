--TEST--
Testa operações básicas e utilitários do ZTensor
--SKIPIF--
<?php if (!extension_loaded("zmatrix")) die("skip"); ?>
--FILE--
<?php
use ZMatrix\ZTensor;

// Operações element-wise
echo "add="     . ZTensor::arr([[1,2],[3,4]])->add([[5,6],[7,8]])          . "\n";
echo "sub="     . ZTensor::arr([[1,2],[3,4]])->sub([[5,6],[7,8]])          . "\n";
echo "mul="     . ZTensor::arr([[1,2],[3,4]])->mul([[5,6],[7,8]])          . "\n";
echo "divide="  . ZTensor::arr([[5,6],[7,8]])->divide([[1,2],[3,4]])       . "\n";

// Comparação
echo "greater=" . ZTensor::arr([[5,6],[7,8]])->greater([[1,2],[3,4]])      . "\n";

// Estatísticas e reduções
echo "sumtotal=" . ZTensor::arr([[1,2],[3,4]])->sumtotal()   . "\n";
echo "mean="     . ZTensor::arr([[1,2],[3,5]])->mean()       . "\n";
echo "min="      . ZTensor::arr([1,2,3,4])->min()            . "\n";
echo "max="      . ZTensor::arr([1,2,3,4])->max()            . "\n";
echo "std="      . round(ZTensor::arr([1,2,3,4])->std(), 4)  . "\n";

// Ativações
echo "sigmoid="  . ZTensor::arr([[-1, 0, 1]])->sigmoid()     . "\n";
echo "relu="     . ZTensor::arr([[-1, 0, 1]])->relu()        . "\n";
echo "tanh="     . ZTensor::arr([[-1, 0, 1]])->tanh()        . "\n";

// Funções matemáticas
echo "abs="      . ZTensor::arr([[-1, -2], [3, -4]])->abs()      . "\n";
echo "exp="      . ZTensor::arr([[0, 1]])->exp()                . "\n";
echo "log="      . ZTensor::arr([[1, 10]])->log()               . "\n";
echo "sqrt="     . ZTensor::arr([[4, 9]])->sqrt()               . "\n";
echo "pow="      . ZTensor::arr([[2, 3]])->pow(3)               . "\n";

// Estáticos
echo "minimum="  . ZTensor::minimum([1,5,3,-2,0], 2.5)     . "\n";
echo "maximum="  . ZTensor::maximum([1,5,3,-2,0], 2.5)     . "\n";
echo "zeros="    . ZTensor::zeros([2,2])                  . "\n";
echo "ones="     . ZTensor::ones([2,2])                   . "\n";
echo "full="     . ZTensor::full([1,3], 9)                . "\n";
echo "identity=" . ZTensor::identity(2)                   . "\n";

// Transposição e formato
echo "reshape="   . ZTensor::arr([[1,2],[3,4]])->reshape([4,1])    . "\n";
echo "transpose=" . ZTensor::arr([[1,2],[3,4]])->transpose()       . "\n";
?>
--EXPECTF--
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
sigmoid=[[0.26894,0.5,0.73106]]
relu=[[0,0,1]]
tanh=[[-0.76159,0,0.76159]]
abs=[[1,2],[3,4]]
exp=[[1,2.71828]]
log=[[0,2.30258]]
sqrt=[[2,3]]
pow=[[8,27]]
minimum=[[1,2.5,2.5,-2,0]]
maximum=[[2.5,5,3,2.5,2.5]]
zeros=[[0,0],[0,0]]
ones=[[1,1],[1,1]]
full=[[9,9,9]]
identity=[[1,0],[0,1]]
reshape=[[1],[2],[3],[4]]
transpose=[[1,3],[2,4]]
