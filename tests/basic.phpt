--TEST--
Testa operações básicas e utilitários do ZTensor
--SKIPIF--
<?php if (!extension_loaded("zmatrix")) die("skip"); ?>
--FILE--
<?php
use ZMatrix\ZTensor;

// Dados de entrada
$a = ZTensor::arr([[1,2],[3,4]]);
$b = ZTensor::arr([[5,6],[7,8]]);

// Operações element-wise
echo "add="     . $a->add($b)          . "\n";   // [[6,8],[10,12]]
echo "sub="     . $a->sub($b)          . "\n";   // [[-4,-4],[-4,-4]]
echo "mul="     . $a->mul($b)          . "\n";   // [[5,12],[21,32]]
echo "divide="  . $b->divide($a)       . "\n";   // [[5,3],[2.333333,2]]

// Comparação
echo "greater=" . $b->greater($a)      . "\n";   // [[1,1],[1,1]]

// Reduções e estatísticas
echo "sumtotal=" . ZTensor::arr([[1,2],[3,4]])->sumtotal()   . "\n"; // 10
echo "mean="     . ZTensor::arr([[1,2],[3,5]])->mean()       . "\n"; // 2.75
echo "min="      . ZTensor::arr([1,2,3,4])->min()            . "\n"; // 1
echo "max="      . ZTensor::arr([1,2,3,4])->max()            . "\n"; // 4
echo "std="      . round(ZTensor::arr([1,2,3,4])->std(),4)   . "\n"; // ~1.290

// Transformações de formato
echo "reshape="   . $a->reshape([4,1])    . "\n"; // [[1],[2],[3],[4]]
echo "transpose=" . $a->transpose()       . "\n"; // [[1,3],[2,4]]

// Funções matemáticas
$c = ZTensor::arr([[-1,-2],[3,-4]]);
echo "abs="      . $c->abs()             . "\n"; // [[1,2],[3,4]]
echo "exp="      . ZTensor::arr([[0,1]])->exp()          . "\n"; // [[1, 2.7182818]]
echo "log="      . ZTensor::arr([[1,10]])->log()         . "\n"; // [[0, 2.302585]]
echo "sqrt="     . ZTensor::arr([[4,9]])->sqrt()         . "\n"; // [[2,3]]
echo "pow="      . ZTensor::arr([[2,3]])->pow(3)         . "\n"; // [[8,27]]

// Ativações
$t = ZTensor::arr([[-1,0,1]]);
echo "sigmoid="  . $t->sigmoid()          . "\n"; // [[0.2689,0.5,0.7311]]
echo "relu="     . $t->relu()             . "\n"; // [[0,0,1]]
echo "tanh="     . $t->tanh()             . "\n"; // [[-0.7616,0,0.7616]]

// Softmax
$s = ZTensor::arr([1,2,3]);
echo "softmax="  . $s->softmax()          . "\n"; // [[0.0900,0.2447,0.6652]]

// Broadcast, estáticos extras
echo "minimum="  . ZTensor::minimum([1,5,3], 2.5)   . "\n"; // [[1,2.5,2.5]]
echo "maximum="  . ZTensor::maximum([1,5,3], 2.5)   . "\n"; // [[2.5,5,3]]
echo "zeros="    . ZTensor::zeros([2,2])           . "\n"; // [[0,0],[0,0]]
echo "ones="     . ZTensor::ones([2,2])            . "\n"; // [[1,1],[1,1]]
echo "full="     . ZTensor::full([1,3], 9)         . "\n"; // [[9,9,9]]
echo "identity=" . ZTensor::identity(2)            . "\n"; // [[1,0],[0,1]]

// Matmul
$m = ZTensor::arr([[1,2,3]]);
$n = ZTensor::arr([[4],[5],[6]]);
echo "matmul="   . $m->matmul($n)         . "\n"; // [[32]]

// Forma e metadados
echo "shape="    . json_encode($a->shape())   . "\n"; // [2,2]
echo "ndim="     . $a->ndim()                . "\n"; // 2
echo "size="     . $a->size()                . "\n"; // 4
echo "isEmpty="  . ($a->isEmpty() ? 'true' : 'false') . "\n";

// ToArray (linha única para verificação)
var_export(ZTensor::arr([[1,2],[3,4]])->toArray());

// Tile e copy/safe
$d = ZTensor::arr([1,2]);
echo "\ntile="    . ZTensor::tile($d, 2)      . "\n"; // [[1,2],[1,2]]
echo "copy="      . $a->copy()               . "\n";
echo "safe="      . ZTensor::safe([[7,8],[9,10]]) . "\n";
?>
--EXPECT--
add=[[6,8],[10,12]]
sub=[[-4,-4],[-4,-4]]
mul=[[5,12],[21,32]]
divide=[[5,3],[2.3333333,2]]
greater=[[1,1],[1,1]]
sumtotal=10
mean=2.75
min=1
max=4
std=1.290
reshape=[[1],[2],[3],[4]]
transpose=[[1,3],[2,4]]
abs=[[1,2],[3,4]]
exp=[[1,2.7182817]]
log=[[0,2.3025851]]
sqrt=[[2,3]]
pow=[[8,27]]
sigmoid=[[0.2689414,0.5,0.7310586]]
relu=[[0,0,1]]
tanh=[[-0.7615942,0,0.7615942]]
softmax=[[0.0900306,0.2447285,0.6652409]]
minimum=[[1,2.5,2.5]]
maximum=[[2.5,5,3]]
zeros=[[0,0],[0,0]]
ones=[[1,1],[1,1]]
full=[[9,9,9]]
identity=[[1,0],[0,1]]
matmul=[[32]]
shape=[2,2]
ndim=2
size=4
isEmpty=false
array (
  0 => 
  array (
    0 => 1.0,
    1 => 2.0,
  ),
  1 => 
  array (
    0 => 3.0,
    1 => 4.0,
  ),
)
tile=[[1,2],[1,2]]
copy=[[1,2],[3,4]]
safe=[[7,8],[9,10]]
