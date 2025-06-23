--TEST--
Testa operações básicas e utilitários do ZTensor
--SKIPIF--
<?php if (!extension_loaded("zmatrix")) die("skip"); ?>
--FILE--
<?php
use ZMatrix\ZTensor;

// Função segura para arredondar e exibir tensores 1D ou 2D
function roundPrint(ZTensor $t, int $precision = 6): string {
    $data = $t->toArray();
    if (is_array($data) && is_array(reset($data))) {
        $flat = array_merge(...$data);
    } else {
        $flat = $data;
    }
    $rounded = array_map(fn($v) => round($v, $precision), $flat);
    return json_encode($rounded);
}

// Operações element-wise
echo "add="     . ZTensor::arr([[1,2],[3,4]])->add([[5,6],[7,8]])          . "\n";
echo "sub="     . ZTensor::arr([[1,2],[3,4]])->sub([[5,6],[7,8]])          . "\n";
echo "mul="     . ZTensor::arr([[1,2],[3,4]])->mul([[5,6],[7,8]])          . "\n";
echo "divide="  . ZTensor::arr([[5,6],[7,8]])->divide([[1,2],[3,4]])       . "\n";

// Comparação
echo "greater=" . ZTensor::arr([[5,6],[7,8]])->greater([[1,2],[3,4]])      . "\n";

// Estatísticas
echo "sumtotal=" . ZTensor::arr([[1,2],[3,4]])->sumtotal() . "\n";
echo "mean="     . ZTensor::arr([[1,2],[3,5]])->mean()     . "\n";
echo "min="      . ZTensor::arr([1,2,3,4])->min()          . "\n";
echo "max="      . ZTensor::arr([1,2,3,4])->max()          . "\n";
echo "std="      . round(ZTensor::arr([1,2,3,4])->std(), 3) . "\n";

// Ativações (rounded)
echo "sigmoid=" . roundPrint(ZTensor::arr([[-1,0,1]])->sigmoid()) . "\n";
echo "relu="    . roundPrint(ZTensor::arr([[-1,0,1]])->relu())    . "\n";
echo "tanh="    . roundPrint(ZTensor::arr([[-1,0,1]])->tanh())    . "\n";

// Matemáticas
echo "abs="     . ZTensor::arr([[-1,-2],[3,-4]])->abs()    . "\n";
echo "exp="     . roundPrint(ZTensor::arr([[0,1]])->exp()) . "\n";
echo "log="     . roundPrint(ZTensor::arr([[1,10]])->log()) . "\n";
echo "sqrt="    . roundPrint(ZTensor::arr([[4,9]])->sqrt()) . "\n";
echo "pow="     . ZTensor::arr([[2,3]])->pow(3) . "\n";

// Estáticos
echo "minimum=" . roundPrint(ZTensor::minimum([1,5,3,-2,0], 2.5)) . "\n";
echo "maximum=" . roundPrint(ZTensor::maximum([1,5,3,-2,0], 2.5)) . "\n";
echo "zeros="   . ZTensor::zeros([2,2]) . "\n";
echo "ones="    . ZTensor::ones([2,2])  . "\n";
echo "full="    . ZTensor::full([1,3], 9) . "\n";
echo "identity=". ZTensor::identity(2) . "\n";

// Formato
echo "reshape="   . ZTensor::arr([[1,2],[3,4]])->reshape([4,1]) . "\n";
echo "transpose=" . ZTensor::arr([[1,2],[3,4]])->transpose()    . "\n";

// Extras
echo "matmul=" . ZTensor::arr([[1,2,3]])->matmul([[4],[5],[6]]) . "\n"; // [[32]]
echo "tile="   . ZTensor::tile(ZTensor::arr([[1,2]]), 3) . "\n"; // [[1,2],[1,2],[1,2]]
echo "safe="   . ZTensor::safe([[7,8],[9,10]]) . "\n"; // [[7,8],[9,10]]
$base = ZTensor::arr([[1,2],[3,4]]);
$bias = ZTensor::arr([10,20]);
echo "broadcast=" . $base->broadcast($bias) . "\n"; // [[11,22],[13,24]]
echo "clip=" . ZTensor::clip([[-1,5,15]], 0.0, 10.0) . "\n"; // [[0,5,10]]
?>
--EXPECTF--
add=[[6,8],[10,12]]
sub=[[-4,-4],[-4,-4]]
mul=[[5,12],[21,32]]
divide=[[5,3],[2.333333,2]]
greater=[[1,1],[1,1]]
sumtotal=10
mean=2.75
min=1
max=4
std=1.291
sigmoid=[0.268941,0.5,0.731059]
relu=[0,0,1]
tanh=[-0.761594,0,0.761594]
abs=[[1,2],[3,4]]
exp=[1,2.718282]
log=[0,2.302585]
sqrt=[2,3]
pow=[[8,27]]
minimum=[1,2.5,2.5,-2,0]
maximum=[2.5,5,3,2.5,2.5]
zeros=[[0,0],[0,0]]
ones=[[1,1],[1,1]]
full=[[9,9,9]]
identity=[[1,0],[0,1]]
reshape=[[1],[2],[3],[4]]
transpose=[[1,3],[2,4]]
matmul=[[32]]
tile=[[1,2],[1,2],[1,2]]
safe=[[7,8],[9,10]]
broadcast=[[11,22],[13,24]]
clip=[[0,5,10]]
