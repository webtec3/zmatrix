--TEST--
Testa TODOS os 60 métodos da extensão ZMatrix
--SKIPIF--
<?php if (!extension_loaded("zmatrix")) die("skip"); ?>
--FILE--
<?php
use ZMatrix\ZTensor;

function roundPrint(ZTensor $t, int $precision = 6): string {
    $data = $t->toArray();
    if (is_array($data) && is_array(reset($data))) {
        $flat = array_merge(...$data);
    } else {
        $flat = is_array($data) ? $data : [$data];
    }
    $rounded = array_map(fn($v) => round($v, $precision), $flat);
    return json_encode($rounded);
}

echo "=== CONSTRUTORES E UTILIDADES ===\n";
echo "__construct=" . new ZTensor([2, 3])->size() . "\n";
echo "__toString=" . (string) ZTensor::arr([1, 2, 3]) . "\n";
echo "shape=" . json_encode(ZTensor::arr([[1,2],[3,4]])->shape()) . "\n";
echo "size=" . ZTensor::arr([[1,2],[3,4]])->size() . "\n";
echo "ndim=" . ZTensor::arr([[1,2],[3,4]])->ndim() . "\n";
echo "isEmpty=" . (ZTensor::zeros([0])->isEmpty() ? "true" : "false") . "\n";
echo "toArray=" . json_encode(ZTensor::arr([1,2,3])->toArray()) . "\n";
echo "copy=" . ZTensor::arr([1,2,3])->copy() . "\n";
echo "key=" . ZTensor::arr([[1,2],[3,4]])->key([0,1]) . "\n";

echo "\n=== CRIAÇÃO DE TENSORES ===\n";
echo "zeros=" . ZTensor::zeros([2, 2]) . "\n";
echo "ones=" . ZTensor::ones([2, 2]) . "\n";
echo "full=" . ZTensor::full([1, 3], 5.0) . "\n";
echo "random=" . (ZTensor::random([2, 2])->size() === 4 ? "OK" : "FAIL") . "\n";
echo "randn=" . (ZTensor::randn([2, 2])->size() === 4 ? "OK" : "FAIL") . "\n";
echo "arange=" . ZTensor::arange(0, 5, 1.0) . "\n";
echo "linspace=" . ZTensor::linspace(0, 10, 5) . "\n";
echo "logspace=" . ZTensor::logspace(0, 2, 3) . "\n";
echo "identity=" . ZTensor::identity(3) . "\n";
echo "eye=" . ZTensor::eye(3, 3) . "\n";
echo "arr=" . ZTensor::arr([[1,2],[3,4]]) . "\n";
echo "safe=" . ZTensor::safe([[5,6],[7,8]]) . "\n";

echo "\n=== OPERAÇÕES ARITMÉTICAS ===\n";
echo "add=" . ZTensor::arr([[1,2],[3,4]])->add([[1,1],[1,1]]) . "\n";
echo "sub=" . ZTensor::arr([[5,6],[7,8]])->sub([[1,1],[1,1]]) . "\n";
echo "mul=" . ZTensor::arr([[2,3],[4,5]])->mul([[2,2],[2,2]]) . "\n";
echo "divide=" . roundPrint(ZTensor::arr([[6,8],[10,12]])->divide([[2,2],[2,2]])) . "\n";
echo "scalarMultiply=" . ZTensor::arr([1,2,3])->scalarMultiply(2.0) . "\n";
echo "scalarDivide=" . roundPrint(ZTensor::arr([2,4,6])->scalarDivide(2.0)) . "\n";
echo "matmul=" . ZTensor::arr([[1,2,3]])->matmul([[4],[5],[6]]) . "\n";
echo "dot=" . ZTensor::arr([1,2,3])->dot([4,5,6]) . "\n";

echo "\n=== OPERAÇÕES MATEMÁTICAS ===\n";
echo "abs=" . ZTensor::arr([[-1,-2],[3,-4]])->abs() . "\n";
echo "exp=" . roundPrint(ZTensor::arr([[0,1]])->exp()) . "\n";
echo "log=" . roundPrint(ZTensor::arr([[1,2.718282]])->log()) . "\n";
echo "sqrt=" . roundPrint(ZTensor::arr([[4,9]])->sqrt()) . "\n";
echo "pow=" . ZTensor::arr([[2,3]])->pow(3) . "\n";

echo "\n=== ATIVAÇÕES (FORWARD) ===\n";
echo "sigmoid=" . roundPrint(ZTensor::arr([[-1,0,1]])->sigmoid()) . "\n";
echo "relu=" . ZTensor::arr([[-2,-1,0,1,2]])->relu() . "\n";
echo "tanh=" . roundPrint(ZTensor::arr([[-1,0,1]])->tanh()) . "\n";
echo "leakyRelu=" . roundPrint(ZTensor::arr([[-1,0,1]])->leakyRelu(0.01)) . "\n";
echo "softmax=" . roundPrint(ZTensor::arr([[1,2,3]])->softmax()) . "\n";

echo "\n=== ATIVAÇÕES (DERIVATIVES) ===\n";
echo "sigmoidDerivative=" . roundPrint(ZTensor::arr([[0.5]])->sigmoidDerivative()) . "\n";
echo "reluDerivative=" . ZTensor::arr([[-1,0,1]])->reluDerivative() . "\n";
echo "tanhDerivative=" . roundPrint(ZTensor::arr([[0]])->tanhDerivative()) . "\n";
echo "leakyReluDerivative=" . roundPrint(ZTensor::arr([[-1,0,1]])->leakyReluDerivative(0.01)) . "\n";
echo "softmaxDerivative=" . (ZTensor::arr([[1,2,3]])->softmax()->softmaxDerivative() ? "OK" : "FAIL") . "\n";

echo "\n=== REDUÇÃO E COMPARAÇÃO ===\n";
$t = ZTensor::arr([1,2,3,4]);
echo "sum=" . $t->sum($t) . "\n";
echo "sumtotal=" . ZTensor::arr([[1,2],[3,4]])->sumtotal() . "\n";
echo "mean=" . ZTensor::arr([1,2,3,4])->mean() . "\n";
echo "min=" . ZTensor::arr([4,2,1,3])->min() . "\n";
echo "max=" . ZTensor::arr([4,2,1,3])->max() . "\n";
echo "std=" . round(ZTensor::arr([1,2,3,4])->std(), 3) . "\n";
echo "greater=" . ZTensor::arr([1,2,3])->greater([2,2,2]) . "\n";
echo "minimum=" . roundPrint(ZTensor::minimum([1,5,3], 2.5)) . "\n";
echo "maximum=" . roundPrint(ZTensor::maximum([1,5,3], 2.5)) . "\n";

echo "\n=== TRANSFORMAÇÃO E FORMATO ===\n";
echo "reshape=" . ZTensor::arr([[1,2],[3,4]])->reshape([4]) . "\n";
echo "transpose=" . ZTensor::arr([[1,2],[3,4]])->transpose() . "\n";
echo "broadcast=" . ZTensor::zeros([2,2])->broadcast(ZTensor::arr([10,20])) . "\n";
echo "tile=" . ZTensor::tile(ZTensor::arr([[1,2]]), 2) . "\n";
echo "clip=" . ZTensor::clip([[-5,0,5,15]], 0.0, 10.0) . "\n";

echo "\n✅ === TOTAL: 60 MÉTODOS TESTADOS COM SUCESSO ===\n";
?>
--EXPECTF--
=== CONSTRUTORES E UTILIDADES ===
__construct=6
__toString=[1,2,3]
shape=[2,2]
size=4
ndim=2
isEmpty=true
toArray=[1,2,3]
copy=[1,2,3]
key=2

=== CRIAÇÃO DE TENSORES ===
zeros=[[0,0],[0,0]]
ones=[[1,1],[1,1]]
full=[[5,5,5]]
random=OK
randn=OK
arange=[0,1,2,3,4]
linspace=[0,2.5,5,7.5,10]
logspace=[1,10,100]
identity=[[1,0,0],[0,1,0],[0,0,1]]
eye=[[1,0,0],[0,1,0],[0,0,1]]
arr=[[1,2],[3,4]]
safe=[[5,6],[7,8]]

=== OPERAÇÕES ARITMÉTICAS ===
add=[[2,3],[4,5]]
sub=[[4,5],[6,7]]
mul=[[4,6],[8,10]]
divide=[3,4,5,6]
scalarMultiply=[2,4,6]
scalarDivide=[1,2,3]
matmul=[[32]]
dot=32

=== OPERAÇÕES MATEMÁTICAS ===
abs=[[1,2],[3,4]]
exp=[1,2.718282]
log=[0,1]
sqrt=[2,3]
pow=[[8,27]]

=== ATIVAÇÕES (FORWARD) ===
sigmoid=[0.268941,0.5,0.731059]
relu=[[0,0,0,1,2]]
tanh=[-0.761594,0,0.761594]
leakyRelu=[-0.01,0,1]
softmax=[0.090031,0.244728,0.665241]

=== ATIVAÇÕES (DERIVATIVES) ===
sigmoidDerivative=[0.25]
reluDerivative=[[0,0,1]]
tanhDerivative=[1]
leakyReluDerivative=[0.01,0.01,1]
softmaxDerivative=OK

=== REDUÇÃO E COMPARAÇÃO ===
sum=10
sumtotal=10
mean=2.5
min=1
max=4
std=1.291
greater=[[0],[1],[1]]
minimum=[1,2.5,2.5]
maximum=[2.5,5,3]

=== TRANSFORMAÇÃO E FORMATO ===
reshape=[1,2,3,4]
transpose=[[1,3],[2,4]]
broadcast=[[10,20],[10,20]]
tile=[[1,2],[1,2]]
clip=[[0,0,5,10]]

✅ === TOTAL: 60 MÉTODOS TESTADOS COM SUCESSO ===
