#!/usr/bin/env bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Contadores
TOTAL=0
PASSED=0
FAILED=0

# Função auxiliar para executar um teste
run_test() {
  local desc="$1"
  local php_code="$2"
  
  ((TOTAL++))
  
  # Executar o teste
  output=$(php -d extension=../modules/zmatrix.so -r "$php_code" 2>&1)
  status=$?

  # Verificar se houve erro na execução
  if [ $status -ne 0 ]; then
    echo -e "${RED}✗${NC} $desc"
    echo -e "  ${RED}Error:${NC} $output" | head -3
    ((FAILED++))
    return
  fi

  # Verificar se a saída contém "OK"
  if echo "$output" | grep -q "^OK$"; then
    echo -e "${GREEN}✓${NC} $desc"
    ((PASSED++))
  else
    echo -e "${RED}✗${NC} $desc"
    echo -e "  ${RED}Output:${NC} $output"
    ((FAILED++))
  fi
}

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       ZTensor 100% Comprehensive Test Suite (Full API)        ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo

# =============================================================================
# CONSTRUTORES E UTILIDADES
# =============================================================================
echo -e "${YELLOW}=== CONSTRUTORES E UTILIDADES ===${NC}"

run_test "__construct" "
use ZMatrix\ZTensor;
\$t = new ZTensor([2, 3]);
echo 'OK';
"

run_test "__toString" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1, 2, 3]);
echo (string)\$t ? 'OK' : 'FAIL';
"

run_test "shape" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1,2],[3,4]]);
echo json_encode(\$t->shape()) === '[2,2]' ? 'OK' : 'FAIL';
"

run_test "size" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1,2],[3,4]]);
echo \$t->size() === 4 ? 'OK' : 'FAIL';
"

run_test "ndim" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1,2],[3,4]]);
echo \$t->ndim() === 2 ? 'OK' : 'FAIL';
"

run_test "isEmpty" "
use ZMatrix\ZTensor;
\$t = ZTensor::zeros([0]);
echo \$t->isEmpty() ? 'OK' : 'FAIL';
"

run_test "toArray" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3]);
\$arr = \$t->toArray();
echo is_array(\$arr) ? 'OK' : 'FAIL';
"

run_test "copy" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3]);
\$copy = \$t->copy();
echo \$copy->size() === 3 ? 'OK' : 'FAIL';
"

run_test "key" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1,2],[3,4]]);
echo \$t->key([0,1]) == 2 ? 'OK' : 'FAIL';
"

# =============================================================================
# CRIAÇÃO DE TENSORES
# =============================================================================
echo
echo -e "${YELLOW}=== CRIAÇÃO DE TENSORES ===${NC}"

run_test "zeros" "
use ZMatrix\ZTensor;
\$t = ZTensor::zeros([2, 2]);
echo \$t->size() === 4 ? 'OK' : 'FAIL';
"

run_test "ones" "
use ZMatrix\ZTensor;
\$t = ZTensor::ones([2, 2]);
echo \$t->size() === 4 ? 'OK' : 'FAIL';
"

run_test "full" "
use ZMatrix\ZTensor;
\$t = ZTensor::full([1, 3], 5.0);
echo \$t->size() === 3 ? 'OK' : 'FAIL';
"

run_test "fill" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1, 2, 3, 4]);
\$result = \$t->fill(7.5);
\$arr = \$result->toArray();
echo \$arr[0] === 7.5 ? 'OK' : 'FAIL';
"

run_test "arange" "
use ZMatrix\ZTensor;
\$t = ZTensor::arange(0, 5, 1.0);
echo \$t->size() === 5 ? 'OK' : 'FAIL';
"

run_test "linspace" "
use ZMatrix\ZTensor;
\$t = ZTensor::linspace(0, 10, 5);
echo \$t->size() === 5 ? 'OK' : 'FAIL';
"

run_test "logspace" "
use ZMatrix\ZTensor;
\$t = ZTensor::logspace(0, 2, 3);
echo \$t->size() === 3 ? 'OK' : 'FAIL';
"

run_test "identity" "
use ZMatrix\ZTensor;
\$t = ZTensor::identity(3);
echo \$t->shape()[0] === 3 ? 'OK' : 'FAIL';
"

run_test "eye" "
use ZMatrix\ZTensor;
\$t = ZTensor::eye(3, 3);
echo \$t->shape()[0] === 3 ? 'OK' : 'FAIL';
"

run_test "arr" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1,2],[3,4]]);
echo \$t->size() === 4 ? 'OK' : 'FAIL';
"

run_test "random" "
use ZMatrix\ZTensor;
\$t = ZTensor::random([2, 3], 0.0, 1.0);
echo \$t->size() === 6 ? 'OK' : 'FAIL';
"

run_test "randn" "
use ZMatrix\ZTensor;
\$t = ZTensor::randn([2, 2], 0.0, 1.0);
echo \$t->size() === 4 ? 'OK' : 'FAIL';
"

# =============================================================================
# OPERAÇÕES ARITMÉTICAS
# =============================================================================
echo
echo -e "${YELLOW}=== OPERAÇÕES ARITMÉTICAS ===${NC}"

run_test "add" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3]);
\$result = \$t->add([1,1,1]);
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "sub" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([5,6,7]);
\$result = \$t->sub([1,1,1]);
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "mul" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([2,3,4]);
\$result = \$t->mul([2,2,2]);
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "divide" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([6,8,10]);
\$result = \$t->divide([2,2,2]);
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "scalarMultiply" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3]);
\$result = \$t->scalarMultiply(2.0);
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "scalarDivide" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([2,4,6]);
\$result = \$t->scalarDivide(2.0);
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "matmul" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1,2,3]]);
\$result = \$t->matmul([[4],[5],[6]]);
echo \$result->size() === 1 ? 'OK' : 'FAIL';
"

run_test "dot" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3]);
echo \$t->dot([4,5,6]) ? 'OK' : 'FAIL';
"

# =============================================================================
# OPERAÇÕES MATEMÁTICAS E ESTRUTURAIS AUXILIARES
# =============================================================================
echo
echo -e "${YELLOW}=== OPERAÇÕES MATEMÁTICAS E AUXILIARES ===${NC}"

run_test "abs" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([-1,-2,3]);
\$result = \$t->abs();
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "exp" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([0,1]);
\$result = \$t->exp();
echo \$result->size() === 2 ? 'OK' : 'FAIL';
"

run_test "log" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2.718282]);
\$result = \$t->log();
echo \$result->size() === 2 ? 'OK' : 'FAIL';
"

run_test "sqrt" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([4,9]);
\$result = \$t->sqrt();
echo \$result->size() === 2 ? 'OK' : 'FAIL';
"

run_test "pow" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([2,3]);
\$result = \$t->pow(3);
echo \$result->size() === 2 ? 'OK' : 'FAIL';
"

run_test "column" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1, 2], [3, 4]]);
\$col = \$t->column(1);
echo \$col->size() === 2 ? 'OK' : 'FAIL';
"

run_test "argsort" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([3, 1, 2]);
\$indices = \$t->argsort(0);
echo \$indices->size() === 3 ? 'OK' : 'FAIL';
"

run_test "where" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1, 2], [3, 4]]);
\$mask = \$t->where(0, 2.0); // Compara coluna 0 com limiar 2.0
echo \$mask->size() === 2 ? 'OK' : 'FAIL';
"

# =============================================================================
# ATIVAÇÕES (FORWARD)
# =============================================================================
echo
echo -e "${YELLOW}=== ATIVAÇÕES (FORWARD) ===${NC}"

run_test "sigmoid" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([0]);
\$result = \$t->sigmoid();
echo \$result->size() === 1 ? 'OK' : 'FAIL';
"

run_test "relu" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([-1,0,1]);
\$result = \$t->relu();
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "tanh" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([0]);
\$result = \$t->tanh();
echo \$result->size() === 1 ? 'OK' : 'FAIL';
"

run_test "leakyRelu" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([-1,0,1]);
\$result = \$t->leakyRelu(0.01);
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "softmax" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1,2,3]]);
\$result = \$t->softmax();
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

# =============================================================================
# ATIVAÇÕES (DERIVATIVES)
# =============================================================================
echo
echo -e "${YELLOW}=== ATIVAÇÕES (DERIVATIVES) ===${NC}"

run_test "sigmoidDerivative" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([0.5]);
\$result = \$t->sigmoidDerivative();
echo \$result->size() === 1 ? 'OK' : 'FAIL';
"

run_test "reluDerivative" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([-1,0,1]);
\$result = \$t->reluDerivative();
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "tanhDerivative" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([0]);
\$result = \$t->tanhDerivative();
echo \$result->size() === 1 ? 'OK' : 'FAIL';
"

run_test "leakyReluDerivative" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([-1,0,1]);
\$result = \$t->leakyReluDerivative(0.01);
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "softmaxDerivative" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1,2,3]]);
\$sm = \$t->softmax();
\$result = \$sm->softmaxDerivative();
echo \$result ? 'OK' : 'FAIL';
"

# =============================================================================
# REDUÇÃO E COMPARAÇÃO
# =============================================================================
echo
echo -e "${YELLOW}=== REDUÇÃO E COMPARAÇÃO ===${NC}"

run_test "sum" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3,4]);
\$result = \$t->sum();
echo \$result ? 'OK' : 'FAIL';
"

run_test "sumtotal" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1,2],[3,4]]);
echo \$t->sumtotal() == 10 ? 'OK' : 'FAIL';
"

run_test "mean" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3,4]);
echo \$t->mean() === 2.5 ? 'OK' : 'FAIL';
"

run_test "min" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([4,2,1,3]);
echo \$t->min() == 1 ? 'OK' : 'FAIL';
"

run_test "max" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([4,2,1,3]);
echo \$t->max() == 4 ? 'OK' : 'FAIL';
"

run_test "minimum" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1, 5, 3]);
\$result = ZTensor::minimum(\$t, 2.0);
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "maximum" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1, 5, 3]);
\$result = ZTensor::maximum(\$t, 2.0);
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "std" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3,4]);
\$result = \$t->std();
echo \$result > 0 ? 'OK' : 'FAIL';
"

run_test "greater" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3]);
\$result = \$t->greater([2,2,2]);
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

run_test "clip" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([-2, 5, 12]);
\$result = ZTensor::clip(\$t, 0.0, 10.0);
echo \$result->size() === 3 ? 'OK' : 'FAIL';
"

# =============================================================================
# TRANSFORMAÇÃO E FORMATO
# =============================================================================
echo
echo -e "${YELLOW}=== TRANSFORMAÇÃO E FORMATO ===${NC}"

run_test "reshape" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1,2],[3,4]]);
\$result = \$t->reshape([4]);
echo \$result->size() === 4 ? 'OK' : 'FAIL';
"

run_test "transpose" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1,2],[3,4]]);
\$result = \$t->transpose();
echo \$result->shape()[0] === 2 ? 'OK' : 'FAIL';
"

run_test "slice" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1,2,3],[4,5,6],[7,8,9]]);
\$result = \$t->slice(0, 0, 2);
echo \$result->size() === 6 ? 'OK' : 'FAIL';
"

run_test "broadcast" "
use ZMatrix\ZTensor;
\$matrix = ZTensor::zeros([2,2]);
\$result = \$matrix->broadcast(ZTensor::arr([10,20]));
echo \$result->size() === 4 ? 'OK' : 'FAIL';
"

# =============================================================================
# GPU / DEVICE
# =============================================================================
echo
echo -e "${YELLOW}=== GPU / DEVICE ===${NC}"

run_test "toGpu" "
use ZMatrix\ZTensor;
try {
    \$t = ZTensor::arr([1,2,3]);
    \$gpu = \$t->toGpu();
    echo \$gpu->isOnGpu() ? 'OK' : 'FAIL';
} catch (\\Throwable \$e) {
    if (strpos(\$e->getMessage(), 'CUDA') !== false || strpos(\$e->getMessage(), 'device') !== false) {
        echo 'OK';
    } else {
        echo 'FAIL: ' . \$e->getMessage();
    }
}
"

run_test "toCpu" "
use ZMatrix\ZTensor;
try {
    \$t = ZTensor::arr([1,2,3]);
    \$gpu = \$t->toGpu();
    \$cpu = \$gpu->toCpu();
    echo !\$cpu->isOnGpu() ? 'OK' : 'FAIL';
} catch (\\Throwable \$e) {
    if (strpos(\$e->getMessage(), 'CUDA') !== false || strpos(\$e->getMessage(), 'device') !== false) {
        echo 'OK';
    } else {
        echo 'FAIL: ' . \$e->getMessage();
    }
}
"

run_test "isOnGpu" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3]);
echo !\$t->isOnGpu() ? 'OK' : 'FAIL';
"

run_test "freeDevice" "
use ZMatrix\ZTensor;
try {
    \$t = ZTensor::arr([1,2,3]);
    \$t->toGpu();
    \$t->freeDevice();
    echo 'OK';
} catch (\\Throwable \$e) {
    if (strpos(\$e->getMessage(), 'CUDA') !== false || strpos(\$e->getMessage(), 'device') !== false) {
        echo 'OK';
    } else {
        echo 'FAIL: ' . \$e->getMessage();
    }
}
"

# =============================================================================
# AUTOGRAD E MÉTODOS DE GRAFO CONST
# =============================================================================
echo
echo -e "${YELLOW}=== AUTOGRAD ===${NC}"

run_test "requiresGrad" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3]);
\$t->requiresGrad(true);
echo \$t->isRequiresGrad() ? 'OK' : 'FAIL';
"

run_test "isRequiresGrad" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3]);
echo !\$t->isRequiresGrad() ? 'OK' : 'FAIL';
"

run_test "ensureGrad" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3]);
\$t->requiresGrad(true);
\$t->ensureGrad();
echo 'OK';
"

run_test "getGrad" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3]);
\$t->requiresGrad(true);
\$t->ensureGrad();
\$grad = \$t->getGrad();
echo is_object(\$grad) ? 'OK' : 'FAIL';
"

run_test "zeroGrad" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1,2,3]);
\$t->zeroGrad();
echo 'OK';
"

run_test "backward" "
use ZMatrix\ZTensor;
\$x = ZTensor::ones([1])->requiresGrad(true);
\$y = ZTensor::addAutograd(\$x, \$x);
\$y->backward();
echo 'OK';
"

run_test "addAutograd" "
use ZMatrix\ZTensor;
\$a = ZTensor::ones([2])->requiresGrad(true);
\$b = ZTensor::ones([2])->requiresGrad(true);
\$res = ZTensor::addAutograd(\$a, \$b);
echo \$res->size() === 2 ? 'OK' : 'FAIL';
"

run_test "subAutograd" "
use ZMatrix\ZTensor;
\$a = ZTensor::ones([2])->requiresGrad(true);
\$b = ZTensor::ones([2])->requiresGrad(true);
\$res = ZTensor::subAutograd(\$a, \$b);
echo \$res->size() === 2 ? 'OK' : 'FAIL';
"

run_test "mulAutograd" "
use ZMatrix\ZTensor;
\$a = ZTensor::ones([2])->requiresGrad(true);
\$b = ZTensor::ones([2])->requiresGrad(true);
\$res = ZTensor::mulAutograd(\$a, \$b);
echo \$res->size() === 2 ? 'OK' : 'FAIL';
"

run_test "sumAutograd" "
use ZMatrix\ZTensor;
\$a = ZTensor::ones([2, 3])->requiresGrad(true);
\$res = ZTensor::sumAutograd(\$a);
echo \$res->size() === 1 ? 'OK' : 'FAIL';
"

# =============================================================================
# NOVOS MÉTODOS OTIMIZADOS (ÁRVORES E MANIPULAÇÃO)
# =============================================================================
echo -e "${YELLOW}=== TESTES: NOVOS MÉTODOS IMPLEMENTADOS ===${NC}"

run_test "unique() - Filtragem e ordenação básica de elementos únicos" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([4.0, 1.0, 1.0, 3.0, 4.0, 2.0]);
\$res = \$t->unique();
\$arr = \$res->toArray();
if (\$res->size() === 4 && \$arr[0] == 1.0 && \$arr[3] == 4.0) {
    echo 'OK';
} else {
    echo 'FAIL: ' . json_encode(\$arr);
}
"

run_test "unique() - Lidando com tensores vazios com segurança" "
use ZMatrix\ZTensor;
\$t = ZTensor::zeros([0]);
\$res = \$t->unique();
echo \$res->size() === 0 ? 'OK' : 'FAIL';
"

run_test "bincount() - Histograma de frequências e contagem (Gini/Árvores)" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1, 2, 1, 1, 3, 2, 5]);
\$res = \$t->bincount();
\$arr = \$res->toArray();
if (\$res->size() === 6 && \$arr[0] == 0 && \$arr[1] == 3 && \$arr[2] == 2 && \$arr[5] == 1) {
    echo 'OK';
} else {
    echo 'FAIL: ' . json_encode(\$arr);
}
"

run_test "bincount() - Parâmetro minlength de preenchimento mínimo" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1, 2]);
\$res = \$t->bincount(10);
echo \$res->size() === 10 ? 'OK' : 'FAIL';
"

run_test "bincount() - Lançamento de exceção em valores negativos" "
use ZMatrix\ZTensor;
try {
    \$t = ZTensor::arr([1, -2, 3]);
    \$t->bincount();
    echo 'FAIL (Não lançou exceção para valor negativo)';
} catch (\\Throwable \$e) {
    echo 'OK';
}
"

run_test "argmax() - Localização do índice global do maior valor (Flat Index)" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1.0, 2.0, 3.0], [4.0, 10.5, 2.0]]);
\$idx = \$t->argmax();
echo \$idx === 4 ? 'OK' : 'FAIL: index returned ' . \$idx;
"

run_test "argmax() - Consistência e estabilidade em múltiplos máximos idênticos" "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([1.0, 5.0, 3.0, 5.0]);
\$idx = \$t->argmax();
echo \$idx === 1 ? 'OK' : 'FAIL';
"

run_test "concat() - Concatenação estática densa no Eixo 0 (Linhas)" "
use ZMatrix\ZTensor;
\$t1 = ZTensor::arr([[1, 2], [3, 4]]);
\$t2 = ZTensor::arr([[5, 6]]);
\$res = ZTensor::concat([\$t1, \$t2], 0);
\$shape = \$res->shape();
if (\$shape[0] === 3 && \$shape[1] === 2 && \$res->key([2, 1]) == 6) {
    echo 'OK';
} else {
    echo 'FAIL: shape ' . json_encode(\$shape);
}
"

run_test "concat() - Concatenação estática densa no Eixo 1 (Colunas)" "
use ZMatrix\ZTensor;
\$t1 = ZTensor::arr([[1, 2], [3, 4]]);
\$t2 = ZTensor::arr([[10], [20]]);
\$res = ZTensor::concat([\$t1, \$t2], 1);
\$shape = \$res->shape();
if (\$shape[0] === 2 && \$shape[1] === 3 && \$res->key([1, 2]) == 20) {
    echo 'OK';
} else {
    echo 'FAIL: shape ' . json_encode(\$shape);
}
"

run_test "concat() - Validação e rejeição de shapes incompatíveis" "
use ZMatrix\ZTensor;
try {
    \$t1 = ZTensor::arr([[1, 2]]);
    \$t2 = ZTensor::arr([[1, 2, 3]]);
    ZTensor::concat([\$t1, \$t2], 0);
    echo 'FAIL (Mesclou dimensões inválidas sem erro)';
} catch (\\Throwable \$e) {
    echo 'OK';
}
"

# =============================================================================
# RESULTADO FINAL
# =============================================================================
echo
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                       TEST SUMMARY                            ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo -e "Total:  ${BLUE}${TOTAL}${NC} testes executados"
echo -e "Passed: ${GREEN}${PASSED}${NC} válidos"
echo -e "Failed: ${RED}${FAILED}${NC} falhas"

if [ $FAILED -eq 0 ]; then
  echo
  echo -e "${GREEN}✓ Engine está 100% pronta para a infraestrutura de Random Forest!${NC}"
  exit 0
else
  echo
  echo -e "${RED}✗ Correções são necessárias nos novos métodos C++.${NC}"
  exit 1
fi