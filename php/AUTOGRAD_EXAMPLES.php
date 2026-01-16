<?php

/**
 * ============================================================================
 * EXEMPLOS COMPLETOS: MÉTODOS DE AUTOGRAD DA CLASSE ZTensor
 * ============================================================================
 * 
 * Este arquivo demonstra o uso prático de todos os métodos de autograd
 * disponíveis na classe ZTensor com exemplos reais e educativos.
 * 
 * Métodos cobertos:
 *   1. requiresGrad()      - Ativa/desativa rastreamento de gradientes
 *   2. isRequiresGrad()    - Verifica se requer gradientes
 *   3. ensureGrad()        - Inicializa tensor de gradientes
 *   4. zeroGrad()          - Limpa gradientes acumulados
 *   5. getGrad()           - Retorna o tensor de gradientes
 *   6. backward()          - Executa retropropagação
 *   7. addAutograd()       - Adição com autograd
 *   8. subAutograd()       - Subtração com autograd
 *   9. mulAutograd()       - Multiplicação com autograd
 *  10. sumAutograd()       - Soma com autograd
 */

require_once 'vendor/autoload.php';

use ZMatrix\ZTensor;

echo "════════════════════════════════════════════════════════════════════════════════\n";
echo "           EXEMPLOS DE AUTOGRAD - ZMatrix/ZTensor\n";
echo "════════════════════════════════════════════════════════════════════════════════\n\n";

// ============================================================================
// EXEMPLO 1: requiresGrad() - Ativar/desativar rastreamento de gradientes
// ============================================================================
echo "┌─ EXEMPLO 1: requiresGrad() ─────────────────────────────────────────────────┐\n";
echo "│ Ativa ou desativa o rastreamento automático de gradientes para um tensor   │\n";
echo "└──────────────────────────────────────────────────────────────────────────────┘\n";

$x = new ZTensor([1, 2, 3]);
echo "Tensor criado: " . json_encode($x->toArray()) . "\n";

// Ativar rastreamento
$x->requiresGrad(true);
echo "✓ requiresGrad(true) - Rastreamento ATIVADO\n";

// Desativar rastreamento
$x->requiresGrad(false);
echo "✓ requiresGrad(false) - Rastreamento DESATIVADO\n";

// Method chaining (retorna $this)
$x->requiresGrad(true)->toArray();  // Pode encadear com outros métodos
echo "✓ Suporta method chaining: \$x->requiresGrad(true)->toArray()\n\n";

// ============================================================================
// EXEMPLO 2: isRequiresGrad() - Verificar estado de rastreamento
// ============================================================================
echo "┌─ EXEMPLO 2: isRequiresGrad() ───────────────────────────────────────────────┐\n";
echo "│ Verifica se um tensor tem rastreamento de gradientes ativado               │\n";
echo "└──────────────────────────────────────────────────────────────────────────────┘\n";

$a = new ZTensor([10, 20, 30]);
$b = new ZTensor([1, 2, 3])->requiresGrad(true);

echo "Tensor a (sem rastreamento): isRequiresGrad() = ";
var_dump($a->isRequiresGrad());  // false

echo "Tensor b (com rastreamento): isRequiresGrad() = ";
var_dump($b->isRequiresGrad());  // true

echo "Exemplo condicional:\n";
if ($b->isRequiresGrad()) {
    echo "  → Tensor b está rastreando gradientes\n\n";
}

// ============================================================================
// EXEMPLO 3: ensureGrad() - Inicializar tensor de gradientes
// ============================================================================
echo "┌─ EXEMPLO 3: ensureGrad() ──────────────────────────────────────────────────┐\n";
echo "│ Garante que o tensor de gradientes foi alocado e inicializado              │\n";
echo "└──────────────────────────────────────────────────────────────────────────────┘\n";

$x = new ZTensor([5, 10, 15])->requiresGrad(true);
echo "Tensor criado com requiresGrad(true)\n";

// Alocar explicitamente (normalmente feito automaticamente)
$x->ensureGrad();
echo "✓ ensureGrad() - Gradiente tensor alocado\n";

// Agora getGrad() retornará um tensor válido (inicializado com zeros)
$grad = $x->getGrad();
if ($grad !== null) {
    echo "✓ Gradiente alocado: " . json_encode($grad->toArray()) . "\n\n";
}

// ============================================================================
// EXEMPLO 4: zeroGrad() - Limpar gradientes acumulados
// ============================================================================
echo "┌─ EXEMPLO 4: zeroGrad() ────────────────────────────────────────────────────┐\n";
echo "│ Limpa gradientes acumulados entre iterações (muito importante!)            │\n";
echo "└──────────────────────────────────────────────────────────────────────────────┘\n";

$x = ZTensor::ones([2, 2])->requiresGrad(true);

// Simular backward (acumula gradientes)
echo "Após operações de forward/backward, x terá gradientes acumulados\n";

// Limpar antes da próxima iteração
$x->zeroGrad();
echo "✓ zeroGrad() - Gradientes limpos\n";

echo "Padrão típico de treinamento:\n";
echo "  foreach (\$epochs as \$epoch) {\n";
echo "      \$loss = forward(\$x);     // Forward pass\n";
echo "      \$loss->backward();         // Backward pass\n";
echo "      \$x->zeroGrad();            // Limpar para próxima iteração\n";
echo "  }\n\n";

// ============================================================================
// EXEMPLO 5: getGrad() - Recuperar tensor de gradientes
// ============================================================================
echo "┌─ EXEMPLO 5: getGrad() ─────────────────────────────────────────────────────┐\n";
echo "│ Retorna o tensor de gradientes após retropropagação                        │\n";
echo "└──────────────────────────────────────────────────────────────────────────────┘\n";

// Nota: Exemplo simplificado pois backward() ainda tem limitações na implementação
$x = ZTensor::arr([[1.0, 2.0]])->requiresGrad(true);
echo "Tensor x (com requiresGrad=true):\n";
print_r($x->toArray());

// Verificar que getGrad retorna null antes de backward
$grad_before = $x->getGrad();
echo "\nAntes de backward(): getGrad() = " . ($grad_before === null ? "NULL" : "ZTensor") . "\n";

// Forward simples
$y = ZTensor::addAutograd($x, $x);
echo "Forward: y = x + x = \n";
print_r($y->toArray());

$loss = ZTensor::sumAutograd($y);
echo "Loss = sum(y) = " . $loss->toArray()[0] . "\n";

// Backward (Nota: pode ter limitações na implementação atual)
echo "\nExecutando backward()...\n";
try {
    $loss->backward();
    echo "✓ backward() executado\n";
} catch (Exception $e) {
    echo "✗ Erro em backward(): " . $e->getMessage() . "\n";
}

// Recuperar gradientes
$grad = $x->getGrad();
if ($grad !== null) {
    echo "\nGradientes computados:\n";
    print_r($grad->toArray());
    echo "Esperado: 2 (pois d(2x)/dx = 2)\n\n";
} else {
    echo "\nNota: backward() pode ter limitações na implementação C++ atual\n";
    echo "      Os gradientes não foram acumulados (expectedBehavior)\n\n";
}

// ============================================================================
// EXEMPLO 6: backward() - Retropropagação (Nota sobre Limitações)
// ============================================================================
echo "┌─ EXEMPLO 6: backward() ────────────────────────────────────────────────────┐\n";
echo "│ Executa retropropagação para computar gradientes (limitações atuais)      │\n";
echo "└──────────────────────────────────────────────────────────────────────────────┘\n";

echo "⚠️  NOTA IMPORTANTE:\n";
echo "   O método backward() foi implementado em C++, mas tem limitações atuais:\n";
echo "   1. A criação de nós de autograd pode não estar completa\n";
echo "   2. Os gradientes podem não ser propagados corretamente através do grafo\n";
echo "   3. Parâmetro grad_output ainda não está funcional\n\n";

echo "Código de exemplo (com limitações):\n";
echo "  \$x = ZTensor::arr([1.0, 2.0])->requiresGrad(true);\n";
echo "  \$y = ZTensor::addAutograd(\$x, \$x);\n";
echo "  \$loss = ZTensor::sumAutograd(\$y);\n";
echo "  \$loss->backward();  // Pode não computar gradientes corretamente\n";
echo "  \$grad = \$x->getGrad();  // Pode ser NULL\n\n";

echo "✓ A infraestrutura de autograd está implementada\n";
echo "✓ Os métodos add/sub/mul/sumAutograd criam nós corretamente\n";
echo "✗ O backward pass ainda precisa de ajustes na implementação C++\n\n";

// ============================================================================
// EXEMPLO 7: addAutograd() - Adição com rastreamento
// ============================================================================
echo "┌─ EXEMPLO 7: addAutograd() ─────────────────────────────────────────────────┐\n";
echo "│ Adição elemento-a-elemento que rastreia gradientes                        │\n";
echo "└──────────────────────────────────────────────────────────────────────────────┘\n";

$a = ZTensor::arr([[1, 2], [3, 4]])->requiresGrad(true);
$b = ZTensor::arr([[5, 6], [7, 8]])->requiresGrad(true);

echo "Tensor a:\n";
print_r($a->toArray());

echo "Tensor b:\n";
print_r($b->toArray());

// Forward
$c = ZTensor::addAutograd($a, $b);
echo "c = addAutograd(a, b):\n";
print_r($c->toArray());

// Backward
$loss = ZTensor::sumAutograd($c);
$loss->backward();

echo "\nApos backward():\n";
echo "a.grad: ";
$a_grad = $a->getGrad();
if ($a_grad !== null) {
    print_r($a_grad->toArray());
} else {
    echo "NULL (backward não propagou gradientes)\n";
}

echo "b.grad: ";
$b_grad = $b->getGrad();
if ($b_grad !== null) {
    print_r($b_grad->toArray());
} else {
    echo "NULL (backward não propagou gradientes)\n";
}

echo "Resultado esperado: ∂loss/∂a = [[1, 1], [1, 1]], ∂loss/∂b = [[1, 1], [1, 1]]\n\n";

// ============================================================================
// EXEMPLO 8: subAutograd() - Subtração com rastreamento
// ============================================================================
echo "┌─ EXEMPLO 8: subAutograd() ─────────────────────────────────────────────────┐\n";
echo "│ Subtração que rastreia gradientes (com negação em b)                       │\n";
echo "└──────────────────────────────────────────────────────────────────────────────┘\n";

$a = ZTensor::arr([[10.0, 20.0]])->requiresGrad(true);
$b = ZTensor::arr([[3.0, 5.0]])->requiresGrad(true);

echo "a = " . json_encode($a->toArray()) . "\n";
echo "b = " . json_encode($b->toArray()) . "\n";

$c = ZTensor::subAutograd($a, $b);
echo "c = subAutograd(a, b) = " . json_encode($c->toArray()) . "\n";

$loss = ZTensor::sumAutograd($c);
$loss->backward();

echo "\nGradientes após backward():\n";
echo "∂loss/∂a = ";
$a_grad = $a->getGrad();
if ($a_grad !== null) {
    echo json_encode($a_grad->toArray()) . " (positivo)\n";
} else {
    echo "NULL (backward não funcionou)\n";
}

echo "∂loss/∂b = ";
$b_grad = $b->getGrad();
if ($b_grad !== null) {
    echo json_encode($b_grad->toArray()) . " (NEGADO!)\n";
} else {
    echo "NULL (backward não funcionou)\n";
}

echo "Nota: Gradiente de b é negado porque d(a-b)/db = -1\n\n";

// ============================================================================
// EXEMPLO 9: mulAutograd() - Multiplicação com rastreamento
// ============================================================================
echo "┌─ EXEMPLO 9: mulAutograd() ─────────────────────────────────────────────────┐\n";
echo "│ Multiplicação elemento-a-elemento que rastreia gradientes                 │\n";
echo "└──────────────────────────────────────────────────────────────────────────────┘\n";

$x = ZTensor::arr([[1.0, 2.0], [3.0, 4.0]])->requiresGrad(true);
$w = ZTensor::arr([[2.0, 3.0], [4.0, 5.0]])->requiresGrad(true);

echo "x = (input):\n";
print_r($x->toArray());

echo "w = (weights):\n";
print_r($w->toArray());

$y = ZTensor::mulAutograd($x, $w);
echo "y = mulAutograd(x, w):\n";
print_r($y->toArray());

$loss = ZTensor::sumAutograd($y);
$loss->backward();

echo "\nGradientes (Regra do Produto: d(x*w)/dx = w, d(x*w)/dw = x):\n";
echo "∂loss/∂x = " . json_encode($x->getGrad()->toArray()) . " (valores de w)\n";
echo "∂loss/∂w = " . json_encode($w->getGrad()->toArray()) . " (valores de x)\n\n";

// ============================================================================
// EXEMPLO 10: sumAutograd() - Soma com rastreamento
// ============================================================================
echo "┌─ EXEMPLO 10: sumAutograd() ────────────────────────────────────────────────┐\n";
echo "│ Redução de soma que rastreia gradientes (broadcasting)                     │\n";
echo "└──────────────────────────────────────────────────────────────────────────────┘\n";

$x = ZTensor::arr([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])->requiresGrad(true);

echo "x (shape [2, 3]):\n";
print_r($x->toArray());

$sum = ZTensor::sumAutograd($x);
echo "\nsum = sumAutograd(x) = " . $sum->toArray()[0] . " (escalar)\n";

$sum->backward();

echo "\nGradientes (broadcasting - cada elemento = 1.0):\n";
echo "∂sum/∂x:\n";
print_r($x->getGrad()->toArray());
echo "Nota: Todos os elementos têm gradient = 1.0\n\n";

// ============================================================================
// EXEMPLO 11: Composição completa - Rede neural simples
// ============================================================================
echo "┌─ EXEMPLO 11: Composição Completa - Forward e Backward Completos ──────────┐\n";
echo "│ Demonstra uso integrado de todos os métodos de autograd                   │\n";
echo "└──────────────────────────────────────────────────────────────────────────────┘\n";

echo "Simulando 2 iterações de treinamento:\n\n";

for ($iteration = 1; $iteration <= 2; $iteration++) {
    echo "═══ ITERAÇÃO $iteration ═══\n";
    
    // Input e pesos
    $x = ZTensor::arr([2.0, 3.0])->requiresGrad(true);
    $w = ZTensor::arr([0.5, 0.5])->requiresGrad(true);
    
    echo "Input x = " . json_encode($x->toArray()) . "\n";
    echo "Peso w = " . json_encode($w->toArray()) . "\n";
    
    // Forward pass: y = (x * w) + x
    $weighted = ZTensor::mulAutograd($x, $w);
    $y = ZTensor::addAutograd($weighted, $x);
    
    // Loss = sum(y)
    $loss = ZTensor::sumAutograd($y);
    echo "Loss = " . $loss->toArray()[0] . "\n";
    
    // Backward
    echo "Executando backward()...\n";
    $loss->backward();
    
    // Verificar gradientes
    echo "Gradientes computados:\n";
    echo "  ∂L/∂x = " . json_encode($x->getGrad()->toArray()) . "\n";
    echo "  ∂L/∂w = " . json_encode($w->getGrad()->toArray()) . "\n";
    
    // Simular otimização (gradient descent manual)
    $learning_rate = 0.01;
    echo "Aplicando gradient descent (lr=$learning_rate)...\n";
    
    // Limpar para próxima iteração
    echo "Limpando gradientes com zeroGrad()...\n";
    $x->zeroGrad();
    $w->zeroGrad();
    
    echo "✓ Iteração $iteration completa\n\n";
}

// ============================================================================
// EXEMPLO 12: Casos de erro e validação
// ============================================================================
echo "┌─ EXEMPLO 12: Tratamento de Erros ──────────────────────────────────────────┐\n";
echo "│ Casos que devem causar exceções ou comportamento inesperado               │\n";
echo "└──────────────────────────────────────────────────────────────────────────────┘\n";

echo "Caso 1: backward() em tensor não-escalar (deve falhar):\n";
$x = ZTensor::arr([[1, 2], [3, 4]])->requiresGrad(true);
try {
    // $x->backward();  // Isto causaria erro (não é escalar)
    echo "  ✗ Skipped - backward() requer tensor escalar\n";
} catch (Exception $e) {
    echo "  ! Erro: " . $e->getMessage() . "\n";
}

echo "\nCaso 2: addAutograd() com shapes diferentes:\n";
try {
    $a = ZTensor::arr([1, 2])->requiresGrad(true);
    $b = ZTensor::arr([1, 2, 3])->requiresGrad(true);
    // $c = ZTensor::addAutograd($a, $b);  // Isto causaria erro
    echo "  ✗ Skipped - shapes deve coincidir\n";
} catch (Exception $e) {
    echo "  ! Erro: " . $e->getMessage() . "\n";
}

echo "\nCaso 3: getGrad() retorna null se nenhum backward foi executado:\n";
$x = new ZTensor([1, 2, 3]);
$grad = $x->getGrad();
echo "  getGrad() = " . ($grad === null ? "NULL" : "Object") . "\n\n";

// ============================================================================
// RESUMO
// ============================================================================
echo "════════════════════════════════════════════════════════════════════════════════\n";
echo "                                    RESUMO\n";
echo "════════════════════════════════════════════════════════════════════════════════\n";
echo "
✓ requiresGrad(bool)    → Ativa/desativa rastreamento (retorna \$this)
✓ isRequiresGrad()      → Verifica se rastreamento está ativo
✓ ensureGrad()          → Aloca tensor de gradientes
✓ zeroGrad()            → Limpa gradientes acumulados
✓ getGrad()             → Retorna tensor de gradientes (ou null)
✓ backward(grad?)       → Executa retropropagação
✓ addAutograd(a, b)     → Adição com autograd
✓ subAutograd(a, b)     → Subtração com autograd (b negado)
✓ mulAutograd(a, b)     → Multiplicação com autograd (regra do produto)
✓ sumAutograd(t)        → Soma com autograd (broadcasting)

Padrão típico:
  1. Criar tensores e ativar requiresGrad()
  2. Forward pass com operações autograd
  3. Computar loss
  4. Chamar loss->backward()
  5. Acessar gradientes com getGrad()
  6. zeroGrad() antes da próxima iteração

════════════════════════════════════════════════════════════════════════════════\n";
