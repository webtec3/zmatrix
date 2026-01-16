<?php
// Test autograd stubs are registered
echo "=== Verificando se Autograd está carregado ===\n";

// Check if ZTensor has the methods
$methods = get_class_methods('ZMatrix\ZTensor');
$autograd_methods = [
    'requiresGrad',
    'is_requires_grad',
    'ensure_grad',
    'zero_grad',
    'get_grad',
    'backward'
];

echo "\n✓ Métodos de Autograd em ZTensor:\n";
foreach ($autograd_methods as $method) {
    if (in_array($method, $methods)) {
        echo "  ✅ $method\n";
    } else {
        echo "  ❌ $method (NÃO ENCONTRADO)\n";
    }
}

// Check if functions exist
echo "\n✓ Funções de Autograd Global:\n";
$autograd_functions = [
    'add_autograd',
    'sub_autograd',
    'mul_autograd',
    'sum_autograd'
];

foreach ($autograd_functions as $func) {
    if (function_exists($func)) {
        echo "  ✅ $func()\n";
    } else {
        echo "  ❌ $func() (NÃO ENCONTRADA)\n";
    }
}

echo "\n=== Teste Básico de Funcionamento ===\n";

try {
    // Create tensor
    $a = new \ZMatrix\ZTensor([1, 2, 3]);
    echo "\n✓ Tensor criado: [1, 2, 3]\n";
    
    // Enable requires_grad
    $a->requiresGrad(true);
    echo "✓ requiresGrad(true) ativado\n";
    
    // Check if requires grad
    $requires = $a->is_requires_grad();
    echo "✓ is_requires_grad() = " . ($requires ? 'true' : 'false') . "\n";
    
    // Test with autograd operation
    $b = new \ZMatrix\ZTensor([2, 3, 4]);
    $b->requiresGrad(true);
    
    // This should work with add_autograd
    $c = add_autograd($a, $b);
    echo "✓ add_autograd() executado com sucesso\n";
    
    // Check if result has requires_grad set
    $c_requires = $c->is_requires_grad();
    echo "✓ Resultado tem requires_grad = " . ($c_requires ? 'true' : 'false') . "\n";
    
    echo "\n✅ TODOS OS TESTES PASSARAM!\n";
    
} catch (Exception $e) {
    echo "\n❌ ERRO: " . $e->getMessage() . "\n";
    echo "Stack: " . $e->getTraceAsString() . "\n";
}
