#!/bin/bash

PROJECT_DIR="/home/omgaalfa/php-projetos/php-extension/zmatrix"
cd "$PROJECT_DIR"

echo "╔════════════════════════════════════════════════╗"
echo "║    EXECUTANDO TODOS OS TESTES PHP - DIA 4     ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Arquivos a executar (excluindo stubs e gen_stub)
PHP_FILES=(
    "test.php"
    "example.php"
    "test_dia4.php"
    "test_dia4_extended.php"
    "benchmark.php"
    "benchmark_comparative.php"
    "benchmark_precise.php"
    "benchmark_simd_test.php"
    "benchmark_validated.php"
    "test_activations.php"
    "test_heavy.php"
    "test_race_conditions.php"
    "validate_math.php"
    "bench_simd.php"
    "stress_test.php"
)

TOTAL=${#PHP_FILES[@]}
PASSED=0
FAILED=0

for i in "${!PHP_FILES[@]}"; do
    FILE="${PHP_FILES[$i]}"
    NUM=$((i + 1))
    
    if [ ! -f "$FILE" ]; then
        echo "[$NUM/$TOTAL] ⏭️  $FILE (não encontrado)"
        continue
    fi
    
    echo "[$NUM/$TOTAL] 🧪 Executando: $FILE"
    
    if timeout 30 php "$FILE" > /tmp/test_output.txt 2>&1; then
        PASSED=$((PASSED + 1))
        echo "      ✅ OK"
    else
        FAILED=$((FAILED + 1))
        echo "      ❌ ERRO"
        echo "      --- Primeira linha de erro ---"
        head -5 /tmp/test_output.txt | sed 's/^/      /'
    fi
    echo ""
done

echo "════════════════════════════════════════════════"
echo "📊 RESULTADOS: $PASSED/$TOTAL OK | $FAILED falhas"
echo "════════════════════════════════════════════════"
