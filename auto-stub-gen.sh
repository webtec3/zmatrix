#!/bin/bash
# auto-stub-gen.php - Wrapper para gerar stubs automaticamente
# Este script é chamado ANTES de cada compilação

# Detectar diretório do script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Verificar se gen_stub.php existe
if [ ! -f "build/gen_stub.php" ]; then
    echo "⚠️  Warning: build/gen_stub.php not found, skipping arginfo generation"
    exit 0
fi

# Se ztensor.stub.php não existe ou está vazio, restaurar do git
if [ ! -f "ztensor.stub.php" ] || [ ! -s "ztensor.stub.php" ]; then
    if git show HEAD:ztensor.stub.php > /dev/null 2>&1; then
        echo "🔄 Restaurando ztensor.stub.php do git..."
        git checkout HEAD -- ztensor.stub.php 2>/dev/null
    else
        echo "❌ Erro: ztensor.stub.php não encontrado e não está em git"
        exit 1
    fi
fi

# Regenerar arginfo automaticamente
echo "🔄 Regenerando arginfo from .stub.php files..."
php build/gen_stub.php zmatrix.stub.php ztensor.stub.php 2>&1

exit 0
