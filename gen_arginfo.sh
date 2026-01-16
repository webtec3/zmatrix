#!/bin/bash
# gen_arginfo.sh - Gera argumentinfo a partir de .stub.php files
# Executado automaticamente durante make se stubs mudarem

set -e

cd "$(dirname "$0")"

# Verificar se gen_stub.php existe
if [ ! -f "build/gen_stub.php" ]; then
    echo "❌ Error: build/gen_stub.php not found"
    exit 1
fi

# Regenerar apenas se stubs forem mais recentes que arginfo
NEEDS_REGEN=0

if [ ! -f "zmatrix_arginfo.h" ] || [ "zmatrix.stub.php" -nt "zmatrix_arginfo.h" ]; then
    NEEDS_REGEN=1
fi

if [ ! -f "ztensor_arginfo.h" ] || [ "ztensor.stub.php" -nt "ztensor_arginfo.h" ]; then
    NEEDS_REGEN=1
fi

if [ $NEEDS_REGEN -eq 1 ]; then
    echo "🔄 Regenerating arginfo from .stub.php files..."
    php build/gen_stub.php zmatrix.stub.php ztensor.stub.php
    echo "✅ Generated: zmatrix_arginfo.h and ztensor_arginfo.h"
fi
