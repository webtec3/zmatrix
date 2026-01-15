#!/bin/bash
# setup_gpu_wsl.sh - Configura GPU permanentemente no WSL2

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         ZMatrix GPU Setup for WSL2 - Automatic            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if running in WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "⚠️  This script is designed for WSL2"
    echo "   Running it on native Linux/macOS may not work"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if /usr/lib/wsl/lib exists
if [ ! -d "/usr/lib/wsl/lib" ]; then
    echo "❌ Error: /usr/lib/wsl/lib not found"
    echo "   GPU support may not be available in this WSL instance"
    exit 1
fi

echo "✅ WSL2 with GPU support detected"
echo ""

# Option 1: Add to bashrc
echo "Option 1: Add to ~/.bashrc (recommended for development)"
read -p "Add LD_LIBRARY_PATH to ~/.bashrc? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if ! grep -q "/usr/lib/wsl/lib" ~/.bashrc 2>/dev/null; then
        echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
        echo "✅ Added to ~/.bashrc"
        echo "   Run: source ~/.bashrc"
    else
        echo "⚠️  Already in ~/.bashrc"
    fi
    echo ""
fi

# Option 2: Create wrapper script
echo "Option 2: Create php-gpu wrapper script"
read -p "Create /usr/local/bin/php-gpu wrapper? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -f /usr/local/bin/php-gpu ] || ! grep -q "/usr/lib/wsl/lib" /usr/local/bin/php-gpu; then
        cat > /tmp/php-gpu.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
exec php "$@"
EOF
        sudo mv /tmp/php-gpu.sh /usr/local/bin/php-gpu
        sudo chmod +x /usr/local/bin/php-gpu
        echo "✅ Created /usr/local/bin/php-gpu"
        echo "   Usage: php-gpu your_script.php"
    else
        echo "⚠️  Already exists"
    fi
    echo ""
fi

# Option 3: Test current setup
echo "Option 3: Test GPU with current PHP"
read -p "Run GPU test? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    if [ -f "$SCRIPT_DIR/gpu_test_complete.php" ]; then
        echo ""
        echo "Running GPU test..."
        echo "═════════════════════════════════════════════════════════"
        LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH php "$SCRIPT_DIR/gpu_test_complete.php" 2>&1 | head -100
        echo "═════════════════════════════════════════════════════════"
    else
        echo "⚠️  gpu_test_complete.php not found in $(pwd)"
        echo "   Manual test:"
        LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH php -r "
use ZMatrix\ZTensor;
\$a = ZTensor::random([1000000]);
\$a->toGpu();
\$b = ZTensor::random([1000000]);
\$b->toGpu();
\$t0 = microtime(true);
\$a->add(\$b);
\$t1 = microtime(true);
echo 'GPU Add: ' . ((\$t1 - \$t0) * 1000) . ' ms\n';
"
    fi
    echo ""
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                     Setup Complete!                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "📋 What you need to know:"
echo ""
echo "1. ALWAYS use LD_LIBRARY_PATH when running PHP with GPU:"
echo "   LD_LIBRARY_PATH=/usr/lib/wsl/lib:\$LD_LIBRARY_PATH php script.php"
echo ""
echo "2. For best performance, move tensors to GPU FIRST:"
echo "   \$tensor->toGpu();  // Move to GPU memory"
echo "   \$tensor->add(...); // Operation is now on GPU"
echo ""
echo "3. Check GPU status:"
echo "   LD_LIBRARY_PATH=/usr/lib/wsl/lib:\$LD_LIBRARY_PATH ZMATRIX_GPU_DEBUG=1 php script.php"
echo ""
echo "4. Monitor GPU usage:"
echo "   watch -n 0.5 nvidia-smi"
echo ""

echo "📦 Performance tips:"
echo "- Without toGpu(): ~228ms for 1M elements (with H2D copy)"
echo "- With toGpu(): ~0.14ms for 1M elements (1600x faster!)"
echo ""

echo "🔗 Next steps:"
echo "1. If using bashrc: source ~/.bashrc && php your_script.php"
echo "2. If using wrapper: php-gpu your_script.php"
echo "3. Otherwise: LD_LIBRARY_PATH=/usr/lib/wsl/lib:\$LD_LIBRARY_PATH php your_script.php"
echo ""
