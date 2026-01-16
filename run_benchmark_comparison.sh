#!/usr/bin/env bash

###############################################################################
#  ZMatrix vs NumPy/CuPy Benchmark Comparison Suite
#  Executa ambos benchmarks e gera relatório comparativo
###############################################################################

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ZMatrix vs NumPy/CuPy Benchmark Comparison                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check dependencies
echo "📋 Checking dependencies..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found!"
    exit 1
fi
echo "✅ Python3: $(python3 --version)"

# Check NumPy
if ! python3 -c "import numpy" 2>/dev/null; then
    echo "❌ NumPy not installed!"
    echo "   Install with: pip3 install numpy"
    exit 1
fi
echo "✅ NumPy installed"

# Check CuPy (optional)
if python3 -c "import cupy" 2>/dev/null; then
    echo "✅ CuPy installed (GPU benchmarks enabled)"
    CUPY_AVAILABLE=1
else
    echo "⚠️  CuPy not installed (GPU benchmarks skipped)"
    echo "   Install with: pip3 install cupy-cuda-12x"
    CUPY_AVAILABLE=0
fi

# Check PHP
if ! command -v php &> /dev/null; then
    echo "❌ PHP not found!"
    exit 1
fi
echo "✅ PHP: $(php --version | head -n 1)"

# Check if ZMatrix extension is loaded
if ! php -r "extension_loaded('zmatrix') or die('ZMatrix not loaded');" 2>/dev/null; then
    echo "❌ ZMatrix extension not loaded!"
    echo "   Please install and enable the extension first."
    exit 1
fi
echo "✅ ZMatrix extension loaded"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Running benchmarks..."
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Run Python benchmarks
echo "🐍 Running Python/NumPy/CuPy benchmarks..."
python3 benchmark_numpy_cupy.py
PYTHON_RESULTS="$PROJECT_DIR/benchmark_numpy_cupy_results.json"

if [ ! -f "$PYTHON_RESULTS" ]; then
    echo "❌ Python benchmarks failed!"
    exit 1
fi
echo "✅ Python benchmarks completed"
echo ""

# Setup GPU environment for PHP
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Run PHP benchmarks
echo "🐘 Running PHP/ZMatrix benchmarks..."
php benchmark_zmatrix.php
PHP_RESULTS="$PROJECT_DIR/benchmark_zmatrix_results.json"

if [ ! -f "$PHP_RESULTS" ]; then
    echo "❌ PHP benchmarks failed!"
    exit 1
fi
echo "✅ PHP benchmarks completed"
echo ""

# Generate comparison report
echo "📊 Generating comparison report..."
php generate_benchmark_report.php "$PYTHON_RESULTS" "$PHP_RESULTS"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Benchmark comparison completed!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📄 Results files:"
echo "   • Python:  $PYTHON_RESULTS"
echo "   • PHP:     $PHP_RESULTS"
echo "   • Report:  $PROJECT_DIR/BENCHMARK_COMPARISON_REPORT.md"
echo ""
echo "📖 View the report with:"
echo "   cat BENCHMARK_COMPARISON_REPORT.md"
echo ""
