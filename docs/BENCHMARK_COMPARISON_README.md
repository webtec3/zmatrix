# 🏁 ZMatrix vs NumPy/CuPy Benchmark Suite

Compare performance de ZMatrix contra NumPy (CPU) e CuPy (GPU).

## 📋 O que está incluído

### Scripts
1. **`benchmark_numpy_cupy.py`** - Benchmarks Python com NumPy e CuPy
2. **`benchmark_zmatrix.php`** - Benchmarks PHP com ZMatrix
3. **`generate_benchmark_report.php`** - Gerador de relatório comparativo
4. **`run_benchmark_comparison.sh`** - Orquestrador que executa tudo

## 🚀 Quick Start

### Requisitos

```bash
# Python
python3 -m pip install numpy
python3 -m pip install cupy-cuda-12x  # Opcional, para GPU

# PHP
# ZMatrix extension deve estar instalado e habilitado
php -m | grep zmatrix
```

### Executar Tudo

```bash
bash run_benchmark_comparison.sh
```

Isso vai:
1. ✅ Verificar dependências
2. ✅ Executar benchmarks Python
3. ✅ Executar benchmarks PHP
4. ✅ Gerar relatório comparativo
5. ✅ Salvar resultados em JSON

## 📊 Benchmarks Incluídos

### 1. Creation and Initialization
- Random [1M elements]
- Zeros [1M elements]
- Ones [1M elements]

**Frameworks:** NumPy, CuPy, ZMatrix

### 2. Arithmetic Operations [5M elements]
- Addition
- Subtraction
- Multiplication
- Division

**Frameworks:** NumPy, CuPy (CPU), ZMatrix (CPU), ZMatrix (GPU)

### 3. Activation Functions [5M elements]
- ReLU
- Sigmoid
- Tanh
- Softmax

**Frameworks:** NumPy, CuPy, ZMatrix (CPU), ZMatrix (GPU)

### 4. Linear Algebra
- Matrix Multiplication [1000x1000]
- Dot Product [1M elements]

**Frameworks:** NumPy, CuPy, ZMatrix

### 5. Statistics [5M elements]
- Sum
- Mean
- Standard Deviation
- Min/Max

**Frameworks:** NumPy, CuPy, ZMatrix

## 📈 Resultados

Os resultados são salvos em:
- `benchmark_numpy_cupy_results.json` - Raw data NumPy/CuPy
- `benchmark_zmatrix_results.json` - Raw data ZMatrix
- `BENCHMARK_COMPARISON_REPORT.md` - Relatório formatado

### Exemplo de Resultado

```markdown
# Benchmark Comparison Report

## Creation and Initialization

| Operation | NumPy | CuPy | ZMatrix | Winner |
|-----------|-------|------|---------|--------|
| Random [1M] | 2.345 ms | 0.234 ms | 1.456 ms | CuPy ⚡ |
| Zeros [1M] | 0.123 ms | 0.045 ms | 0.089 ms | CuPy ⚡ |
| Ones [1M] | 0.098 ms | 0.042 ms | 0.076 ms | CuPy ⚡ |
```

## 🎯 Interpretação de Resultados

### Speedup
- **🚀 10x+**: ZMatrix muito mais rápido
- **⚡ 5-10x**: ZMatrix significativamente mais rápido
- **✅ 2-5x**: ZMatrix mais rápido
- **➡️ 1-2x**: Performance similar
- **🐢 <1x**: NumPy é mais rápido

### Casos de Uso

**ZMatrix vence em:**
- Operações com CPU em PHP
- GPU acceleration para tensores grandes
- Integração com aplicações PHP

**NumPy vence em:**
- Ecossistema Python maduro
- Comunidade e documentação
- Tooling e debugging

## 🔧 Execução Manual

### Python Benchmarks
```bash
python3 benchmark_numpy_cupy.py
```

Saída: `benchmark_numpy_cupy_results.json`

### PHP Benchmarks
```bash
php benchmark_zmatrix.php
```

Saída: `benchmark_zmatrix_results.json`

### Gerar Relatório
```bash
php generate_benchmark_report.php \
  benchmark_numpy_cupy_results.json \
  benchmark_zmatrix_results.json
```

Saída: `BENCHMARK_COMPARISON_REPORT.md`

## 📊 Analisar Resultados

### Ver JSON bruto
```bash
cat benchmark_numpy_cupy_results.json | python3 -m json.tool
cat benchmark_zmatrix_results.json | python3 -m json.tool
```

### Ver Relatório Markdown
```bash
cat BENCHMARK_COMPARISON_REPORT.md
```

### Importar em Python para análise
```python
import json

with open('benchmark_numpy_cupy_results.json') as f:
    numpy_results = json.load(f)

with open('benchmark_zmatrix_results.json') as f:
    zmatrix_results = json.load(f)

# Comparar tempos
for key in numpy_results:
    np_time = numpy_results[key]['avg_ms']
    zm_time = zmatrix_results.get(key.replace('numpy', 'zmatrix'), {}).get('avg_ms')
    if zm_time:
        speedup = np_time / zm_time
        print(f"{key}: {speedup:.2f}x")
```

## 🐛 Troubleshooting

### CuPy não encontrado
```bash
pip3 install cupy-cuda-12x
# Escolha a versão correta de CUDA:
# cupy-cuda-11x para CUDA 11.x
# cupy-cuda-12x para CUDA 12.x
```

### ZMatrix extension não carregado
```bash
php -m | grep zmatrix
# Se não aparecer, compile e instale:
./configure --enable-zmatrix
make && sudo make install
echo "extension=zmatrix.so" | sudo tee -a /etc/php/8.x/cli/php.ini
```

### Erro de permissão em bash script
```bash
chmod +x run_benchmark_comparison.sh
bash run_benchmark_comparison.sh
```

### GPU não detectado em benchmarks
```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
php benchmark_zmatrix.php
```

## 📈 Otimizar Benchmarks

### Para operações mais longas
Editar `iterations` em cada script:

**Python:**
```python
result = benchmark(name, func, iterations=10)  # Aumentar de 5
```

**PHP:**
```php
benchmark_php("Name", $func, 10)  # Aumentar de 5
```

### Para tensores maiores
Editar `size` nas operações:

```python
size = 10_000_000  # Aumentar de 5_000_000
```

```php
$size = 10_000_000;  // Aumentar de 5_000_000
```

## 📚 Referências

- [NumPy Documentation](https://numpy.org/doc/)
- [CuPy Documentation](https://docs.cupy.dev/)
- [ZMatrix README](README.md)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)

## 🎓 Próximos Passos

1. **Executar benchmarks**: `bash run_benchmark_comparison.sh`
2. **Analisar resultados**: Abrir `BENCHMARK_COMPARISON_REPORT.md`
3. **Comparar operações**: Ver qual framework é melhor para seu caso
4. **Otimizar**: Usar o framework mais rápido para cada operação

## 📊 Exemplo de Saída

```
╔════════════════════════════════════════════════════════════════╗
║  ZMatrix vs NumPy/CuPy Benchmark Comparison                   ║
╚════════════════════════════════════════════════════════════════╝

📋 Checking dependencies...

✅ Python3: Python 3.10.0
✅ NumPy installed
✅ CuPy installed (GPU benchmarks enabled)
✅ PHP: PHP 8.1.0
✅ ZMatrix extension loaded

═══════════════════════════════════════════════════════════════
Running benchmarks...
═══════════════════════════════════════════════════════════════

🐍 Running Python/NumPy/CuPy benchmarks...
✅ Python benchmarks completed

🐘 Running PHP/ZMatrix benchmarks...
✅ PHP benchmarks completed

📊 Generating comparison report...
✅ Report generated: ./BENCHMARK_COMPARISON_REPORT.md
   
✅ Benchmark comparison completed!
```

---

**Happy Benchmarking! 🚀**
