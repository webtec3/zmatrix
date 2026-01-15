# GPU Setup para ZMatrix no WSL2

## Status

✅ **GPU está funcionando!** RTX 3060 detectada e ativa.
- Problema: `libcuda.so` está em `/usr/lib/wsl/lib/` - WSL expõe drivers especiais
- Solução: Configurar `LD_LIBRARY_PATH` ou usar rpath

## Solução 1: LD_LIBRARY_PATH (Temporária)

Execute PHP com:
```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH php seu_script.php
```

**Teste rápido:**
```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH php -r "
use ZMatrix\ZTensor;
\$a = ZTensor::random([1000000], -1.0, 1.0);
\$b = ZTensor::random([1000000], -1.0, 1.0);
\$a->toGpu();
\$b->toGpu();
\$t0 = microtime(true);
\$a->add(\$b);
\$t1 = microtime(true);
echo 'GPU Add: ' . ((\$t1 - \$t0) * 1000) . ' ms\n';
"
```

Resultado esperado: **~1-2 ms** (não 262ms sem GPU!)

## Solução 2: LD_LIBRARY_PATH Permanente (Recomendado)

### Opção A: Arquivo .bashrc (permanente para shell)
```bash
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Depois rode seus scripts normalmente:
```bash
php seu_script.php
```

### Opção B: Wrapper Script
Crie `php-gpu.sh`:
```bash
#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
exec php "$@"
```

Use:
```bash
chmod +x php-gpu.sh
./php-gpu.sh seu_script.php
```

### Opção C: Apache/Nginx (se usar servidor web)

**Apache (.htaccess):**
```apache
SetEnv LD_LIBRARY_PATH /usr/lib/wsl/lib
```

**Nginx (não suporta direto, use systemd):**
```bash
sudo systemctl edit php8.3-fpm
```
Adicione:
```
[Service]
Environment="LD_LIBRARY_PATH=/usr/lib/wsl/lib"
```

## Como Usar GPU em ZMatrix

### 1. Ativar Debug (opcional)
```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH php -r "
putenv('ZMATRIX_GPU_DEBUG=1');
// seus códigos...
" 2>&1 | grep "zmatrix\|gpu"
```

Saída esperada:
```
[zmatrix][gpu] devices=1
[zmatrix][gpu] add n=1000000
```

### 2. Mover Tensores para GPU
```php
use ZMatrix\ZTensor;

// Criar tensores
$a = ZTensor::random([1000000], -1.0, 1.0);
$b = ZTensor::random([1000000], -1.0, 1.0);

// **IMPORTANTE**: Mover para GPU antes das operações
$a->toGpu();
$b->toGpu();

// Operações ficarão na GPU (automático)
$a->add($b);        // ~0.1-1 ms
$a->mul($b);        // ~0.1-1 ms
$a->relu();         // ~0.1-1 ms
$a->sigmoid();      // ~0.1-1 ms
$a->tanh();         // ~0.1-1 ms
$a->exp();          // ~0.1-1 ms
$a->abs();          // ~0.1-1 ms

// Verificar localização
if ($a->isOnGpu()) {
    echo "Tensor está na GPU\n";
}

// Retornar para CPU (se necessário)
$a->toCpu();
```

### 3. Comparar GPU vs CPU

```php
use ZMatrix\ZTensor;

$size = 10_000_000; // 10M elementos
$a = ZTensor::random([$size], -1.0, 1.0);
$b = ZTensor::random([$size], -1.0, 1.0);

// CPU
$cpu_a = ZTensor::arr($a);
$t0 = microtime(true);
$cpu_a->add($b);
$t1 = microtime(true);
$cpu_time = ($t1 - $t0) * 1000;

// GPU
$gpu_a = ZTensor::arr($a);
$gpu_a->toGpu();
$gpu_b = ZTensor::arr($b);
$gpu_b->toGpu();
$t0 = microtime(true);
$gpu_a->add($gpu_b);
$t1 = microtime(true);
$gpu_time = ($t1 - $t0) * 1000;

$speedup = $cpu_time / $gpu_time;
echo sprintf("CPU: %.2f ms | GPU: %.2f ms | Speedup: %.2fx\n", 
    $cpu_time, $gpu_time, $speedup);
```

## Operações Suportadas na GPU

### Elemento-a-elemento
- `add($other)` - Adição
- `sub($other)` - Subtração
- `mul($other)` - Multiplicação

### Escalares
- `add(0.5)` - Adição escalar
- `sub(0.5)` - Subtração escalar
- `mul(2.0)` - Multiplicação escalar
- `scalarDivide(2.0)` - Divisão escalar

### Ativações
- `relu()` - ReLU
- `leakyRelu(0.01)` - Leaky ReLU
- `sigmoid()` - Sigmoid
- `tanh()` - Tanh
- `exp()` - Exponencial
- `abs()` - Valor Absoluto

## Limiar de GPU

Operações com **menos de 200k elementos** usam CPU (mais rápido).
Operações com **200k+ elementos** usam GPU automaticamente (se ativado).

Para forçar CPU em debug:
```bash
ZMATRIX_FORCE_CPU=1 php seu_script.php
```

## Troubleshooting

### Erro: "no CUDA-capable device is detected"
**Solução:** Adicione `LD_LIBRARY_PATH`:
```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH php seu_script.php
```

### Erro: "CUDA out of memory"
**Solução:** Use `toCpu()` para liberar memória GPU:
```php
$a->toCpu();  // Move de volta para RAM
$a->free_device();  // Libera memória GPU
```

### Tempos ainda lentos (262ms vs 1ms)
**Problema:** Tensor não está em GPU
**Solução:**
```php
$a->toGpu();  // Move ANTES das operações
$a->add($b);  // Agora é rápido!
```

### Verificar GPU ativa
```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH ZMATRIX_GPU_DEBUG=1 php -r "
use ZMatrix\ZTensor;
\$a = ZTensor::random([1000000]);
\$a->toGpu();
\$a->relu();
" 2>&1 | grep gpu
```

Esperado: `[zmatrix][gpu] relu n=1000000`

## Performance Esperado

```
Operação          CPU (1M)    GPU (1M)    Speedup
─────────────────────────────────────────────────
add               2.5 ms      0.1 ms      25x
sub               2.5 ms      0.1 ms      25x
mul               2.5 ms      0.1 ms      25x
relu              1.8 ms      0.1 ms      18x
sigmoid           8.0 ms      0.5 ms      16x
tanh              8.5 ms      0.5 ms      17x
exp               5.0 ms      0.3 ms      17x
```

## Próximos Passos (Opcional)

1. **Compilar com rpath**: Para não depender de `LD_LIBRARY_PATH`
   ```bash
   ./configure --with-cuda-rpath
   make
   make install
   ```

2. **Usar tensores 100% GPU**: Para operações muito grandes
   ```php
   $result = $gpu_a->dot($gpu_b);  // Resultado na GPU
   echo $result->isOnGpu() ? "GPU" : "CPU";
   ```

3. **Medir consumo de GPU**:
   ```bash
   watch -n 0.5 nvidia-smi
   ```
   Abra outro terminal enquanto o script roda

## Verificação Final

```bash
#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv,noheader

echo -e "\n=== ZMatrix GPU Test ==="
php test_gpu_ops.php

echo -e "\n=== GPU vs CPU ==="
php test_gpu_vs_cpu.php
```

Salve como `test_gpu.sh` e execute:
```bash
chmod +x test_gpu.sh
./test_gpu.sh
```
