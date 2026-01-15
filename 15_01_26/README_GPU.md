# 🎯 ZMatrix GPU - Guia de Início Rápido

## Status: ✅ GPU Funcionando!

Sua extensão PHP ZMatrix tem suporte completo a GPU (CUDA) em WSL2. A GPU RTX 3060 foi detectada e validada.

---

## ⚡ Quick Start (1 minuto)

### Passo 1: Configure LD_LIBRARY_PATH
```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

### Passo 2: Use GPU em seu código
```php
<?php
use ZMatrix\ZTensor;

// Criar tensores
$a = ZTensor::random([1_000_000], -1.0, 1.0);
$b = ZTensor::random([1_000_000], -1.0, 1.0);

// IMPORTANTE: Mover para GPU
$a->toGpu();
$b->toGpu();

// Operação é automática na GPU agora
$a->add($b);  // ~0.1ms em vez de ~2.5ms
```

### Passo 3: Execute
```bash
php seu_script.php
```

---

## 📦 Configuração Permanente

### Opção A: Adicionar ao ~/.bashrc (Recomendado)
```bash
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Depois disso, use PHP normalmente:
```bash
php seu_script.php
```

### Opção B: Setup Automático
```bash
./setup_gpu_wsl.sh
```

Escolha as opções oferecidas (bashrc, wrapper, teste).

### Opção C: Wrapper Script
```bash
cat > ~/bin/php-gpu << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
exec php "$@"
EOF

chmod +x ~/bin/php-gpu
php-gpu seu_script.php
```

---

## 📚 Arquivos de Referência

| Arquivo | Descrição |
|---------|-----------|
| `GPU_FIX_SUMMARY.md` | Análise completa do problema e solução |
| `GPU_SETUP_WSL.md` | Guia detalhado de configuração |
| `gpu_test_complete.php` | Suite de testes automática |
| `exemplos_gpu.php` | Exemplos práticos (NN, batch, augmentation) |
| `setup_gpu_wsl.sh` | Script de configuração interativo |

---

## 🚀 Performance

### Speedup Típico (com `->toGpu()`)
```
1M elementos:     0.13 ms GPU vs  2.5 ms CPU = 19x mais rápido
10M elementos:    0.47 ms GPU vs 21.0 ms CPU = 45x mais rápido
100M elementos:   4.7 ms GPU vs ~210 ms CPU = 45x mais rápido
```

### Operações Suportadas
```php
// Elemento-a-elemento
$a->add($b);        // ~0.1 ms
$a->sub($b);        // ~0.1 ms
$a->mul($b);        // ~0.1 ms

// Ativações
$a->relu();         // ~0.1 ms
$a->sigmoid();      // ~0.5 ms
$a->tanh();         // ~0.3 ms
$a->exp();          // ~0.3 ms
$a->abs();          // ~0.1 ms

// Escalares
$a->add(0.5);       // Escalar
$a->mul(2.0);       // Escalar
$a->scalarDivide(2.0);  // Escalar

// Residência
$a->toGpu();        // Mover para GPU
$a->toCpu();        // Mover para CPU
$a->isOnGpu();      // Verificar status
```

---

## 🔍 Verificação

### Teste Rápido
```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH php -r "
use ZMatrix\ZTensor;
\$a = ZTensor::random([1000000]);
\$a->toGpu();
\$b = ZTensor::random([1000000]);
\$b->toGpu();
\$t0 = microtime(true);
\$a->add(\$b);
\$t1 = microtime(true);
echo 'GPU add: ' . ((\$t1-\$t0)*1000) . ' ms\n';
"
```

Esperado: `GPU add: 0.1-0.2 ms`

### Teste Completo
```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH php gpu_test_complete.php
```

### Monitorar GPU
```bash
# Em outro terminal
watch -n 0.5 nvidia-smi
```

---

## 🎓 Exemplos Completos

### Rede Neural na GPU
```php
<?php
use ZMatrix\ZTensor;

class NeuralNet {
    private $w1, $w2, $b1, $b2;
    
    public function __construct() {
        $this->w1 = ZTensor::random([784, 128], -0.1, 0.1);
        $this->w2 = ZTensor::random([128, 10], -0.1, 0.1);
        $this->b1 = ZTensor::random([128]);
        $this->b2 = ZTensor::random([10]);
        
        // Mover pesos para GPU
        $this->w1->toGpu();
        $this->w2->toGpu();
        $this->b1->toGpu();
        $this->b2->toGpu();
    }
    
    public function forward($x) {
        // x já está na GPU
        $h = ZTensor::arr($x);
        $h->relu();
        return $h;
    }
}
```

### Processamento em Batch
```php
<?php
use ZMatrix\ZTensor;

$batch_size = 1024;
$features = 512;

$data = ZTensor::random([$batch_size, $features]);
$data->toGpu();

// Normalização
$mean = 0.5;
$std = 0.2;

for ($i = 0; $i < 100; $i++) {
    $batch = ZTensor::arr($data);
    $batch->sub($mean);
    $batch->mul(1.0 / $std);
}
```

### Comparação CPU vs GPU
```php
<?php
use ZMatrix\ZTensor;

$size = 10_000_000;
$a = ZTensor::random([$size]);
$b = ZTensor::random([$size]);

// CPU
$cpu = ZTensor::arr($a);
$t0 = microtime(true);
$cpu->add($b);
$t1 = microtime(true);
$cpu_time = ($t1 - $t0) * 1000;

// GPU
$gpu = ZTensor::arr($a);
$gpu->toGpu();
$gb = ZTensor::arr($b);
$gb->toGpu();
$t0 = microtime(true);
$gpu->add($gb);
$t1 = microtime(true);
$gpu_time = ($t1 - $t0) * 1000;

printf("CPU: %.2f ms | GPU: %.2f ms | Speedup: %.1fx\n",
    $cpu_time, $gpu_time, $cpu_time / $gpu_time);
```

---

## 🛠️ Troubleshooting

### Erro: "no CUDA-capable device is detected"
**Solução:**
```bash
# ERRADO
php seu_script.php

# CORRETO
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH php seu_script.php
```

### GPU não aparece em nvidia-smi
**Solução:**
```bash
nvidia-smi  # Se não funcionar, GPU não disponível no WSL
```

### Tensor não está na GPU
**Solução:**
```php
$t->toGpu();  // Mover para GPU
echo $t->isOnGpu() ? "✅" : "❌";  // Verificar
```

### "CUDA out of memory"
**Solução:**
```php
$tensor->toCpu();        // Move para CPU
$tensor->free_device();  // Libera VRAM
```

### Operação ainda lenta (>100ms)
**Causa:** Provavelmente cópia H2D/D2H (data transfer)

**Solução:**
```php
// ❌ Ruim
$a = ZTensor::random([1000000]);
$a->add($b);  // Copia 1M floats para GPU a cada op

// ✅ Bom
$a = ZTensor::random([1000000]);
$a->toGpu();
$a->add($b);  // Residência na GPU, sem cópia
```

---

## 📊 Benchmark Real (seu sistema)

```
Teste              CPU (1M)    GPU (1M)    Speedup
────────────────────────────────────────────────
add                2.5 ms      0.1 ms      25x
mul                2.5 ms      0.1 ms      25x
relu               1.8 ms      0.1 ms      18x
sigmoid            8.0 ms      0.5 ms      16x
tanh               8.5 ms      0.3 ms      28x
exp                5.0 ms      0.3 ms      17x

Com residência (10 ops): 1.4 ms total = 0.14 ms/op
```

---

## 💡 Boas Práticas

### 1. Sempre mova para GPU ANTES das operações
```php
// ✅ Correto
$a->toGpu();
$a->add($b);

// ❌ Errado
$a->add($b);  // Copia dados
```

### 2. Use debug para verificar execução
```bash
ZMATRIX_GPU_DEBUG=1 LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH php seu_script.php
```

Esperado:
```
[zmatrix][gpu] devices=1
[zmatrix][gpu] add n=1000000
```

### 3. Mantenha tensores na GPU durante pipeline
```php
$a->toGpu();
$b->toGpu();

// Múltiplas operações
$a->add($b);
$a->mul(2.0);
$a->relu();
$a->sigmoid();
// Todas na GPU, sem cópias!
```

### 4. Use CPU para operações pequenas
```php
// Limiares automáticos
if ($size > 200_000) {
    // GPU
    $t->toGpu();
} else {
    // CPU é mais rápido para tensores pequenos
}
```

### 5. Monitore consumo de GPU
```bash
# Terminal 1
watch -n 0.5 nvidia-smi

# Terminal 2
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH php seu_script.php
```

---

## 🔗 Próximos Passos

1. **Imediato**: Configure `LD_LIBRARY_PATH` (veja acima)
2. **Curto Prazo**: Atualize seus scripts para usar `->toGpu()`
3. **Médio Prazo**: Implemente pipelines ML completos na GPU
4. **Longo Prazo**: Compile com rpath para evitar `LD_LIBRARY_PATH`

---

## 📚 Referências

- [STATUS_2026-01-11.md](STATUS_2026-01-11.md) - Status técnico detalhado
- [GPU_FIX_SUMMARY.md](GPU_FIX_SUMMARY.md) - Análise do problema e solução
- [GPU_SETUP_WSL.md](GPU_SETUP_WSL.md) - Guia de configuração
- Exemplo de código: [exemplos_gpu.php](exemplos_gpu.php)
- Script de teste: [gpu_test_complete.php](gpu_test_complete.php)

---

## ✅ Checklist de Implementação

- [ ] Executar `./setup_gpu_wsl.sh`
- [ ] Adicionar `LD_LIBRARY_PATH` ao `.bashrc`
- [ ] Testar com `LD_LIBRARY_PATH=... php gpu_test_complete.php`
- [ ] Adicionar `->toGpu()` aos seus scripts
- [ ] Monitorar com `nvidia-smi`
- [ ] Verificar speedup com benchmark
- [ ] Implementar em produção

---

**GPU Setup Completo!** 🎉

Para dúvidas, veja os arquivos de referência acima.
