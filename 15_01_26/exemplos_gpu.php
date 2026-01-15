#!/usr/bin/env php
<?php
/**
 * Exemplos práticos de uso de GPU com ZMatrix
 * 
 * Execute com:
 * LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH php exemplos_gpu.php
 */

use ZMatrix\ZTensor;

echo "╔════════════════════════════════════════════════════════════╗\n";
echo "║      ZMatrix GPU - Exemplos Práticos                      ║\n";
echo "╚════════════════════════════════════════════════════════════╝\n\n";

// Exemplo 1: Rede Neural Simples na GPU
echo "═══════════════════════════════════════════════════════════════\n";
echo "EXEMPLO 1: Operações de Rede Neural na GPU\n";
echo "═══════════════════════════════════════════════════════════════\n\n";

class SimpleNeuralNetGPU {
    private $w1, $b1, $w2, $b2;
    private $batch_size = 32;
    private $hidden_size = 64;
    private $input_size = 784;
    private $output_size = 10;
    
    public function __construct() {
        // Inicializar pesos
        $this->w1 = ZTensor::random([$this->input_size, $this->hidden_size], -0.1, 0.1);
        $this->b1 = ZTensor::random([$this->hidden_size]);
        $this->w2 = ZTensor::random([$this->hidden_size, $this->output_size], -0.1, 0.1);
        $this->b2 = ZTensor::random([$this->output_size]);
        
        // Mover para GPU
        $this->w1->toGpu();
        $this->b1->toGpu();
        $this->w2->toGpu();
        $this->b2->toGpu();
    }
    
    public function forward($x) {
        // x está na GPU já
        $h = ZTensor::arr($x);
        
        // Forward pass
        $h->sigmoid();  // Ativação
        
        return $h;
    }
}

$nn = new SimpleNeuralNetGPU();
echo "✅ Rede neural criada na GPU\n";
echo "   Pesos: (784→64→10)\n";
echo "   Tensores movidos para VRAM\n\n";

// Exemplo 2: Processamento de Batch
echo "═══════════════════════════════════════════════════════════════\n";
echo "EXEMPLO 2: Processamento de Batch na GPU\n";
echo "═══════════════════════════════════════════════════════════════\n\n";

$batch_size = 1024;
$features = 512;

echo "Criando batch de dados: ($batch_size, $features)\n";
$data = ZTensor::random([$batch_size * $features], -1.0, 1.0);
$data->toGpu();

echo "Processando normalizações na GPU:\n";

// Normalização Z-score
$mean = 0.0;
$std = 1.0;

$t0 = microtime(true);
for ($i = 0; $i < 10; $i++) {
    $batch = ZTensor::arr($data);
    $batch->sub($mean);
    $batch->mul(1.0 / $std);
}
$t1 = microtime(true);

echo sprintf("  10 normalizações: %.2f ms (avg: %.2f ms/op)\n", 
    ($t1 - $t0) * 1000, 
    ($t1 - $t0) * 1000 / 10);

echo "  ✅ Processamento de batch concluído\n\n";

// Exemplo 3: Data Augmentation
echo "═══════════════════════════════════════════════════════════════\n";
echo "EXEMPLO 3: Data Augmentation na GPU\n";
echo "═══════════════════════════════════════════════════════════════\n\n";

function apply_augmentation($image_gpu) {
    $aug = ZTensor::arr($image_gpu);
    
    // Rotação simulada (operações element-wise)
    $noise = ZTensor::random($image_gpu->shape(), -0.05, 0.05);
    $noise->toGpu();
    $aug->add($noise);
    
    // Clipping
    $aug->relu();  // Remove negativos
    
    return $aug;
}

$img = ZTensor::random([224, 224, 3], 0.0, 1.0);
$img->toGpu();

echo "Aplicando augmentação a 100 imagens 224x224x3:\n";
$t0 = microtime(true);
for ($i = 0; $i < 100; $i++) {
    apply_augmentation($img);
}
$t1 = microtime(true);

echo sprintf("  Tempo total: %.2f ms (%.2f ms/imagem)\n", 
    ($t1 - $t0) * 1000,
    ($t1 - $t0) * 1000 / 100);
echo "  ✅ Augmentation concluído\n\n";

// Exemplo 4: Cálculos de Perda
echo "═══════════════════════════════════════════════════════════════\n";
echo "EXEMPLO 4: Cálculo de Funções de Perda\n";
echo "═══════════════════════════════════════════════════════════════\n\n";

function mse_loss($pred_gpu, $target_gpu) {
    $diff = ZTensor::arr($pred_gpu);
    $diff->sub($target_gpu);
    $diff->mul($diff);  // Elevar ao quadrado
    return $diff->sumtotal() / $diff->size();
}

function cross_entropy_approx($logits_gpu) {
    // Approximation (sem softmax completo)
    $ce = ZTensor::arr($logits_gpu);
    $ce->exp();
    return $ce->sumtotal();
}

$pred = ZTensor::random([1000], -1.0, 1.0);
$target = ZTensor::random([1000], 0.0, 1.0);

$pred->toGpu();
$target->toGpu();

$t0 = microtime(true);
$loss = mse_loss($pred, $target);
$t1 = microtime(true);

echo sprintf("MSE Loss (1000 elementos): %.6f (tempo: %.3f ms)\n", 
    $loss, ($t1 - $t0) * 1000);

$t0 = microtime(true);
$ce = cross_entropy_approx($pred);
$t1 = microtime(true);

echo sprintf("Cross-Entropy (1000 elementos): %.6f (tempo: %.3f ms)\n", 
    $ce, ($t1 - $t0) * 1000);
echo "  ✅ Cálculos de perda concluído\n\n";

// Exemplo 5: Comparação CPU vs GPU
echo "═══════════════════════════════════════════════════════════════\n";
echo "EXEMPLO 5: Benchmark CPU vs GPU\n";
echo "═══════════════════════════════════════════════════════════════\n\n";

$sizes = [100_000, 1_000_000, 10_000_000];

echo "Operação: add()\n";
echo str_pad("Size", 15) . str_pad("CPU (ms)", 15) . str_pad("GPU (ms)", 15) . "Speedup\n";
echo str_repeat("─", 60) . "\n";

foreach ($sizes as $size) {
    $a_orig = ZTensor::random([$size], -1.0, 1.0);
    $b = ZTensor::random([$size], -1.0, 1.0);
    
    // CPU
    $a_cpu = ZTensor::arr($a_orig);
    $t0 = microtime(true);
    for ($i = 0; $i < 5; $i++) $a_cpu->add($b);
    $t1 = microtime(true);
    $cpu_ms = ($t1 - $t0) * 1000 / 5;
    
    // GPU
    $a_gpu = ZTensor::arr($a_orig);
    $a_gpu->toGpu();
    $b_gpu = ZTensor::arr($b);
    $b_gpu->toGpu();
    $t0 = microtime(true);
    for ($i = 0; $i < 5; $i++) $a_gpu->add($b_gpu);
    $t1 = microtime(true);
    $gpu_ms = ($t1 - $t0) * 1000 / 5;
    
    $speedup = $cpu_ms / $gpu_ms;
    
    printf("%s %s %s %.1fx\n", 
        str_pad(number_format($size), 15),
        str_pad(number_format($cpu_ms, 2), 15),
        str_pad(number_format($gpu_ms, 2), 15),
        $speedup
    );
}
echo "\n  ✅ Benchmark concluído\n\n";

// Exemplo 6: Pipeline Completo
echo "═══════════════════════════════════════════════════════════════\n";
echo "EXEMPLO 6: Pipeline ML Completo\n";
echo "═══════════════════════════════════════════════════════════════\n\n";

class GPUPipeline {
    public function run() {
        echo "Pipeline: Load → Preprocess → Augment → Normalize → Compute\n";
        echo str_repeat("─", 60) . "\n";
        
        // Load
        $t_start = microtime(true);
        $data = ZTensor::random([1000, 784], 0.0, 1.0);
        $data->toGpu();
        $t = microtime(true) - $t_start;
        echo sprintf("[1/5] Load data:      %.2f ms\n", $t * 1000);
        
        // Preprocess
        $t_start = microtime(true);
        $processed = ZTensor::arr($data);
        $processed->mul(255.0);  // Scale
        $t = microtime(true) - $t_start;
        echo sprintf("[2/5] Preprocess:     %.2f ms\n", $t * 1000);
        
        // Augment
        $t_start = microtime(true);
        for ($i = 0; $i < 10; $i++) {
            $aug = ZTensor::arr($processed);
            $aug->relu();
        }
        $t = microtime(true) - $t_start;
        echo sprintf("[3/5] Augmentation:   %.2f ms (10x)\n", $t * 1000);
        
        // Normalize
        $t_start = microtime(true);
        for ($i = 0; $i < 10; $i++) {
            $norm = ZTensor::arr($processed);
            $norm->mul(1.0 / 255.0);  // Normalize to [0,1]
        }
        $t = microtime(true) - $t_start;
        echo sprintf("[4/5] Normalize:      %.2f ms (10x)\n", $t * 1000);
        
        // Compute
        $t_start = microtime(true);
        $result = ZTensor::arr($processed);
        $result->sigmoid();
        $loss = $result->sumtotal();
        $t = microtime(true) - $t_start;
        echo sprintf("[5/5] Compute loss:   %.2f ms\n", $t * 1000);
        
        echo "─────────────────────────────────────────────────────────\n";
        echo "✅ Pipeline completo executado na GPU\n";
    }
}

$pipeline = new GPUPipeline();
$pipeline->run();

echo "\n";
echo "═══════════════════════════════════════════════════════════════\n";
echo "✅ TODOS OS EXEMPLOS EXECUTADOS COM SUCESSO!\n";
echo "═══════════════════════════════════════════════════════════════\n\n";

echo "📝 Resumo de Boas Práticas:\n";
echo "1. Mova tensores para GPU ANTES das operações: \$t->toGpu()\n";
echo "2. Mantenha tensores na GPU durante múltiplas operações\n";
echo "3. Use ZMATRIX_GPU_DEBUG=1 para verificar execução\n";
echo "4. Sempre use LD_LIBRARY_PATH=/usr/lib/wsl/lib:...\n";
echo "5. Monitore GPU com: nvidia-smi\n";
echo "\n";
