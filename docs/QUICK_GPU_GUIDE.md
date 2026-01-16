# Quick Start Guide: GPU Methods in ZMatrix

## The Four GPU Methods

### 1. Move to GPU
```php
$tensor = ZTensor::arr([[1, 2], [3, 4]]);
$tensor->toGpu();  // Now on GPU!
```

### 2. Move back to CPU
```php
$tensor->toCpu();  // Back to CPU
```

### 3. Check if on GPU
```php
if ($tensor->isOnGpu()) {
    echo "Using GPU acceleration!";
}
```

### 4. Free GPU Memory
```php
$tensor->freeDevice();  // Explicitly free, move to CPU
```

---

## Real-World Example: Neural Network Forward Pass

```php
// Load model tensors
$weights = ZTensor::arr($weights_data);
$bias = ZTensor::arr($bias_data);

// Move to GPU for computation
$weights->toGpu();
$bias->toGpu();

// Process batch
for ($batch = 0; $batch < $num_batches; $batch++) {
    // Operations stay on GPU (fast!)
    $input = $batches[$batch];
    $input->toGpu();
    
    // All operations use GPU
    $output = $input->matmul($weights);
    $output->add($bias);
    $output->relu();
    
    // Get results from GPU
    $output->toCpu();
    $results[] = $output->toArray();
}
```

---

## Performance Tips

1. **Keep tensors on GPU for multiple operations** - Avoid H2D/D2H overhead
2. **Use GPU for large tensors** - Benefit visible with 200k+ elements
3. **Batch operations** - Process multiple tensors together on GPU
4. **Free memory when done** - Call `freeDevice()` for large tensors

---

## Running Tests

```bash
# Run all GPU tests
php test_gpu_vs_cpu.php

# With debug output
ZMATRIX_GPU_DEBUG=1 php test_gpu_vs_cpu.php
```

---

## Common Errors & Solutions

| Error | Solution |
|-------|----------|
| "CUDA support not available" | CUDA not installed or not detected. Run on a GPU system. |
| GPU operations slower than CPU | Tensor too small. GPU benefits visible with 200k+ elements. |
| Out of GPU memory | Call `freeDevice()` to free memory from previous tensors |
| Inconsistent results | Move tensors to same device (both GPU or both CPU) before operations |

---

## Method Signatures

```php
/**
 * Move tensor to GPU memory
 * @return ZTensor self
 * @throws RuntimeException if CUDA unavailable
 */
public function toGpu(): ZTensor

/**
 * Move tensor back to CPU memory
 * @return ZTensor self
 * @throws RuntimeException if CUDA unavailable
 */
public function toCpu(): ZTensor

/**
 * Check if tensor is on GPU
 * @return bool true if on GPU, false if on CPU
 */
public function isOnGpu(): bool

/**
 * Free GPU memory (moves to CPU)
 * @return void
 */
public function freeDevice(): void
```

---

## Frequently Asked Questions

**Q: Do I need to call both toGpu() and toCpu()?**
A: No. You can keep tensors on GPU for multiple operations, then call toCpu() at the end.

**Q: What happens if I mix GPU and CPU tensors?**
A: Operations on GPU tensors work with GPU data. Operations on CPU tensors work with CPU data.

**Q: Is GPU faster for small tensors?**
A: Not always. GPU overhead is high for small tensors (< 200k elements). CPU is often faster.

**Q: How much memory does the GPU need?**
A: Depends on tensor size. Each float = 4 bytes. For a 1000×1000 tensor = 4MB. GPUs typically have 2-12GB.

**Q: Can I use multiple GPUs?**
A: Current implementation uses single GPU. Multi-GPU support coming in future version.

**Q: Will my code break if CUDA is unavailable?**
A: No, but GPU methods will throw RuntimeException. Wrap in try-catch or check `isOnGpu()` result.

---

## Testing Your GPU Setup

```php
<?php
use ZMatrix\ZTensor;

// Test GPU availability
try {
    $t = ZTensor::arr([[1.0]]);
    $t->toGpu();
    
    echo "✅ GPU is working!\n";
    echo "Speedup: " . (CUDA_SPEEDUP_FACTOR) . "x for large tensors\n";
    
    $t->toCpu();
} catch (Exception $e) {
    echo "❌ GPU error: " . $e->getMessage() . "\n";
}
?>
```

---

**For detailed test results and benchmarks, see:** [test_gpu_vs_cpu.php](test_gpu_vs_cpu.php)

**For API documentation, see:** [stubs/ZTensor.php](stubs/ZTensor.php)
