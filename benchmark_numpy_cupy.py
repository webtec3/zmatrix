#!/usr/bin/env python3
"""
NumPy + CuPy Benchmark Suite
Compara performance de CPU (NumPy) vs GPU (CuPy)
"""

import numpy as np
import time
import json
import sys
from pathlib import Path

# Tentar importar CuPy (GPU)
CUPY_AVAILABLE = False
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy available - GPU benchmarks will run\n")
except ImportError:
    print("⚠️  CuPy not available - GPU benchmarks skipped")
    print("   Install with: pip install cupy-cuda-12x\n")

# Configuração
SEED = 42
RESULTS = {}

def benchmark(name, func, iterations=5):
    """Execute benchmark com múltiplas iterações"""
    times = []
    
    for _ in range(iterations):
        start = time.time()
        result = func()
        end = time.time()
        times.append((end - start) * 1000)  # ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        'name': name,
        'avg_ms': float(avg_time),
        'std_ms': float(std_time),
        'min_ms': float(min(times)),
        'max_ms': float(max(times))
    }

print("╔════════════════════════════════════════════════════════════════╗")
print("║   NumPy + CuPy Benchmark Suite (Python)                       ║")
print("╚════════════════════════════════════════════════════════════════╝\n")

# ─── TESTE 1: CRIAÇÃO DE TENSORES ───
print("═══════════════════════════════════════════════════════════════")
print("TEST 1: Creation and Initialization")
print("═══════════════════════════════════════════════════════════════\n")

# NumPy Random
result = benchmark(
    "NumPy Random [1M]",
    lambda: np.random.uniform(-1, 1, 1_000_000)
)
print(f"NumPy random:  {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['creation_numpy_random'] = result

# NumPy Zeros
result = benchmark(
    "NumPy Zeros [1M]",
    lambda: np.zeros(1_000_000)
)
print(f"NumPy zeros:   {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['creation_numpy_zeros'] = result

# NumPy Ones
result = benchmark(
    "NumPy Ones [1M]",
    lambda: np.ones(1_000_000)
)
print(f"NumPy ones:    {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['creation_numpy_ones'] = result

if CUPY_AVAILABLE:
    print()
    
    # CuPy Random
    result = benchmark(
        "CuPy Random [1M]",
        lambda: cp.random.uniform(-1, 1, 1_000_000)
    )
    print(f"CuPy random:   {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['creation_cupy_random'] = result
    
    # CuPy Zeros
    result = benchmark(
        "CuPy Zeros [1M]",
        lambda: cp.zeros(1_000_000)
    )
    print(f"CuPy zeros:    {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['creation_cupy_zeros'] = result
    
    # CuPy Ones
    result = benchmark(
        "CuPy Ones [1M]",
        lambda: cp.ones(1_000_000)
    )
    print(f"CuPy ones:     {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['creation_cupy_ones'] = result

# ─── TESTE 2: OPERAÇÕES ARITMÉTICAS ───
print("\n═══════════════════════════════════════════════════════════════")
print("TEST 2: Arithmetic Operations [5M elements]")
print("═══════════════════════════════════════════════════════════════\n")

size = 5_000_000
np.random.seed(SEED)
a_np = np.random.uniform(-1, 1, size).astype(np.float32)
b_np = np.random.uniform(-1, 1, size).astype(np.float32)

# NumPy Add
result = benchmark(
    "NumPy Add",
    lambda: a_np + b_np,
    iterations=10
)
print(f"NumPy add:     {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['arithmetic_numpy_add'] = result

# NumPy Sub
result = benchmark(
    "NumPy Sub",
    lambda: a_np - b_np,
    iterations=10
)
print(f"NumPy sub:     {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['arithmetic_numpy_sub'] = result

# NumPy Mul
result = benchmark(
    "NumPy Mul",
    lambda: a_np * b_np,
    iterations=10
)
print(f"NumPy mul:     {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['arithmetic_numpy_mul'] = result

# NumPy Divide
result = benchmark(
    "NumPy Divide",
    lambda: a_np / (b_np + 1e-10),
    iterations=10
)
print(f"NumPy div:     {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['arithmetic_numpy_div'] = result

if CUPY_AVAILABLE:
    print()
    
    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)
    
    # CuPy Add
    result = benchmark(
        "CuPy Add",
        lambda: a_cp + b_cp,
        iterations=10
    )
    print(f"CuPy add:      {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['arithmetic_cupy_add'] = result
    
    # CuPy Sub
    result = benchmark(
        "CuPy Sub",
        lambda: a_cp - b_cp,
        iterations=10
    )
    print(f"CuPy sub:      {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['arithmetic_cupy_sub'] = result
    
    # CuPy Mul
    result = benchmark(
        "CuPy Mul",
        lambda: a_cp * b_cp,
        iterations=10
    )
    print(f"CuPy mul:      {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['arithmetic_cupy_mul'] = result
    
    # CuPy Divide
    result = benchmark(
        "CuPy Divide",
        lambda: a_cp / (b_cp + 1e-10),
        iterations=10
    )
    print(f"CuPy div:      {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['arithmetic_cupy_div'] = result

# ─── TESTE 3: ATIVAÇÕES ───
print("\n═══════════════════════════════════════════════════════════════")
print("TEST 3: Activation Functions [5M elements]")
print("═══════════════════════════════════════════════════════════════\n")

# NumPy ReLU
result = benchmark(
    "NumPy ReLU",
    lambda: np.maximum(a_np, 0),
    iterations=10
)
print(f"NumPy ReLU:    {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['activation_numpy_relu'] = result

# NumPy Sigmoid
result = benchmark(
    "NumPy Sigmoid",
    lambda: 1 / (1 + np.exp(-a_np)),
    iterations=5
)
print(f"NumPy Sigmoid: {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['activation_numpy_sigmoid'] = result

# NumPy Tanh
result = benchmark(
    "NumPy Tanh",
    lambda: np.tanh(a_np),
    iterations=10
)
print(f"NumPy Tanh:    {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['activation_numpy_tanh'] = result

# NumPy Softmax (1D)
a_softmax = np.random.uniform(-1, 1, 10_000).astype(np.float32)
result = benchmark(
    "NumPy Softmax",
    lambda: np.exp(a_softmax - np.max(a_softmax)) / np.sum(np.exp(a_softmax - np.max(a_softmax))),
    iterations=10
)
print(f"NumPy Softmax: {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['activation_numpy_softmax'] = result

if CUPY_AVAILABLE:
    print()
    
    # CuPy ReLU
    result = benchmark(
        "CuPy ReLU",
        lambda: cp.maximum(a_cp, 0),
        iterations=10
    )
    print(f"CuPy ReLU:     {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['activation_cupy_relu'] = result
    
    # CuPy Sigmoid
    result = benchmark(
        "CuPy Sigmoid",
        lambda: 1 / (1 + cp.exp(-a_cp)),
        iterations=5
    )
    print(f"CuPy Sigmoid:  {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['activation_cupy_sigmoid'] = result
    
    # CuPy Tanh
    result = benchmark(
        "CuPy Tanh",
        lambda: cp.tanh(a_cp),
        iterations=10
    )
    print(f"CuPy Tanh:     {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['activation_cupy_tanh'] = result
    
    # CuPy Softmax
    a_cp_softmax = cp.asarray(a_softmax)
    result = benchmark(
        "CuPy Softmax",
        lambda: cp.exp(a_cp_softmax - cp.max(a_cp_softmax)) / cp.sum(cp.exp(a_cp_softmax - cp.max(a_cp_softmax))),
        iterations=10
    )
    print(f"CuPy Softmax:  {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['activation_cupy_softmax'] = result

# ─── TESTE 4: ÁLGEBRA LINEAR ───
print("\n═══════════════════════════════════════════════════════════════")
print("TEST 4: Linear Algebra")
print("═══════════════════════════════════════════════════════════════\n")

# Matrix Multiplication [1000 x 1000]
size = 1000
A = np.random.uniform(-1, 1, (size, size)).astype(np.float32)
B = np.random.uniform(-1, 1, (size, size)).astype(np.float32)

result = benchmark(
    "NumPy MatMul [1Kx1K]",
    lambda: np.matmul(A, B),
    iterations=3
)
print(f"NumPy MatMul [1Kx1K]: {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['linalg_numpy_matmul_1k'] = result

# Dot Product [1M elements]
vec1 = np.random.uniform(-1, 1, 1_000_000).astype(np.float32)
vec2 = np.random.uniform(-1, 1, 1_000_000).astype(np.float32)

result = benchmark(
    "NumPy Dot [1M]",
    lambda: np.dot(vec1, vec2),
    iterations=10
)
print(f"NumPy Dot [1M]:       {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['linalg_numpy_dot_1m'] = result

if CUPY_AVAILABLE:
    print()
    
    A_cp = cp.asarray(A)
    B_cp = cp.asarray(B)
    
    result = benchmark(
        "CuPy MatMul [1Kx1K]",
        lambda: cp.matmul(A_cp, B_cp),
        iterations=3
    )
    print(f"CuPy MatMul [1Kx1K]:  {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['linalg_cupy_matmul_1k'] = result
    
    vec1_cp = cp.asarray(vec1)
    vec2_cp = cp.asarray(vec2)
    
    result = benchmark(
        "CuPy Dot [1M]",
        lambda: cp.dot(vec1_cp, vec2_cp),
        iterations=10
    )
    print(f"CuPy Dot [1M]:        {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['linalg_cupy_dot_1m'] = result

# ─── TESTE 5: ESTATÍSTICAS ───
print("\n═══════════════════════════════════════════════════════════════")
print("TEST 5: Statistics [5M elements]")
print("═══════════════════════════════════════════════════════════════\n")

# NumPy Sum
result = benchmark(
    "NumPy Sum",
    lambda: np.sum(a_np),
    iterations=10
)
print(f"NumPy Sum:     {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['stats_numpy_sum'] = result

# NumPy Mean
result = benchmark(
    "NumPy Mean",
    lambda: np.mean(a_np),
    iterations=10
)
print(f"NumPy Mean:    {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['stats_numpy_mean'] = result

# NumPy Std
result = benchmark(
    "NumPy Std",
    lambda: np.std(a_np),
    iterations=10
)
print(f"NumPy Std:     {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['stats_numpy_std'] = result

# NumPy Min/Max
result = benchmark(
    "NumPy Min/Max",
    lambda: (np.min(a_np), np.max(a_np)),
    iterations=10
)
print(f"NumPy Min/Max: {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
RESULTS['stats_numpy_minmax'] = result

if CUPY_AVAILABLE:
    print()
    
    # CuPy Sum
    result = benchmark(
        "CuPy Sum",
        lambda: cp.sum(a_cp),
        iterations=10
    )
    print(f"CuPy Sum:      {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['stats_cupy_sum'] = result
    
    # CuPy Mean
    result = benchmark(
        "CuPy Mean",
        lambda: cp.mean(a_cp),
        iterations=10
    )
    print(f"CuPy Mean:     {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['stats_cupy_mean'] = result
    
    # CuPy Std
    result = benchmark(
        "CuPy Std",
        lambda: cp.std(a_cp),
        iterations=10
    )
    print(f"CuPy Std:      {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['stats_cupy_std'] = result
    
    # CuPy Min/Max
    result = benchmark(
        "CuPy Min/Max",
        lambda: (cp.min(a_cp), cp.max(a_cp)),
        iterations=10
    )
    print(f"CuPy Min/Max:  {result['avg_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    RESULTS['stats_cupy_minmax'] = result

# ─── SALVAR RESULTADOS ───
print("\n═══════════════════════════════════════════════════════════════")
print("Saving results...")
print("═══════════════════════════════════════════════════════════════\n")

output_file = Path(__file__).parent / "benchmark_numpy_cupy_results.json"
with open(output_file, 'w') as f:
    json.dump(RESULTS, f, indent=2)

print(f"✅ Results saved to: {output_file}")
print(f"   Total benchmarks: {len(RESULTS)}")

print("\n" + "═" * 65)
print("NumPy + CuPy benchmarks completed!")
print("═" * 65 + "\n")
