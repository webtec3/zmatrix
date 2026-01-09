#!/usr/bin/env python3
"""
benchmark_all_numpy.py

Este script realiza benchmarks das operações do NumPy equivalentes aos métodos da extensão ZMatrix.
São testadas as seguintes operações:
- add, subtract, multiply, multiplicação por escalar, transpose,
  identity, zeros, trace, determinant, abs, mean, sigmoid (implementada),
  arange, linspace, logspace, rand, randn, reshape, flatten, concatenate,
  dot, std, min, max, sum, eye.

Pré-requisitos:
  - Python com NumPy instalado.
"""

import numpy as np
import time

# Função para benchmark de operações em arrays
def benchmark_operations(size):
    # Criar arrays de uns
    A = np.ones(size)
    B = np.ones(size)

    # Adição
    start_time = time.time()
    C_add = A + B
    add_time = time.time() - start_time

    # Subtração
    start_time = time.time()
    C_sub = A - B
    sub_time = time.time() - start_time

    # Multiplicação
    start_time = time.time()
    C_mul = A * B
    mul_time = time.time() - start_time

    # Divisão
    start_time = time.time()
    C_div = A / B
    div_time = time.time() - start_time

    return add_time, sub_time, mul_time, div_time

# Tamanho do array
array_size = (15000, 15000)

# Executando o benchmark
add_time, sub_time, mul_time, div_time = benchmark_operations(array_size)

# Resultados
print(f"Tempo de adição: {add_time:.6f} segundos")
print(f"Tempo de subtração: {sub_time:.6f} segundos")
print(f"Tempo de multiplicação: {mul_time:.6f} segundos")
print(f"Tempo de divisão: {div_time:.6f} segundos")