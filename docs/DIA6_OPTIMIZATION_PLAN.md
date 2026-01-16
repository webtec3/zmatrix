# DIA 6 - EXTENDED SIMD OPTIMIZATION PLAN

**Data**: 10 de Janeiro de 2026  
**Objetivo**: Expandir otimizações SIMD para operações adicionais

---

## 🎯 Operações Candidatas para Otimização

### Prioridade ALTA (Impacto significativo)

1. **divide()** - Elemento-a-elemento
   - Instruções: `_mm256_div_ps()`
   - Speedup esperado: ~7-8x (similar a add/mul)
   - Uso: Operações de normalização
   - Status: ⏳ TODO

2. **scalar_multiply()** - Broadcast + multiplicação
   - Instruções: `_mm256_set1_ps()` + `_mm256_mul_ps()`
   - Speedup esperado: ~7-8x
   - Uso: Muito comum em processamento
   - Status: ⏳ TODO

3. **scalar_divide()** - Broadcast + divisão
   - Instruções: `_mm256_set1_ps()` + `_mm256_div_ps()`
   - Speedup esperado: ~7-8x
   - Uso: Normalização
   - Status: ⏳ TODO

4. **scalar_add()** e **scalar_subtract()**
   - Instruções: `_mm256_set1_ps()` + `_mm256_add_ps()` / `_mm256_sub_ps()`
   - Speedup esperado: ~7-8x
   - Status: ⏳ TODO

### Prioridade MÉDIA (Impacto moderado)

5. **leaky_relu()** - Com parâmetro alpha
   - Instruções: `_mm256_cmp_ps()` + `_mm256_blendv_ps()`
   - Speedup esperado: ~3-5x
   - Status: ⏳ TODO

6. **pow()** - Exponenciação
   - Desafio: Não há `_mm256_pow_ps()` nativa
   - Alternativa: Usar exponenciação com LUT (Lookup Table)
   - Speedup esperado: ~2-4x
   - Status: ⏳ RESEARCH NEEDED

7. **exp()** - Exponencial
   - Instruções: Approximate exp com polinômios ou LUT
   - Speedup esperado: ~3-5x
   - Status: ⏳ TODO

8. **log()** - Logaritmo
   - Instruções: Approximate log com polinômios
   - Speedup esperado: ~3-5x
   - Status: ⏳ TODO

### Prioridade BAIXA (Uso raro)

9. **clip()** - Clamp para outro tensor
   - Instruções: `_mm256_max_ps()` + `_mm256_min_ps()`
   - Speedup esperado: ~3-4x
   - Status: ⏳ TODO (se houver tempo)

10. **std()** - Desvio padrão
    - Desafio: Operação de redução complexa
    - Pode melhorar com SIMD para quadrados
    - Status: ⏳ RESEARCH NEEDED

---

## 📋 Implementação por Fase

### Fase 1: Operações Escalares (FÁCIL - Alta Prioridade)
```cpp
// Padrão para estas operações:
// 1. Criar vetor com scalar: _mm256_set1_ps(scalar)
// 2. Aplicar operação em paralelo
// 3. Armazenar resultado
// 4. Processar tail loop manualmente
```

**Funções**: scalar_add, scalar_subtract, scalar_multiply, scalar_divide

### Fase 2: Elemento-a-Elemento (MÉDIO)
```cpp
// Padrão:
// 1. Carregar dois vetores: _mm256_loadu_ps()
// 2. Aplicar operação: _mm256_div_ps(), etc
// 3. Armazenar: _mm256_storeu_ps()
// 4. Loop tail
```

**Funções**: divide

### Fase 3: Funções Matemáticas (DIFÍCIL)
```cpp
// Requer aproximação polinomial ou LUT
// Mais complexo, mas grande impacto em ML
```

**Funções**: exp, log, pow, leaky_relu

---

## 🧪 Testes Planejados

Para cada função otimizada:
1. Correctness test (comparar com versão scalar)
2. Performance benchmark (speedup measurement)
3. Edge case validation (NaN, Inf, denormalized)

---

## 📊 Objetivo Final

```
Operações Antes do DIA 6:
├─ add/mul/sub:         7.98x ✅
├─ relu/sigmoid/tanh:   3.61x ✅
├─ abs/sqrt:            3-7x  ✅
├─ min/max/sum:         3-4x  ✅
└─ scalar/divide:       1.0x  ⏳

Operações Depois do DIA 6 (ALVO):
├─ add/mul/sub:         7.98x ✅
├─ relu/sigmoid/tanh:   3.61x ✅
├─ abs/sqrt:            3-7x  ✅
├─ min/max/sum:         3-4x  ✅
├─ scalar/divide:       7-8x  ⏳ → NEW
├─ leaky_relu:          3-5x  ⏳ → NEW
└─ exp/log/pow:         3-5x  ⏳ → NEW
```

---

## 🚀 Próximos Passos

1. Implementar Fase 1 (scalar operations)
2. Criar benchmark suite
3. Implementar Fase 2 (divide)
4. Testar e validar
5. Considerar Fase 3 (se houver tempo)
6. Documentar resultados finais
