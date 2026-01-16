# DIA 5 - PROFILING & ANOMALY INVESTIGATION

## Status: PARTIALLY COMPLETE

### 1. Profiling com Perf
- **Status**: ❌ Não disponível
- **Motivo**: `perf` não instalado no WSL
- **Alternativa Usada**: Valgrind para análise de memória

### 2. Valgrind Memory Leak Detection
- **Status**: ✅ CONCLUÍDO
- **Resultado**: 
  - definitely lost: 319,264 bytes em 980 blocks
  - indirectly lost: 2,160,112 bytes
  - possibly lost: 6,004 bytes
  - still reachable: 86,805 bytes
- **Conclusão**: Leaks are in PHP core/libraries, not ZMatrix extension

### 3. Investigação Anomalia mean()/sum()
- **Status**: ⚠️ ANOMALIA IDENTIFICADA, NÃO CORRIGIDA
- **Problema Descoberto**: 
  - `sumtotal()` retorna aproximadamente 52% do valor esperado
  - Exemplos:
    * ZTensor([100], 2.5) → sum = 130 (esperado: 250)
    * ZTensor([1000], 0.5) → sum = 250 (esperado: 500)
    * ZTensor([1024], 1.0) → sum = 512 (esperado: 1024)
  
- **Análise Realizada**:
  1. ✅ Verificado `full()` - cria dados corretamente
  2. ✅ Verificado `toArray()` - retorna 100% dos dados corretos
  3. ✅ Soma manual em PHP - resulta em 250 (correto!)
  4. ✅ `sumtotal()` em C++ - retorna 130 (ERRADO)
  5. ❌ Testado desabilitando SIMD - ainda retorna 130
  
- **Conclusão**: Problema está na função `sum()` em C++, NÃO em SIMD/dados

### 4. Cache Optimization
- **Status**: 🔄 AGUARDANDO correção da anomalia

### 5. Documentação Final
- **Status**: 🔄 AGUARDANDO conclusão de todas as tarefas

---

## Próximas Etapas (DIA 5 continuação)

1. **Debugar sum()**: Investigar por que retorna 52% (aproximadamente)
2. **Revisar conversão de tipos**: Possível problema float → double
3. **Executar teste com diferentes valores para padrão**
4. **Corrigir e revalidar**
5. **Documentação final completa**

## Observações Importantes

- A anomalia é **crítica** mas **isolada** na função sum()
- NÃO afeta:
  - Compilação (0 warnings, 0 errors)
  - Testes gerais (15/15 passam)
  - Race conditions (0 detectadas)
  - Memoria (leaks em sys libs, não ZMatrix)

- **Impacto**: mean(), std() que dependem de sum() também afetados

