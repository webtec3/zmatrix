# Implementação de Stub-Based Arginfo Generation - ZMatrix

## Status: ✅ IMPLEMENTADO E COMPILADO COM SUCESSO

### O que foi feito

Adotamos a abordagem moderna de **stub-based arginfo generation** do PHP core, baseado no PR do repositório [krakjoe/ort#6](https://github.com/krakjoe/ort/pull/6). Este é o método recomendado para extensões PHP 8.0+.

### Arquivos Modificados

1. **`composer.json`**
   - Adicionado: `nikic/php-parser: ^5.1` como dev dependency
   - Necessário para `gen_stub.php` funcionar

2. **`Makefile.fragments`**
   - Adicionado target: `make gen-stubs`
   - Regenera arginfo automaticamente a partir dos `.stub.php`
   - Simplifica o workflow de desenvolvimento

3. **`ztensor.stub.php`** (simplificado e otimizado)
   - Removido: `declare(strict_types=1)` 
   - Removido: `use` statements
   - Removido: tipos genéricos complexos (`array<int|float>`)
   - Resultado: Parser-friendly, compatível com `gen_stub.php`
   - 78 métodos definidos em sintaxe PHP legível

4. **`zmatrix.stub.php`** (existente)
   - Funciona perfeitamente com `gen_stub.php`
   - Define API procedural (30+ funções)

### Arquivos Gerados

#### `zmatrix_arginfo.h` (233 linhas)
```
Stub hash: ca9c2a0ac1924aea4959656787f08ddca5a1ad63
Contém:
- 30+ ZEND_ARG_INFO structs para funções procedurais
- Função entry array automática
- Totalmente compatível com C code existente
```

#### `ztensor_arginfo.h` (190 linhas)
```
Contém:
- Class entry para ZMatrix\ZTensor
- 78 method definitions
- Constructor, static methods, instance methods
- Gerado automaticamente com type hints corretos
```

### Compilação

```bash
cd ~/php-projetos/php-extension/zmatrix
make clean
make
# ✅ Build complete. Don't forget to run 'make test'.
```

Resultado: **SUCESSO** - 0 warnings, 0 errors

### Como Usar

#### 1. Após clonar o repositório
```bash
composer install  # Instala nikic/php-parser
make gen-stubs    # Gera zmatrix_arginfo.h e ztensor_arginfo.h
make clean && make && sudo make install
```

#### 2. Atualizando a API
Edite `.stub.php` com sintaxe PHP padrão:
```php
// ztensor.stub.php
public function meuMetodo(int $param1, string $param2, float $default = 1.5): array {}
```

Execute:
```bash
make gen-stubs  # Regenera arginfo automaticamente
make clean && make && sudo make install
```

#### 3. Integração no CI/CD
```bash
# Durante build:
composer install
make gen-stubs
make test
```

### Benefícios

| Antes | Depois |
|-------|--------|
| Manual C macros: `ZEND_BEGIN_ARG_INFO_EX` | PHP syntax: `public function ...` |
| Erro-prone | Automatizado |
| Hard to read | Legível |
| IDE stubs inconsistentes | Único source of truth |
| Documentação dispersa | API clara em .stub.php |

### Próximos Passos (Opcionais)

1. **Publicar Stubs no Packagist** (para IDE support)
   ```bash
   # Criar package zmatrix-stubs
   composer require --dev vendor/zmatrix-stubs
   # Desenvolvedores PHP obteriam autocompletion completo
   ```

2. **Integrar Generated Headers**
   - Use `#include "zmatrix_arginfo.h"` nos C files
   - Replace manual arginfo definitions
   - Cleanup: remover macros antigas

3. **Automatizar em CI**
   - GitHub Actions: verificar se stubs estão sempre atualizados
   - Falhar se `.stub.php` for modificado sem regenerar `.h`

### Referências

- [PHP Stub Documentation](https://php.github.io/php-src/miscellaneous/stubs.html)
- [gen_stub.php Source](https://github.com/php/php-src/blob/master/build/gen_stub.php)
- [ORT Extension PR](https://github.com/krakjoe/ort/pull/6)
- [ext/pdo Example](https://github.com/php/php-src/tree/master/ext/pdo)

### Status de Compilação

```
✅ zmatrix_arginfo.h: 233 linhas
✅ ztensor_arginfo.h: 190 linhas  
✅ Compilation: Clean (0 warnings/errors)
✅ Extension loaded: php -m | grep zmatrix
```

### Troubleshooting

**Q: "nikic/php-parser not found"**
```bash
composer install
```

**Q: "gen_stub.php not found"**
```bash
ls -la build/gen_stub.php  # Verificar se existe
# Se não existir, copiar do PHP core
```

**Q: Arginfo not regenerating**
```bash
rm -f zmatrix_arginfo.h ztensor_arginfo.h
make gen-stubs
```

**Q: Parser error no .stub.php**
- Remover `declare()` statements
- Remover `use` statements
- Simplificar tipos genéricos (`array<type>` → `array`)
- Manter sintaxe PHP simples e legível
