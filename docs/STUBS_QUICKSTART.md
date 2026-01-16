# Quick Start: Stub-Based Arginfo

## O que mudou?

O ZMatrix agora usa **stub-based arginfo generation** - a abordagem moderna do PHP core para definir APIs de extensões.

### Antes ❌
```c
// Manual C macros (error-prone)
ZEND_BEGIN_ARG_INFO_EX(php_ztensor_construct_arginfo, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, dataOrShape, IS_ARRAY, 1)
ZEND_END_ARG_INFO()
```

### Depois ✅
```php
// Legível, automático
public function __construct(ZTensor|array|null $dataOrShape = null) {}
```

## Quick Start

### 1️⃣ Setup (Uma vez)
```bash
cd ~/php-projetos/php-extension/zmatrix
composer install
```

### 2️⃣ Desenvolvimento Normal
```bash
# Editar .stub.php com sintaxe PHP
vim ztensor.stub.php

# Regenerar arginfo automaticamente
make gen-stubs

# Compilar normalmente
make clean && make && sudo make install
```

### 3️⃣ Verificar
```bash
php -m | grep zmatrix
# Output: zmatrix ✅
```

## Arquivos Importantes

| Arquivo | Descrição | Editar? |
|---------|-----------|---------|
| `zmatrix.stub.php` | API procedural | ✅ SIM |
| `ztensor.stub.php` | API classe ZTensor | ✅ SIM |
| `zmatrix_arginfo.h` | **Gerado** | ❌ NÃO |
| `ztensor_arginfo.h` | **Gerado** | ❌ NÃO |

## Problemas Comuns

**❌ "nikic/php-parser not found"**
```bash
composer install
```

**❌ "make gen-stubs: command not found"**
```bash
# Verifique Makefile.fragments
grep "gen-stubs" Makefile.fragments
```

**❌ Parser error ao regenerar**
- Remova `declare(strict_types=1);`
- Remova `use` statements
- Use apenas tipos PHP simples (sem `<int>`)

## Exemplos

### Adicionar nova função
```php
// ztensor.stub.php
public function myNewMethod(int $param, string $name = "default"): array {}
```

Depois:
```bash
make gen-stubs && make && sudo make install
```

### Modificar assinatura
```php
// Antes
public function foo(float $x): ZTensor {}

// Depois
public function foo(float $x, bool $inplace = false): ZTensor {}
```

Depois:
```bash
make gen-stubs && make && sudo make install
```

## Status Atual

✅ Implementado e compilado com sucesso  
✅ `zmatrix_arginfo.h`: 233 linhas geradas  
✅ `ztensor_arginfo.h`: 190 linhas geradas  
✅ Extensão carrega normalmente em PHP  

## Documentação Completa

- [STUB_BASED_ARGINFO.md](STUB_BASED_ARGINFO.md) - Guia detalhado
- [STUB_IMPLEMENTATION_REPORT.md](STUB_IMPLEMENTATION_REPORT.md) - Relatório técnico
- [PHP Stubs Docs](https://php.github.io/php-src/miscellaneous/stubs.html) - Referência oficial

## Suporte

Para ajuda com tipos PHP em stubs:
```php
// ✅ Tipos válidos
array, string, int, float, bool
ZTensor, ZTensor|array, ?int (nullable)
int|float (union types)
```

```php
// ❌ Tipos inválidos em stubs
array<int>, array<int|float>, array<string, mixed>
```
