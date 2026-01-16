# Stub-Based Arginfo Generation for ZMatrix

## Overview

ZMatrix now adopts the modern PHP extension development approach using **stub-based arginfo generation**, aligning with PHP core's recommended practices (used in ext/dom, ext/pdo, ext/date, etc.).

This means:
- ✅ **Single source of truth** for API definitions (`.stub.php` files)
- ✅ **Automatic arginfo generation** from readable PHP syntax
- ✅ **IDE autocompletion** support for end users
- ✅ **Easier API maintenance** and documentation
- ✅ **Non-breaking integration** with existing C code

## Files Involved

| File | Purpose |
|------|---------|
| `zmatrix.stub.php` | PHP API definition for procedural functions |
| `ztensor.stub.php` | PHP API definition for ZTensor class |
| `build/gen_stub.php` | Generator script from PHP core (modified for Composer) |
| `zmatrix_arginfo.h` | **Generated** C header with arginfo (auto-generated) |
| `ztensor_arginfo.h` | **Generated** C header with arginfo (auto-generated) |
| `Makefile.fragments` | Make target for regenerating arginfo |
| `composer.json` | Added `nikic/php-parser` dev dependency |

## Workflow

### 1. **Define/Update API in Stub Files**

Edit `.stub.php` files with PHP syntax:

```php
// zmatrix.stub.php
<?php

/** @generate-function-entries */

/**
 * Multiplica matriz/tensor por escalar
 * @param array $a Input array/tensor
 * @param float $scalar Scalar value
 * @param bool $inplace Modify in place
 * @return array Result tensor
 */
function zmatrix_scalar_multiply(array $a, float $scalar, bool $inplace = false): array {}
```

### 2. **Regenerate Arginfo**

```bash
# Single command to regenerate all arginfo
make gen-stubs

# Or manually with PHP
php build/gen_stub.php zmatrix.stub.php ztensor.stub.php
```

This generates:
- `zmatrix_arginfo.h` - Complete function entries and class definitions
- `ztensor_arginfo.h` - Class and method arginfo

### 3. **Use in C Code**

Simply `#include "zmatrix_arginfo.h"` in C files:

```c
// src/zmatrix.c
#include "zmatrix_arginfo.h"

// Function entries are pre-generated:
// - zmatrix_functions[] (functions)
// - php_ztensor_methods[] (methods)
// - php_ztensor_ce (class entry)
```

## Benefits

### For Developers
- **Readable API**: See what you're building in familiar PHP syntax
- **Less boilerplate**: No more manual `ZEND_BEGIN_ARG_INFO_EX` macros
- **Consistency**: API definition is single source of truth
- **IDEs**: Type hints and autocompletion work everywhere

### For Users
- **Autocompletion**: IDEs can show function signatures
- **Type hints**: Static analysis tools understand parameter types
- **Documentation**: Stubs can be published as `zmatrix-stubs` package

### For Maintainers
- **Automation**: Regenerate in one command
- **Reduced errors**: Less manual C macro maintenance
- **Version safety**: Generated code includes PHP version compatibility
- **Build integration**: Regenerates only when stubs change

## Integration Steps

### Step 1: Install Dependencies
```bash
composer install
# Installs nikic/php-parser ^5.1 needed by gen_stub.php
```

### Step 2: Generate Arginfo
```bash
make gen-stubs
```

### Step 3: Verify Generation
```bash
ls -la zmatrix_arginfo.h ztensor_arginfo.h
# Both files should be generated successfully
```

### Step 4: Compile Normally
```bash
make clean
make
sudo make install
```

The generated headers are automatically included by the C code.

## Updating After Changes

After modifying `.stub.php` files:

```bash
# Option 1: Regenerate before building
make gen-stubs
make clean
make
sudo make install

# Option 2: One-line from root
make clean && make gen-stubs && make && sudo make install
```

## Advanced: Publishing Stubs for IDEs

To provide IDE support for ZMatrix users without the extension:

1. Create `zmatrix-stubs` package on Packagist
2. Copy `.stub.php` files (remove `@generate-*` annotations)
3. Users install: `composer require --dev vendor/zmatrix-stubs`
4. IDEs automatically recognize stubs and provide autocompletion

Example package structure:
```
zmatrix-stubs/
├── composer.json
├── zmatrix.stub.php
└── ztensor.stub.php
```

## Reference

- [PHP Stub Documentation](https://php.github.io/php-src/miscellaneous/stubs.html)
- [gen_stub.php in PHP Core](https://github.com/php/php-src/blob/master/build/gen_stub.php)
- [Example: ext/pdo stubs](https://github.com/php/php-src/tree/master/ext/pdo)
- [ORT Extension Reference](https://github.com/krakjoe/ort/pull/6)

## Troubleshooting

### Error: "nikic/php-parser not found"
```bash
composer install
# This installs the required PHP-Parser library
```

### Error: "gen_stub.php not found"
```bash
ls -la build/gen_stub.php
# If missing, ensure build/ directory exists and gen_stub.php was copied
```

### Arginfo not updating
```bash
# Force regeneration:
rm -f zmatrix_arginfo.h ztensor_arginfo.h
make gen-stubs
```

## Current Status

✅ **Implemented**:
- `zmatrix.stub.php` and `ztensor.stub.php` files created
- `build/gen_stub.php` script available
- `composer.json` updated with nikic/php-parser dependency
- `Makefile.fragments` updated with gen-stubs target

⏳ **Next Steps (Optional)**:
- Run `make gen-stubs` to generate arginfo
- Integrate generated headers into C code
- Test compilation and functionality
- Create `zmatrix-stubs` package for Packagist distribution
