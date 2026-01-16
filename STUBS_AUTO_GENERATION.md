# Como os Stubs São Gerados - Fluxo Automático

## 📋 Resposta Rápida

**SIM, os stubs serão gerados automaticamente!**

### Comando automático:
```bash
# 1. Você executa:
./gen_arginfo.sh

# 2. Script automaticamente:
#    - Verifica se .stub.php foi modificado
#    - Se sim, executa: php build/gen_stub.php zmatrix.stub.php ztensor.stub.php
#    - Gera: zmatrix_arginfo.h e ztensor_arginfo.h
```

## 🔄 Fluxo de Desenvolvimento

### Opção 1: Manual (Simples)
```bash
# 1. Editar stub (ex: adicionar novo método)
vim ztensor.stub.php

# 2. Regenerar arginfo
./gen_arginfo.sh

# 3. Compilar normalmente
make clean && make && sudo make install
```

### Opção 2: Automático (Dev Makefile)
```bash
# Tudo em um comando!
make -f Makefile.dev install

# Internamente executa:
#   1. ./gen_arginfo.sh    (gera arginfo se stubs mudaram)
#   2. ./configure         (configura build)
#   3. make                (compila)
#   4. sudo make install   (instala)
```

### Opção 3: Durante Configure (Futuro)
Adicionado ao `config.m4` para rodar automaticamente durante `./configure`

## 📜 O Script: gen_arginfo.sh

```bash
#!/bin/bash
# Verifica se stubs foram modificados
if [ "zmatrix.stub.php" -nt "zmatrix_arginfo.h" ]; then
    # Regenerar arginfo
    php build/gen_stub.php zmatrix.stub.php ztensor.stub.php
fi
```

**Comportamento:**
- ✅ Se `.stub.php` foi modificado → Regenera arginfo
- ✅ Se arginfo não existe → Gera automaticamente
- ✅ Se arginfo está atualizado → Pula regeneração (rápido!)

## 🔄 Sequência Completa

```
1. Você edita ztensor.stub.php
   └─ Adiciona novo método
   
2. Você executa: ./gen_arginfo.sh
   └─ Detecta mudança em .stub.php
   └─ Executa: php build/gen_stub.php zmatrix.stub.php ztensor.stub.php
   └─ Gera: ztensor_arginfo.h (atualizado)
   
3. Você executa: make clean && make
   └─ Compila C++ code
   └─ Linka ztensor_arginfo.h (arginfo atualizado)
   └─ Gera: zmatrix.so (extension)
   
4. Você executa: sudo make install
   └─ Instala: /usr/lib/php/20240924/zmatrix.so
   
5. Pronto!
   └─ Nova função/método disponível em PHP
```

## 📊 Comparação: Antes vs Depois

| Etapa | Antes | Depois |
|-------|-------|--------|
| Editar API | `zmatrix.c` (C macros) | `ztensor.stub.php` (PHP syntax) |
| Gerar arginfo | Manual (ZEND_ARG_INFO) | Automático (`gen_stub.php`) |
| Compilar | `make clean && make` | `./gen_arginfo.sh && make` |
| Fluxo total | 5 min | 2 min |

## 🚀 Recomendação: Use o Makefile.dev

```bash
# Desenvolvimento rápido - tudo automático!
make -f Makefile.dev install
```

Isto executa:
1. `./gen_arginfo.sh` → Regenera se necessário
2. `./configure` → Configura build
3. `make` → Compila
4. `sudo make install` → Instala

## 📝 Setup Inicial

Uma única vez após clonar:
```bash
composer install                          # Instala nikic/php-parser
chmod +x gen_arginfo.sh                   # Torna script executável
chmod +x Makefile.dev                     # (Opcional)
```

Pronto! Agora sempre que você editar `.stub.php` e rodar `./gen_arginfo.sh`, os arginfo serão regenerados automaticamente!

## ⚙️ Como Verificar

```bash
# Ver timestamp dos arquivos
ls -lh zmatrix.stub.php ztensor.stub.php zmatrix_arginfo.h ztensor_arginfo.h

# zmatrix_arginfo.h deve ser mais recente que zmatrix.stub.php se estão sincronizados
```

## 🔧 Troubleshooting

**Problema: "zmatrix_arginfo.h não foi gerado"**
```bash
# Verificar se gen_stub.php existe
ls -la build/gen_stub.php

# Executar manualmente com debug
php build/gen_stub.php zmatrix.stub.php ztensor.stub.php -v
```

**Problema: "Script não tem permissão"**
```bash
chmod +x gen_arginfo.sh
./gen_arginfo.sh
```

**Problema: "Arginfo antigo sendo usado"**
```bash
# Força regeneração
rm -f zmatrix_arginfo.h ztensor_arginfo.h
./gen_arginfo.sh
make clean && make && sudo make install
```
