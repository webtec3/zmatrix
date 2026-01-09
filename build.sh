#!/bin/bash

EXT_NAME="zmatrix"
PHP_VERSION="8.4"
PHP_CONFIG_PATH="/usr/bin/php-config${PHP_VERSION}"
PHPIZE_CMD="phpize${PHP_VERSION}"

# Cores
BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()    { echo -e "${BLUE}[BUILD]${NC} $1"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()  { echo -e "${RED}[ERROR]${NC} $1"; }
success(){ echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# Validar php-config e phpize
if [ ! -f "$PHP_CONFIG_PATH" ]; then
    error "php-config não encontrado: $PHP_CONFIG_PATH"
    exit 1
fi

if ! command -v "$PHPIZE_CMD" &>/dev/null; then
    if command -v phpize &>/dev/null; then
        warn "${PHPIZE_CMD} não encontrado. Usando phpize padrão."
        PHPIZE_CMD="phpize"
    else
        error "phpize não encontrado. Instale php${PHP_VERSION}-dev."
        exit 1
    fi
fi

# Forçar nvcc no PATH
export PATH="/usr/local/cuda/bin:$PATH"

log "🧹 Limpando build anterior..."
make distclean >/dev/null 2>&1 || warn "'make distclean' falhou ou não era necessário."

log "🔧 Executando ${PHPIZE_CMD}..."
${PHPIZE_CMD} --clean >/dev/null 2>&1 || true
${PHPIZE_CMD}

log "⚙️  Configurando extensão..."
./configure --with-php-config=${PHP_CONFIG_PATH} --enable-zmatrix | tee configure.log

if [ $? -ne 0 ]; then
    error "Falha na configuração com ./configure"
    exit 1
fi

echo ""
log "🔍 Verificando se CUDA foi ativado:"
if grep -q "HAVE_CUDA" configure.log; then
    success "CUDA está habilitado (HAVE_CUDA definido)"
else
    warn "CUDA NÃO FOI habilitado. Será usado fallback para CPU."
fi

log "🔨 Compilando extensão..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    error "Falha na compilação com make."
    exit 1
fi

log "📦 Instalando extensão..."
sudo make install

# Verificar onde foi instalada
INSTALL_DIR=$(grep "Installing shared extensions:" <<< "$(sudo make install 2>&1)" | awk '{print $NF}')
echo "ℹ️  Extensão instalada em: ${INSTALL_DIR}"
if [[ "$INSTALL_DIR" != *"/20240924"* ]]; then
    warn "A extensão pode ter sido instalada no diretório errado (${INSTALL_DIR})"
fi

# Ativar no php.ini
EXT_LINE="extension=${EXT_NAME}.so"
PHP_INI_CLI="/etc/php/${PHP_VERSION}/cli/php.ini"

if [ -f "$PHP_INI_CLI" ]; then
    if ! grep -qF -- "$EXT_LINE" "$PHP_INI_CLI"; then
        echo "$EXT_LINE" | sudo tee -a "$PHP_INI_CLI"
        success "Extensão adicionada em $PHP_INI_CLI"
    else
        log "Extensão já está no php.ini CLI"
    fi
else
    warn "php.ini CLI não encontrado. Tentando fallback..."
    PHP_INI_FALLBACK=$(php -i | grep "Loaded Configuration" | awk '{print $5}')
    if [ -f "$PHP_INI_FALLBACK" ]; then
        if ! grep -qF -- "$EXT_LINE" "$PHP_INI_FALLBACK"; then
            echo "$EXT_LINE" | sudo tee -a "$PHP_INI_FALLBACK"
            success "Extensão adicionada em $PHP_INI_FALLBACK"
        else
            log "Extensão já está no fallback ini"
        fi
    else
        error "Nenhum php.ini encontrado"
    fi
fi

log "🔁 Reiniciando PHP-FPM (se aplicável)..."
sudo systemctl restart php${PHP_VERSION}-fpm 2>/dev/null || warn "PHP-FPM para ${PHP_VERSION} não está ativo"

success "✅ Build completo com verificação de CUDA"
php -m | grep "$EXT_NAME"

log "🚀 Testando ZTensor::add()..."
php -r "ZMatrix\ZTensor::full([1000],1)->add(ZMatrix\ZTensor::full([1000],2));"

