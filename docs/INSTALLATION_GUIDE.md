# 📦 Guia Completo de Instalação do ZMatrix

## Sumário Executivo

**Instalação mínima (CPU):** 3 linhas
```bash
git clone https://github.com/omegaalfa/zmatrix.git && cd zmatrix
phpize && ./configure && make && sudo make install
echo "extension=zmatrix.so" | sudo tee -a /etc/php/8.1/cli/php.ini
```

**Com GPU (CUDA):** 4 linhas
```bash
git clone https://github.com/omegaalfa/zmatrix.git && cd zmatrix
phpize && ./configure --with-cuda-path=/usr/local/cuda && make && sudo make install
echo "extension=zmatrix.so" | sudo tee -a /etc/php/8.1/cli/php.ini
php -r "echo ZMatrix\ZTensor::arr([[1,2]])->toArray()[0][0];"  // Teste
```

---

## 📋 Dependências

### CPU-Only (Mínimas)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential php-dev autoconf pkg-config libblas-dev liblapack-dev
```

**CentOS/RHEL:**
```bash
sudo yum groupinstall "Development Tools" -y
sudo yum install -y php-devel autoconf pkg-config blas-devel lapack-devel
```

**macOS:**
```bash
brew install php autoconf pkg-config lapack
```

### Com GPU (CUDA)

Além das dependências de CPU:

```bash
# CUDA Toolkit 12.0
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.85.12_linux_x86_64.run
sudo sh cuda_12.0.0_525.85.12_linux_x86_64.run

# Drivers NVIDIA (se não tiver)
sudo apt-get install -y nvidia-driver-525
```

---

## 🚀 Passo a Passo

### 1. Verificar Dependências

```bash
# Verificar PHP
php -v

# Verificar compilador
gcc --version
g++ --version

# Verificar BLAS (CPU)
dpkg -l | grep blas

# Verificar CUDA (se quiser GPU)
nvcc --version
nvidia-smi
```

### 2. Clonar o Repositório

```bash
git clone https://github.com/omegaalfa/zmatrix.git
cd zmatrix
```

### 3. Compilar (CPU-Only)

```bash
phpize
./configure --enable-zmatrix
make -j$(nproc)
sudo make install
```

### 4. Compilar (Com GPU)

```bash
phpize
./configure --enable-zmatrix --with-cuda-path=/usr/local/cuda
make -j$(nproc)
sudo make install
```

### 5. Ativar a Extensão

```bash
# Encontrar arquivo php.ini
php -r 'echo php_ini_loaded_file();'

# Adicionar extensão
echo "extension=zmatrix.so" | sudo tee -a /etc/php/8.1/cli/php.ini

# Verificar
php -m | grep zmatrix
```

### 6. Testar a Instalação

```bash
# Teste básico
php -r "
use ZMatrix\ZTensor;
\$t = ZTensor::arr([[1, 2, 3], [4, 5, 6]]);
print_r(\$t->toArray());
echo 'ZMatrix instalado com sucesso! ✅\n';
"
```

### 7. Testar GPU (se instalado)

```bash
php -r "
use ZMatrix\ZTensor;
\$t = ZTensor::random([1000, 1000]);
try {
    \$t->toGpu();
    echo 'GPU disponível: ' . (\$t->isOnGpu() ? 'SIM ✅' : 'NÃO ❌') . '\n';
    \$t->toCpu();
} catch (Exception \$e) {
    echo 'GPU indisponível (normal): ' . \$e->getMessage() . '\n';
}
"
```

---

## 🛠️ Troubleshooting

### Problema: "phpize: command not found"

```bash
# Ubuntu/Debian
sudo apt-get install -y php-dev

# CentOS/RHEL
sudo yum install -y php-devel

# macOS
brew install php
```

### Problema: "Cannot find libblas.so"

```bash
# Instale BLAS
sudo apt-get install -y libblas-dev liblapack-dev libopenblas-dev

# Ou use OpenBLAS
sudo apt-get install -y libopenblas-dev
```

### Problema: "CUDA not found"

```bash
# Verifique instalação
nvcc --version
nvidia-smi

# Se não tiver, instale:
# https://developer.nvidia.com/cuda-downloads

# Se tiver, especifique o caminho
./configure --with-cuda-path=/usr/local/cuda-12.0
```

### Problema: "Extension not loaded"

```bash
# Verifique localização da extensão
find /usr -name "zmatrix.so" 2>/dev/null

# Verifique php.ini
php -i | grep "Loaded Configuration File"

# Adicione manualmente se necessário
sudo sh -c 'echo "extension=/caminho/para/zmatrix.so" >> /etc/php/8.1/cli/php.ini'
```

### Problema: "Permission denied" em make install

```bash
# Use sudo
sudo make install

# Ou configure para diretório do usuário
./configure --prefix=$HOME/.php
make && make install
# Então adicione ao php.ini com caminho completo
echo "extension=$HOME/.php/lib/php/extensions/*/zmatrix.so" >> php.ini
```

---

## ✅ Matriz de Compatibilidade

| Sistema | CPU | GPU | Teste |
|---------|-----|-----|-------|
| Ubuntu 20.04 | ✅ | ✅ | `php test_gpu_vs_cpu.php` |
| Ubuntu 22.04 | ✅ | ✅ | `php test_gpu_vs_cpu.php` |
| Debian 11 | ✅ | ✅ | ✓ |
| CentOS 7 | ✅ | ✅ | ✓ |
| CentOS 8 | ✅ | ✅ | ✓ |
| macOS (Intel) | ✅ | ❌ | ✓ |
| macOS (Apple Silicon) | ✅ | ❌ | ⚠️ |
| WSL2 (Ubuntu) | ✅ | ✅ | ✓ |
| Docker | ✅ | ⚠️ | ✓ |

---

## 🎯 Recomendações por Caso de Uso

### Desenvolvimento Local (rápido)

```bash
./configure --enable-zmatrix
make && sudo make install
```

### Produção com CPU

```bash
./configure --enable-zmatrix --enable-shared
make -j$(nproc) && sudo make install
```

### Produção com GPU

```bash
./configure --enable-zmatrix --with-cuda-path=/usr/local/cuda --enable-shared
make -j$(nproc) && sudo make install
```

### Container/Docker (sem GPU)

```dockerfile
FROM php:8.1-cli
RUN apt-get update && apt-get install -y build-essential php-dev autoconf pkg-config libblas-dev liblapack-dev
WORKDIR /tmp
RUN git clone https://github.com/omegaalfa/zmatrix.git && cd zmatrix && phpize && ./configure && make -j && make install
RUN echo "extension=zmatrix.so" >> /usr/local/etc/php/conf.d/docker-php-ext-zmatrix.ini
```

---

## 📞 Suporte

- 📖 Documentação: [README.md](README.md)
- 🐛 Issues: [GitHub Issues](https://github.com/omegaalfa/zmatrix/issues)
- 📚 Exemplos: Veja diretórios `examples/` e `tests/`
- 🚀 GPU Guide: [QUICK_GPU_GUIDE.md](QUICK_GPU_GUIDE.md)
