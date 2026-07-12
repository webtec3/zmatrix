# 📊 ZMatrix - High-Performance Matrix and Tensor Operations for PHP

ZMatrix é uma extensão nativa para PHP escrita em C++ de altíssima performance voltada para álgebra linear, computação científica e Machine Learning. Ela implementa a classe `ZMatrix\ZTensor`, encapsulando tensores n-dimensionais com paralelismo automático via **OpenMP** e aceleração de hardware via **BLAS (cblas_sgemm)** e **CUDA**.

Com o `ZMatrix`, o ecossistema PHP ganha um motor matemático robusto capaz de rodar Redes Neurais complexas e algoritmos baseados em árvores (`Random Forest`, `XGBoost`) localmente e nativamente, sem a necessidade de acoplar stacks externas em Python.

---

## 🚀 Instalação e Compilação

Para compilar a extensão e ativá-la no seu ambiente local:

```bash
phpize
./configure --enable-zmatrix
make -j$(nproc)
sudo make install

```

### Ativação com suporte a GPU (CUDA)

```bash
phpize
./configure --with-cuda-path=/usr/local/cuda
make clean
make -j$(nproc)
sudo make install

```

Adicione a linha abaixo ao seu arquivo `php.ini`:

```ini
extension=zmatrix.so

```

---

## 📋 Índice da API do ZTensor

1. [Construtores e Métodos de Criação](#1-construtores-e-métodos-de-criação)
2. [Propriedades do Tensor](#2-propriedades-do-tensor)
3. [Operações Aritméticas e Álgebra Linear](#3-operações-aritméticas-e-álgebra-linear)
4. [Operações Matemáticas Elemento a Elemento](#4-operações-matemáticas-elemento-a-elemento)
5. [Funções de Ativação e Derivadas](#5-funções-de-ativação-e-derivadas)
6. [Estatística, Redução e Comparação](#6-estatística-redução-e-comparação)
7. [Transformação, Fatiamento e Formato](#7-transformação-fatiamento-e-formato)
8. [Gerenciamento de Dispositivos (CPU / GPU CUDA)](#8-gerenciamento-de-dispositivos-cpu--gpu-cuda)
9. [Grafo de Computação e Autograd](#9-grafo-de-computação-e-autograd)

---

### Uso Básico

```php
use ZMatrix\ZTensor;

```

---

## 1. Construtores e Métodos de Criação

### `__construct`

Instancia um novo objeto `ZTensor`. Pode receber um formato (`shape`), um array multidimensional PHP nativo ou nenhum argumento (criando um tensor vazio).

```php
// Criando por formato (inicializado com zeros)
$t1 = new ZTensor([2, 3]);

// Criando direto por dados
$t2 = new ZTensor([[1, 2], [3, 4]]);

```

### `arr`

Método estático de fábrica (*factory*) para converter um array PHP aninhado ou clonar outro `ZTensor`.

```php
$t = ZTensor::arr([[1.5, 2.3], [4.0, 5.1]]);

```

### `safe`

Equivalente estático seguro ao método `arr()`. Garante a higienização dos tipos numéricos do PHP para o tipo `float` interno da engine C++.

```php
$t = ZTensor::safe([1, 2, 3, 4]);

```

### `zeros`

Cria um tensor preenchido inteiramente com o valor padrão `0.0f` no formato especificado.

```php
$t = ZTensor::zeros([3, 3]);
// [[0,0,0], [0,0,0], [0,0,0]]

```

### `ones`

Cria um tensor preenchido inteiramente com o valor padrão `1.0f` no formato especificado.

```php
$t = ZTensor::ones([2, 5]);

```

### `full`

Cria um tensor inicializado com um valor escalar constante customizado.

```php
$t = ZTensor::full([2, 2], 7.25);
// [[7.25, 7.25], [7.25, 7.25]]

```

### `fill`

Modifica e preenche o tensor atual com um valor escalar constante específico.

```php
$t = ZTensor::zeros([2, 2]);$t->fill(3.14);

```

### `identity`

Gera uma matriz identidade quadrada de dimensão $N \times N$.

```php
$t = ZTensor::identity(3);
// [[1,0,0], [0,1,0], [0,0,1]]

```

### `eye`

Gera uma matriz diagonal com suporte a linhas ($N$), colunas ($M$) e deslocamento opcional de diagonal ($k$).

```php
$t = ZTensor::eye(3, 4, 1); // 3 linhas, 4 colunas, diagonal superior (+1)

```

### `arange`

Gera uma sequência linear de dados no formato de vetor 1D com limites de início (*start*), fim (*stop*) e passo (*step*).

```php
$t = ZTensor::arange(0, 10, 2.5);
// [0.0, 2.5, 5.0, 7.5]

```

### `linspace`

Gera um número específico de valores uniformemente espaçados dentro de um intervalo definido.

```php
$t = ZTensor::linspace(0, 1, 5);
// [0.0, 0.25, 0.5, 0.75, 1.0]

```

### `logspace`

Gera uma sequência espaçada logaritmicamente entre duas potências com uma base definida (padrão é 10).

```php
$t = ZTensor::logspace(1, 3, 3); // 10^1 a 10^3 com 3 amostras
// [10.0, 100.0, 1000.0]

```

### `random`

Gera uma matriz com distribuição aleatória uniforme contida entre os intervalos `min` e `max`.

```php
$t = ZTensor::random([2, 3], -1.0, 1.0);

```

### `randn`

Gera uma matriz baseada em uma distribuição normal estável (Gaussiana) parametrizada por média (`mean`) e desvio padrão (`std_dev`).

```php
$t = ZTensor::randn([3, 3], 0.0, 1.0); // Distribuição normal padrão

```

---

## 2. Propriedades do Tensor

### `shape`

Retorna um array PHP indexado contendo as dimensões atuais da estrutura do tensor.

```php
$t = ZTensor::zeros([3, 5]);
print_r($t->shape()); // Out: [3, 5]

```

### `size`

Retorna a contagem linear acumulada contendo o número total de elementos alocados na memória do tensor.

```php
$t = ZTensor::zeros([2, 3, 4]);
echo $t->size(); // Out: 24

```

### `ndim`

Retorna a dimensionalidade (número de eixos coordenados) do tensor.

```php
$t = ZTensor::arr([1, 2, 3]);
echo $t->ndim(); // Out: 1

```

### `isEmpty`

Retorna um valor booleano indicando se o tensor está vazio (sem eixos ou tamanho zero).

```php
$t = new ZTensor();
var_dump($t->isEmpty()); // Out: true

```

### `toArray`

Exporta a estrutura binária contígua do tensor C++ de volta para uma estrutura nativa de arrays aninhados do PHP.

```php
$t = ZTensor::ones([2, 2]);
$arr =$t->toArray();

```

---

## 3. Operações Aritméticas e Álgebra Linear

*Nota: As operações aritméticas básicas oferecem suporte nativo a **Broadcasting** automático entre tensores 2D e 1D.*

### `add`

Soma elemento a elemento outro tensor, array PHP ou adiciona um valor escalar de forma in-place.

```php
$a = ZTensor::arr([1, 2, 3]);
$a->add([1, 1, 1]); //$a vira [2, 3, 4]

```

### `sub`

Subtrai elemento a elemento outro tensor, array PHP ou valor escalar de forma in-place.

```php
$a = ZTensor::arr([5, 5, 5]);
$a->sub(2); //$a vira [3, 3, 3]

```

### `mul`

Aplica a multiplicação elemento a elemento de Hadamard (não confundir com multiplicação matricial). Suporta escalares.

```php
$a = ZTensor::arr([2, 3]);
$a->mul([2, 4]); //$a vira [4, 12]

```

### `divide`

Aplica a divisão elemento a elemento de forma in-place. Dispara uma exceção caso detecte divisão por zero.

```php
$a = ZTensor::arr([10, 20]);
$a->divide([2, 4]); //$a vira [5, 5]

```

### `scalarMultiply`

Multiplica de forma otimizada todos os índices do tensor por um valor escalar fracionário.

```php
$t = ZTensor::ones([2, 2]);$t->scalarMultiply(5.5);

```

### `scalarDivide`

Divide de forma otimizada todos os índices do tensor por um valor escalar estável.

```php
$t = ZTensor::full([2, 2], 10.0);$t->scalarDivide(2.0);

```

### `matmul`

Multiplicação de Matrizes Pura (Aplica o algoritmo otimizado `cblas_sgemm` do BLAS). Requer tensores bidimensionais compatíveis com a regra $(M \times K) \cdot (K \times N)$.

```php
$a = ZTensor::arr([[1, 2], [3, 4]]); // [2x2]
$b = ZTensor::arr([[5], [6]]);       // [2x1]$c = $a->matmul($b);                 // Resultado [2x1]

```

### `dot`

Produto escalar genérico. Se os inputs forem vetores 1D, calcula o produto interno clássico. Se forem matrizes 2D/1D, aplica a multiplicação correspondente.

```php
$v1 = ZTensor::arr([1, 2]);$v2 = ZTensor::arr([3, 4]);
echo $v1->dot($v2); // Out: 11.0 (1*3 + 2*4)

```

---

## 4. Operações Matemáticas Elemento a Elemento

### `abs`

Aplica o cálculo do valor absoluto (módulo) in-place para remover sinais negativos de todos os coeficientes.

```php
$t = ZMatrix\ZTensor::arr([-5, 2, -10]);$t->abs(); // [5, 2, 10]

```

### `exp`

Aplica a função exponencial de base natural $e^x$ in-place em todos os componentes.

```php
$t = ZTensor::arr([0, 1]);$t->exp(); // [1.0, 2.71828...]

```

### `log`

Aplica o logaritmo natural ($\ln x$) in-place. Lança exceções caso encontre valores menores ou iguais a zero.

```php
$t = ZTensor::arr([1, 2.718282]);$t->log(); // [0.0, 1.0]

```

### `sqrt`

Calcula e aplica a raiz quadrada ($\sqrt{x}$) de forma in-place. Lança exceções caso existam números negativos.

```php
$t = ZTensor::arr([4, 16]);$t->sqrt(); // [2.0, 4.0]

```

### `pow`

Eleva cada elemento contido na matriz à potência exponencial do argumento escalar passado.

```php
$t = ZTensor::arr([2, 3]);$t->pow(3); // [8.0, 27.0]

```

---

## 5. Funções de Ativação e Derivadas

### `sigmoid`

Aplica a curva logística de ativação Sigmoid in-place: $f(x) = \frac{1}{1 + e^{-x}}$.

```php
$t = ZTensor::arr([0.0]);$t->sigmoid(); // [0.5]

```

### `sigmoidDerivative`

Calcula a derivada da Sigmoid sobre os valores locais estabilizados do tensor: $f'(x) = f(x) \cdot (1 - f(x))$.

```php
$t->sigmoidDerivative();

```

### `relu`

Aplica a retificação linear clássica (ReLU) in-place: $f(x) = \max(0, x)$.

```php
$t = ZTensor::arr([-5, 2]);$t->relu(); // [0.0, 2.0]

```

### `reluDerivative`

Calcula o gradiente descendente da ReLU substituindo os eixos por valores booleanos normalizados ($1.0$ se $x > 0$, caso contrário $0.0$).

```php
$t->reluDerivative();

```

### `leakyRelu`

Aplica a retificação linear com vazamento controlado por um coeficiente constante multiplicador (`alpha`).

```php
$t = ZTensor::arr([-2, 4]);$t->leakyRelu(0.01); // [-0.02, 4.0]

```

### `leakyReluDerivative`

Gera a derivada correspondente da função Leaky ReLU ($1.0$ se $x > 0$, caso contrário `alpha`).

```php
$t->leakyReluDerivative(0.01);

```

### `tanh`

Aplica a tangente hiperbólica in-place limitando as coordenadas lógicas entre os intervalos de $[-1.0, 1.0]$.

```php
$t = ZTensor::arr([0.0]);$t->tanh(); // [0.0]

```

### `tanhDerivative`

Mapeia o cálculo derivativo da tangente hiperbólica: $f'(x) = 1 - \tanh^2(x)$.

```php
$t->tanhDerivative();

```

### `softmax`

Calcula e distribui a exponencial probabilística Softmax ao longo das linhas do tensor (usado para classificação multiclasse).

```php
$t = ZTensor::arr([[1.0, 2.0, 3.0]]);$t->softmax(); // Converte em probabilidades que somam 1.0

```

### `softmaxDerivative`

Calcula e extrai os componentes da matriz Jacobiana correspondentes ao gradiente da ativação Softmax.

```php
$t->softmaxDerivative();

```

---

## 6. Estatística, Redução e Comparação

### `sumtotal`

Soma de forma global e irrestrita todos os componentes contidos na memória do tensor, retornando um valor `double`.

```php
$t = ZTensor::arr([[1, 2], [3, 4]]);
echo $t->sumtotal(); // Out: 10.0

```

### `mean`

Calcula a média aritmética simples de todos os elementos contidos no objeto.

```php
$t = ZTensor::arr([1, 2, 3, 4]);
echo $t->mean(); // Out: 2.5

```

### `min`

Retorna o menor coeficiente escalar mapeado dentro da estrutura do tensor.

```php
$t = ZTensor::arr([4, -1, 3]);
echo $t->min(); // Out: -1.0

```

### `max`

Retorna o maior coeficiente escalar mapeado dentro da estrutura do tensor.

```php
$t = ZTensor::arr([4, -1, 3]);
echo $t->max(); // Out: 4.0

```

### `std`

Retorna o desvio padrão amostral corrigido de todos os elementos alocados no tensor.

```php
$t = ZTensor::arr([1, 2, 3, 4]);
echo $t->std();

```

### `sum`

Redução focada por eixos lógicos (`axis`). O resultado acumulado é injetado diretamente em um tensor de saída previamente alocado com a dimensionalidade reduzida correspondente.

```php
$t = ZTensor::arr([[1, 2], [3, 4]]);
$out = ZTensor::zeros([2]);$t->sum($out, 1); // Soma ao longo do eixo das colunas (eixo 1) //$out vira [3, 7]

```

### `greater`

Compara elemento a elemento se o tensor atual é maior que outra matriz ou array, retornando um tensor binário ($1.0$ onde for verdadeiro, $0.0$ onde for falso).

```php
$a = ZTensor::arr([1, 5]);
$b =$a->greater([2, 2]); // [0.0, 1.0]

```

### `clip`

Método estático utilitário que restringe e intercepta todos os coeficientes do tensor de entrada dentro das fronteiras lineares especificadas por `min` e `max`.

```php
$t = ZTensor::arr([-5, 5, 20]);
$clipped = ZTensor::clip($t, 0.0, 10.0); // [0.0, 5.0, 10.0]

```

### `minimum`

Método estático utilitário que gera um novo tensor contendo a comparação mínima elemento a elemento entre uma matriz e um valor escalar constante.

```php
$t = ZTensor::arr([1, 5, 2]);
$res = ZTensor::minimum($t, 3.0); // [1.0, 3.0, 2.0]

```

### `maximum`

Método estático utilitário que gera um novo tensor contendo a comparação máxima elemento a elemento entre uma matriz e um valor escalar constante.

```php
$t = ZTensor::arr([1, 5, 2]);
$res = ZTensor::maximum($t, 3.0); // [3.0, 5.0, 3.0]

```

---

## 7. Transformação, Fatiamento e Formato

### `reshape`

Modifica os eixos lógicos do tensor sem alterar a ordem física dos dados alocados na memória contígua.

```php
$t = ZTensor::arr([1, 2, 3, 4]); // 1D
$matrix =$t->reshape([2, 2]);   // Transforma em 2D de formato [2x2]

```

### `transpose`

Inverte os eixos de uma matriz bidimensional trocando o posicionamento original de linhas por colunas.

```php
$t = ZTensor::arr([[1, 2, 3], [4, 5, 6]]);
$transposta =$t->transpose(); // Transforma o formato de [2x3] para [3x2]

```

### `slice`

Recorta uma janela ou bloco bidimensional contíguo especificando os eixos coordenados de início e a janela de deslocamento final.

```php
$t = ZTensor::arr([[1,2,3], [4,5,6], [7,8,9]]);
$subMatriz =$t->slice(0, 0, 2); // Fatia sub-regiões estruturadas

```

### `broadcast`

Expande e propaga de forma matemática um viés ou vetor unidimensional (`bias`) através das linhas correspondentes de uma matriz bidimensional dominante.

```php
$matrix = ZTensor::zeros([3, 2]);
$bias = ZTensor::arr([10, 20]);$res = $matrix->broadcast($bias); // Adiciona [10, 20] a todas as 3 linhas

```

### `copy`

Realiza uma cópia profunda (*deep copy*) realocando novos ponteiros e isolando completamente a memória do novo tensor em relação ao original.

```php
$t = ZTensor::arr([1, 2]);
$dublo =$t->copy();

```

### `key`

Acessa de forma pontual o valor real de um escalar diretamente através das suas coordenadas multidimensionais indexadas.

```php
$t = ZTensor::arr([[1, 2], [3, 4]]);
echo $t->key([1, 0]); // Acessa Linha 1, Coluna 0. Out: 3.0

```

### `tile`

Método estático para repetir e empilhar verticalmente uma matriz um número determinado de vezes.

```php
$t = ZTensor::arr([[1, 2]]);
$tiled = ZTensor::tile($t, 3); // Empilha a linha 3 vezes. Formato vira [3x2]

```

### `column`

Extrai de forma isolada uma coluna específica de uma matriz bidimensional, retornando um novo tensor coluna.

```php
$t = ZTensor::arr([[1, 2], [3, 4]]);
$col =$t->column(0); // Extrai a primeira coluna: [1, 3]

```

### `argsort`

Retorna os índices que ordenariam o tensor ao longo do eixo selecionado. Essencial para algoritmos de busca e árvores.

```php
$t = ZTensor::arr([30, 10, 20]);
$idx =$t->argsort(0); // [1, 2, 0] (posições dos valores ordenados: 10, 20, 30)

```

### `where`

Varre uma coluna específica avaliando se os valores atendem a um limiar numérico informado, retornando uma máscara de índices compatíveis.

```php
$t = ZTensor::arr([[1, 10], [4, 20]]);
$mask =$t->where(0, 2.0); // Varre a coluna 0 buscando valores maiores que 2.0

```

---

## 8. Gerenciamento de Dispositivos (CPU / GPU CUDA)

### `toGpu`

Transfere de forma síncrona o mapeamento de memória do array contíguo da memória RAM da CPU direto para os núcleos de processamento paralelo da VRAM da GPU (via **CUDA**).

```php
$t = ZTensor::random([1000, 1000]);
$gpuTensor =$t->toGpu(); // Pronto para aceleração por hardware massiva

```

### `toCpu`

Realiza a operação inversa, trazendo os dados processados e armazenados na VRAM da placa de vídeo de volta para a memória RAM controlada pela CPU.

```php
$cpuTensor =$gpuTensor->toCpu();

```

### `isOnGpu`

Retorna um sinalizador booleano indicando se a instância do tensor reside atualmente na memória da GPU.

```php
if ($t->isOnGpu()) {
    // Código otimizado para GPU
}

```

### `freeDevice`

Libera explicitamente os blocos de memória e os ponteiros alocados na GPU associados ao tensor, evitando vazamentos de memória na VRAM (*VRAM Memory Leaks*).

```php
$gpuTensor->freeDevice();

```

---

## 9. Grafo de Computação e Autograd

O mecanismo de **Autograd** monitora e constrói de forma implícita o grafo de computação dinâmica de operações matemáticas, permitindo a execução automática da retropropagação (*backpropagation*) de gradientes.

### `requiresGrad`

Ativa ou desativa o monitoramento de rastreamento do grafo de computação e acumulação de gradientes para o tensor selecionado.

```php
$x = ZTensor::arr([2.0])->requiresGrad(true);

```

### `isRequiresGrad`

Retorna o estado booleano atual indicando se o tensor está configurado para registrar informações de gradiente.

```php
echo $x->isRequiresGrad() ? 'Sim' : 'Não';

```

### `ensureGrad`

Garante que a alocação de memória reservada para os gradientes internos do tensor esteja instanciada e pronta para receber os passos do otimizador.

```php
$x->ensureGrad();

```

### `getGrad`

Retorna um novo objeto `ZTensor` contendo os gradientes acumulados gerados após a execução de um passo de retropropagação.

```php
$gradientes =$x->getGrad();

```

### `zeroGrad`

Zera os gradientes acumulados na instância do tensor. Crucial para limpar o histórico de gradientes a cada nova época de treinamento.

```php
$weights->zeroGrad();

```

### `backward`

Dispara de forma recursiva a retropropagação da regra da cadeia a partir do nó de saída atual, calculando e distribuindo os gradientes de todas as variáveis dependentes no grafo.

```php
$loss->backward(); // Calcula dW e dB para a rede automaticamente

```

### `addAutograd`

Método estático que realiza a operação de soma registrando o nó de computação correspondente no grafo para o Autograd.

```php
$z = ZTensor::addAutograd($x,$y);

```

### `subAutograd`

Método estático que realiza a operação de subtração registrando o nó de computação correspondente no grafo para o Autograd.

```php
$z = ZTensor::subAutograd($x,$y);

```

### `mulAutograd`

Método estático que realiza a operação de multiplicação de Hadamard registrando o nó de computação correspondente no grafo para o Autograd.

```php
$z = ZTensor::mulAutograd($x,$y);

```

### `sumAutograd`

Método estático que realiza a redução de soma registrando o nó de computação correspondente no grafo para o Autograd.

```php
$z = ZTensor::sumAutograd($x);

```
## 10. Operações de Manipulação Estrutural e Contagem (Otimizadas para Árvores de Decisão)

Esta seção documenta as primitivas de vetorização lógica essenciais para o desenvolvimento de ecossistemas ensemble (como `Random Forest`, `XGBoost`, `LightGBM`), focadas em ordenação rápida, histogramas de frequência e fusão densa de memória contígua.

### `unique`
Extrai e isola todos os elementos únicos contidos no tensor, aplicando ordenação ascendente automática. É executado inteiramente na camada nativa C++ através de algoritmos STL altamente otimizados antes de mover os ponteiros para um novo vetor 1D independente.
* **Retorno:** `ZTensor` — Um novo tensor unidimensional contendo apenas os valores exclusivos ordenados.
* **Exceções:** Lança `\Exception` caso a instância do tensor não tenha sido devidamente inicializada.

```php
use ZMatrix\ZTensor;

$t = ZTensor::arr([4.0, 1.0, 1.0, 3.0, 4.0, 2.0]);
$res =$t->unique();

print_r($res->toArray());
// Output: [1.0, 2.0, 3.0, 4.0]

// Suporte nativo a tensores vazios
$vazio = ZTensor::zeros([0]);
echo $vazio->unique()->size(); // Output: 0
```

### bincount
Calcula o histograma de frequências contando o número de ocorrências de cada valor inteiro não-negativo mapeado no tensor.

Performance: Utiliza paralelismo dinâmico via OpenMP com cláusulas de sincronização atômica (#pragma omp atomic) para evitar concorrência desordenada de memória (race conditions) durante incrementos simultâneos efetuados por múltiplas threads da CPU.

Parâmetros: * int $minlength (Opcional): Força um tamanho mínimo preenchido por zeros para o tensor de saída (útil para mapear classes ausentes em subconjuntos de árvores). O padrão é 0.

Retorno: ZTensor — Um tensor unidimensional de contagem onde o índice representa o valor inteiro encontrado e o elemento contido representa a sua frequência acumulada.

Exceções: Lança \Exception caso o tensor contenha números negativos ou não esteja inicializado.
```php
use ZMatrix\ZTensor;

// Contagem clássica de frequência de classes (ex: labels de treino)
$labels = ZTensor::arr([1, 2, 1, 1, 3, 2, 5]);
$histograma =$labels->bincount();

print_r($histograma->toArray());
// O índice mapeia os valores de 0 a 5. Valores ausentes (como o 0 e o 4) recebem 0.0
// Output: [0.0, 3.0, 2.0, 1.0, 0.0, 1.0]

// Uso do minlength para forçar tamanho fixo de buckets
$res_min = ZTensor::arr([1, 2])->bincount(10);
echo $res_min->size(); // Output: 10
```

### argmax
Varre linearmente toda a estrutura de dados do tensor e localiza a posição física contígua (flat index) do maior coeficiente escalar encontrado.

Performance: Implementa um algoritmo híbrido de redução paralela. Cada thread do OpenMP calcula o máximo local de forma isolada (nowait) em registradores da CPU, fundindo as respostas em uma variável global protegida por região crítica (#pragma omp critical) apenas no final. Garante estabilidade matemática retornando estritamente a primeira ocorrência em caso de múltiplos máximos idênticos.

Retorno: int — O índice achatado linear correspondente ao valor máximo absoluto.

Exceções: Lança \Exception se o tensor estiver vazio ou não inicializado.

```php
use ZMatrix\ZTensor;

// Localização em matriz bidimensional
$matriz = ZTensor::arr([
    [1.0, 2.0,  3.0], 
    [4.0, 10.5, 2.0]
]);

$flat_index =$matriz->argmax();
echo $flat_index; // Output: 4 (Referente ao valor 10.5 na posição linear [1, 1])

// Estabilidade garantida (retorna o primeiro índice do empate)
$empate = ZTensor::arr([1.0, 5.0, 3.0, 5.0]);
echo $empate->argmax(); // Output: 1 (ignora a posição 3)

```

### concat
Método estático que realiza a fusão estrutural de uma coleção de tensores ao longo de um eixo de coordenadas (axis) pré-determinado.

Performance: Aplica a otimização Fast-Path por blocos contíguos. Se o eixo de concatenação for a dimensão externa dominante (axis = 0), a extensão ignora loops aninhados e transfere blocos inteiros de memória física diretamente via std::memcpy, atingindo velocidades próximas ao limite físico de leitura/escrita do barramento do sistema. Suporta indexação de eixos negativos (ex: -1 mapeia o último eixo).

Parâmetros:

array<ZTensor> $tensors: Uma lista contendo arrays PHP nativos ou instâncias do ZTensor que serão unificados.

int $axis (Opcional): O eixo coordenado alvo da fusão espacial. O padrão é 0.

Retorno: ZTensor — Um novo tensor denso e unificado contendo o shape expandido no eixo selecionado.

Exceções: Lança \Exception se a lista estiver vazia ou se houver incompatibilidade dimensional de formato (shape mismatch) entre os tensores participantes.

```php
use ZMatrix\ZTensor;

$t1 = ZTensor::arr([[1, 2], [3, 4]]); // [2x2]$t2 = ZTensor::arr([[5, 6]]);        // [1x2]

// Concatenação vertical (Eixo 0 - Linhas)
$vertical = ZTensor::concat([$t1,$t2], 0);
print_r($vertical->shape()); // Output: [3, 2]
echo $vertical->key([2, 1]); // Output: 6.0

// Concatenação horizontal (Eixo 1 - Colunas)
$t3 = ZTensor::arr([[10], [20]]); // [2x1]$horizontal = ZTensor::concat([$t1,$t3], 1);
print_r($horizontal->shape()); // Output: [2, 3]

```


---

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

```

```