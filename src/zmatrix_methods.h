#ifndef ZMATRIX_METHODS_H
#define ZMATRIX_METHODS_H

/* =========================================================================
 * ZMatrix PHP Method Declarations
 * ========================================================================= */
// ==========================================================================
// Métodos da Classe ZMatrix\ZTensor (PHP_METHOD)
// ==========================================================================
 PHP_METHOD(ZTensor, __construct)
 {
     zval *data_or_shape_zv = nullptr;
     zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);

     ZEND_PARSE_PARAMETERS_START(0, 1)
         Z_PARAM_OPTIONAL
         Z_PARAM_ZVAL(data_or_shape_zv)
     ZEND_PARSE_PARAMETERS_END();

     if (self_obj->tensor != nullptr) {
         delete self_obj->tensor;
         self_obj->tensor = nullptr;
     }

     try {
         if (data_or_shape_zv == nullptr) {
             self_obj->tensor = new ZTensor();
         } else if (Z_TYPE_P(data_or_shape_zv) == IS_ARRAY) {
             self_obj->tensor = new ZTensor(php_array_to_tensor(data_or_shape_zv));
         } else if (Z_TYPE_P(data_or_shape_zv) == IS_OBJECT &&
                    instanceof_function(Z_OBJCE_P(data_or_shape_zv), zmatrix_ce_ZTensor)) {
             zmatrix_ztensor_object *other_obj = Z_MATRIX_ZTENSOR_P(data_or_shape_zv);
             if (other_obj->tensor != nullptr) {
                 self_obj->tensor = new ZTensor(*other_obj->tensor); // Cria uma cópia do tensor
             } else {
                 throw std::invalid_argument("Objeto ZTensor fornecido não está inicializado.");
             }
         } else {
             throw std::invalid_argument("Construtor ZTensor aceita apenas ZTensor, array ou nenhum argumento.");
         }
     } catch (const std::exception& e) {
         if (self_obj->tensor != nullptr) {
             delete self_obj->tensor;
             self_obj->tensor = nullptr;
         }
         zend_throw_exception(zend_ce_exception, e.what(), 0);
     }
 }

PHP_METHOD(ZTensor, arr)
{
    zval *input_zv;

    // Parseia os parâmetros: esperamos 1 argumento.
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(input_zv) // Pega o argumento como um zval genérico
    ZEND_PARSE_PARAMETERS_END();

    try {
        if (Z_TYPE_P(input_zv) == IS_ARRAY) {
            // Caso 1: O input é um array PHP
            ZTensor result_tensor = php_array_to_tensor(input_zv); //
            zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor); //
        } else if (Z_TYPE_P(input_zv) == IS_OBJECT &&
                   instanceof_function(Z_OBJCE_P(input_zv), zmatrix_ce_ZTensor)) {
            // Caso 2: O input é um objeto ZTensor
            zmatrix_ztensor_object *other_obj = Z_MATRIX_ZTENSOR_P(input_zv);
            if (other_obj->tensor != nullptr) {
                // Cria uma NOVA instância de ZTensor (C++) como uma cópia do tensor interno do objeto fornecido
                ZTensor new_copied_tensor = ZTensor(*other_obj->tensor); // Chama o construtor de cópia de ZTensor C++
                zmatrix_return_tensor_obj(new_copied_tensor, return_value, zmatrix_ce_ZTensor); //
            } else {
                // O objeto ZTensor fornecido não foi inicializado internamente
                zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
                RETURN_THROWS();
            }
        } else {
            // Tipo de argumento inválido
            zend_throw_exception_ex(zend_ce_type_error, 0, "ZTensor::arr() expects parameter 1 to be array or ZTensor, %s given", zend_zval_type_name(input_zv));
            RETURN_THROWS();
        }
    } catch (const std::exception& e) {
        // Captura exceções C++ de php_array_to_tensor, construtor de cópia de ZTensor, ou zmatrix_return_tensor_obj
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, add)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    // 1) self tensor
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }
    ZTensor &A = *self_obj->tensor;

    // 2) Escalar (int or float)?
    if (Z_TYPE_P(other_zv) == IS_LONG || Z_TYPE_P(other_zv) == IS_DOUBLE) {
        float scalar = (Z_TYPE_P(other_zv) == IS_LONG
            ? (float)Z_LVAL_P(other_zv)
            : (float)Z_DVAL_P(other_zv));
        A.scalar_add(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // 3) Tensor/array
    ZTensor *other_ptr = nullptr, tmp_other;
    if (!zmatrix_get_tensor_ptr(other_zv, other_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }
    ZTensor &B = *other_ptr;
    const auto &shapeA = A.shape, &shapeB = B.shape;

    // 4) Vetor‑1D de tamanho 1 → escalar
    if (shapeB.size()==1 && shapeB[0]==1) {
        float scalar = B.data.data()[0];
        A.scalar_add(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // A partir daqui passamos a usar try/catch para que qualquer std::runtime_error
    // seja convertida em exceção PHP
    try {
        // 5) Same shape
        if (shapeA == shapeB) {
            A.add(B);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 6) Broadcast 2D×1D
        if (shapeA.size()==2 && shapeB.size()==1 && shapeB[0]==shapeA[1]) {
            size_t M=shapeA[0], N=shapeA[1];
            ZTensor C(shapeA);
            float *cd=C.data.data(), *bd=(float*)B.data.data();
            for(size_t i=0;i<M;++i){
                memcpy(cd+i*N, bd, N*sizeof(float));
            }
            A.add(C);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 7) Broadcast inverso → exceção PHP
        // 7) Broadcast reverso: B é maior que A mas compatível
        if (shapeB.size() == 2 && shapeA.size() == 1 && shapeA[0] == shapeB[1]) {
            size_t M = shapeB[0], N = shapeB[1];
            ZTensor expandedA(shapeB);
            float *dst = expandedA.data.data();
            const float *src = A.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(dst + i * N, src, N * sizeof(float));
            }
            A = expandedA; // substitui A com broadcast
            A.add(B);      // executa soma
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        zend_throw_exception(zend_ce_exception,
            "For reverse broadcasting, call B->add(A) instead of A->add(B)", 0);
        RETURN_THROWS();
    }
    catch(const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

// ------------------------------------------------------------
// PHP_METHOD(ZTensor, sub)
PHP_METHOD(ZTensor, sub)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    // 1) self tensor
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }
    ZTensor &A = *self_obj->tensor;

    // 2) Escalar?
    if (Z_TYPE_P(other_zv)==IS_LONG||Z_TYPE_P(other_zv)==IS_DOUBLE) {
        float scalar = (Z_TYPE_P(other_zv)==IS_LONG
            ? (float)Z_LVAL_P(other_zv)
            : (float)Z_DVAL_P(other_zv));
        A.scalar_subtract(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // 3) Tensor/array
    ZTensor *other_ptr=nullptr, tmp_other;
    if(!zmatrix_get_tensor_ptr(other_zv, other_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }
    ZTensor &B=*other_ptr;
    const auto &shapeA=A.shape, &shapeB=B.shape;

    // 4) Vetor‑1D de tamanho 1 → escalar
    if(shapeB.size()==1 && shapeB[0]==1) {
        float scalar=B.data.data()[0];
        A.scalar_subtract(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    try {
        // 5) Same shape
        if (shapeA == shapeB) {
            A.subtract(B);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 6) Broadcast 2D×1D: A [M×N], B [N]
        if (shapeA.size() == 2 && shapeB.size() == 1 && shapeB[0] == shapeA[1]) {
            size_t M = shapeA[0], N = shapeA[1];
            ZTensor C(shapeA);
            float *cd = C.data.data(), *bd = B.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cd + i * N, bd, N * sizeof(float));
            }
            A.subtract(C);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 7) Broadcast reverso: A [N], B [M×N]
        if (shapeA.size() == 1 && shapeB.size() == 2 && shapeA[0] == shapeB[1]) {
            size_t M = shapeB[0], N = shapeB[1];
            ZTensor C(shapeB);
            float *cd = C.data.data();
            const float *ad = A.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cd + i * N, ad, N * sizeof(float));
            }
            C.subtract(B); // A (broadcastado) - B
            A = C;
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 8) Outros casos incompatíveis
        zend_throw_exception(zend_ce_exception,
            "Shapes incompatíveis para sub() com broadcast", 0);
        RETURN_THROWS();
    }
    catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, mul)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }
    ZTensor &A = *self_obj->tensor;

    // 1) Caso escalar (int ou float)
    if (Z_TYPE_P(other_zv) == IS_LONG || Z_TYPE_P(other_zv) == IS_DOUBLE) {
        float scalar = (Z_TYPE_P(other_zv) == IS_LONG)
            ? (float)Z_LVAL_P(other_zv)
            : (float)Z_DVAL_P(other_zv);
        A.multiply_scalar(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // 2) Caso tensor/array
    ZTensor *other_ptr = nullptr, tmp_other;
    if (!zmatrix_get_tensor_ptr(other_zv, other_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }
    ZTensor &B = *other_ptr;
    const auto &shapeA = A.shape, &shapeB = B.shape;

    // 2a) B é vetor 1D de tamanho 1 → escalar
    if (shapeB.size() == 1 && shapeB[0] == 1) {
        float scalar = B.data.data()[0];
        A.multiply_scalar(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // Bloco try para capturar qualquer std::exception e converter em exceção PHP
    try {
        // 3) Mesmos formatos → operação direta
        if (shapeA == shapeB) {
            A.mul(B);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 4) Broadcast linha: A [M×N], B [N]
        if (shapeA.size() == 2 && shapeB.size() == 1 && shapeB[0] == shapeA[1]) {
            size_t M = shapeA[0], N = shapeA[1];
            ZTensor C(shapeA);
            float *cdat = C.data.data();
            const float *bdat = B.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cdat + i * N, bdat, N * sizeof(float));
            }
            A.mul(C);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 5) Broadcast reverso: A [N], B [M×N]
        if (shapeA.size() == 1 && shapeB.size() == 2 && shapeA[0] == shapeB[1]) {
            size_t M = shapeB[0], N = shapeB[1];
            ZTensor C(shapeB);
            float *cdat = C.data.data();
            const float *adat = A.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cdat + i * N, adat, N * sizeof(float));
            }
            C.mul(B);
            A = C; // resultado da operação substitui A
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 6) Outros casos não suportados
        zend_throw_exception(zend_ce_exception,
            "Shapes incompatíveis para mul() com broadcast", 0);
        RETURN_THROWS();
    } catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, scalarMultiply)
{
    double scalar;
   ZEND_PARSE_PARAMETERS_START(1, 1) Z_PARAM_DOUBLE(scalar) ZEND_PARSE_PARAMETERS_END();
   zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
   if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
   try { self_obj->tensor->multiply_scalar(static_cast<float>(scalar)); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
   catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, transpose)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { ZTensor result = self_obj->tensor->transpose(); zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor); }
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, abs)
{
   zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
   if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
   try { self_obj->tensor->abs(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
   catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, sigmoid)
{
   zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
   if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
   try { self_obj->tensor->sigmoid(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
   catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, sigmoidDerivative)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    try {
        self_obj->tensor->sigmoid_derivative();
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, toGpu)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }
#ifdef HAVE_CUDA
    try {
        self_obj->tensor->to_gpu();
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
#else
    zend_throw_exception(zend_ce_exception, "CUDA support not available", 0);
    RETURN_THROWS();
#endif
}

PHP_METHOD(ZTensor, toCpu)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }
#ifdef HAVE_CUDA
    try {
        self_obj->tensor->to_cpu();
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
#else
    zend_throw_exception(zend_ce_exception, "CUDA support not available", 0);
    RETURN_THROWS();
#endif
}

PHP_METHOD(ZTensor, isOnGpu)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }
#ifdef HAVE_CUDA
    RETURN_BOOL(self_obj->tensor->is_on_gpu());
#else
    RETURN_BOOL(0);
#endif
}


// --- Métodos de Redução (sum, mean, min, max, std - global) ---
PHP_METHOD(ZTensor, sumtotal)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { RETURN_DOUBLE(self_obj->tensor->sum()); }
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, mean)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { RETURN_DOUBLE(self_obj->tensor->mean()); }
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, min)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { RETURN_DOUBLE(static_cast<double>(self_obj->tensor->min())); } // Cast float->double
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, max)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { RETURN_DOUBLE(self_obj->tensor->max()); }
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, std)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { RETURN_DOUBLE(static_cast<double>(self_obj->tensor->std())); } // Cast float->double
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}


// --- Métodos de Propriedade/Informação ---
PHP_METHOD(ZTensor, shape)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    array_init(return_value);
    for (size_t dim : self_obj->tensor->shape) { add_next_index_long(return_value, dim); }
}

PHP_METHOD(ZTensor, ndim)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    RETURN_LONG(self_obj->tensor->shape.size());
}

PHP_METHOD(ZTensor, size)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    RETURN_LONG(self_obj->tensor->size());
}

PHP_METHOD(ZTensor, isEmpty)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    RETURN_BOOL(self_obj->tensor == nullptr || self_obj->tensor->empty());
}

PHP_METHOD(ZTensor, toArray)
{
     ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { tensor_to_php_array(*(self_obj->tensor), return_value); }
    catch (const std::exception& e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, zeros)
{
    zval *shape_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(shape_zv)
    ZEND_PARSE_PARAMETERS_END();

    std::vector<size_t> shape;
    HashTable *ht = Z_ARRVAL_P(shape_zv);
    zval *dim_zv;

    ZEND_HASH_FOREACH_VAL(ht, dim_zv) {
        if (Z_TYPE_P(dim_zv) != IS_LONG || Z_LVAL_P(dim_zv) < 0) {
            zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_INVALID_SHAPE, 0);
            RETURN_THROWS();
        }
        shape.push_back(Z_LVAL_P(dim_zv));
    } ZEND_HASH_FOREACH_END();

    if (shape.empty()) {
        zend_throw_exception(zend_ce_exception, "Shape cannot be empty for zeros", 0);
        RETURN_THROWS();
    }

    try {
        ZTensor result = ZTensor::zeros(shape);  // <-- CORRETO AQUI
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}


// Método estático para criar tensor preenchido com valor específico
PHP_METHOD(ZTensor, full)
{
    zval *shape_zv;
    double value;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY(shape_zv)
        Z_PARAM_DOUBLE(value)
    ZEND_PARSE_PARAMETERS_END();

    std::vector<size_t> shape;
    HashTable *ht = Z_ARRVAL_P(shape_zv);
    zval *dim_zv;

    ZEND_HASH_FOREACH_VAL(ht, dim_zv) {
        if (Z_TYPE_P(dim_zv) != IS_LONG || Z_LVAL_P(dim_zv) < 0) {
            zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_INVALID_SHAPE, 0);
            RETURN_THROWS();
        }
        shape.push_back(Z_LVAL_P(dim_zv));
    } ZEND_HASH_FOREACH_END();

    if (shape.empty()) {
        zend_throw_exception(zend_ce_exception, "Shape cannot be empty for full", 0);
        RETURN_THROWS();
    }

    try {
        ZTensor result = ZTensor::full(shape, static_cast<float>(value)); // ✅ chama a versão otimizada
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, identity)
{
     zend_long size;
     ZEND_PARSE_PARAMETERS_START(1, 1)
         Z_PARAM_LONG(size)
     ZEND_PARSE_PARAMETERS_END();
     if (size <= 0) {
         zend_throw_exception(zend_ce_exception, "Identity size must be positive", 0);
         RETURN_THROWS();
     }
     ZTensor result = ZTensor::identity(static_cast<size_t>(size));
     zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
}

// --- Método estático ZTensor::random otimizado ---
PHP_METHOD(ZTensor, random)
{
    zval *shape_zv;
    double min_val = 0.0, max_val = 1.0;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ARRAY(shape_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(min_val)
        Z_PARAM_DOUBLE(max_val)
    ZEND_PARSE_PARAMETERS_END();

    std::vector<size_t> shape;
    HashTable *ht = Z_ARRVAL_P(shape_zv);
    zval *dim_zv;

    ZEND_HASH_FOREACH_VAL(ht, dim_zv) {
        if (Z_TYPE_P(dim_zv) != IS_LONG || Z_LVAL_P(dim_zv) < 0) {
            zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_INVALID_SHAPE, 0);
            RETURN_THROWS();
        }
        shape.push_back(Z_LVAL_P(dim_zv));
    } ZEND_HASH_FOREACH_END();

    if (min_val > max_val) {
        zend_throw_exception(zend_ce_exception, "Minimum value cannot be greater than maximum in random", 0);
        RETURN_THROWS();
    }

    try {
        ZTensor result = ZTensor::random(shape, static_cast<float>(min_val), static_cast<float>(max_val));
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}
// --- Implementação do Método matmul ---
PHP_METHOD(ZTensor, matmul)
{
    zval *other_zv;
    // zend_bool use_blas = 1; // Opcional BLAS (não usado nesta implementação C++ simples)

    // Parseia o argumento 'other'
    ZEND_PARSE_PARAMETERS_START(1, 1) // Apenas 1 argumento obrigatório por enquanto
        Z_PARAM_ZVAL(other_zv)
        // Z_PARAM_OPTIONAL // Descomentar se adicionar use_blas
        // Z_PARAM_BOOL(use_blas)
    ZEND_PARSE_PARAMETERS_END();

    // Obtém o ponteiro para o tensor interno do objeto atual ($this)
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    // Verifica se o tensor interno está inicializado
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS(); // Retorna indicando que uma exceção foi lançada
    }

    // Obtém o ponteiro para o tensor do argumento 'other', convertendo de array se necessário
    ZTensor *other_ptr = nullptr;
    ZTensor tmp_other; // Armazena tensor temporário se 'other' for array
    if (!zmatrix_get_tensor_ptr(other_zv, other_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS(); // Retorna se a obtenção/conversão falhar
    }

    try {
        // Chama o método matmul da classe C++ ZTensor
        // Este método C++ deve conter a lógica real da multiplicação,
        // incluindo verificações de shape e a computação.
        ZTensor result = self_obj->tensor->matmul(*other_ptr);

        // Cria um novo objeto PHP ZTensor para retornar o resultado
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);

    } catch (const std::exception& e) {
        // Captura exceções C++ (ex: shape incompatível, tensor vazio) e as lança como exceções PHP
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS(); // Retorna indicando que uma exceção foi lançada
    }
}

PHP_METHOD(ZTensor, divide)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    // 1) Pega o objeto interno
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }
    ZTensor &A = *self_obj->tensor;

    // 2) Caso ESCALAR (int ou float)
    if (Z_TYPE_P(other_zv) == IS_LONG || Z_TYPE_P(other_zv) == IS_DOUBLE) {
        float scalar = (Z_TYPE_P(other_zv) == IS_LONG)
            ? static_cast<float>(Z_LVAL_P(other_zv))
            : static_cast<float>(Z_DVAL_P(other_zv));
        A.scalar_divide(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // 3) Caso tensor/array
    ZTensor *other_ptr = nullptr, tmp_other;
    if (!zmatrix_get_tensor_ptr(other_zv, other_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }
    ZTensor &B = *other_ptr;
    const auto &shapeA = A.shape, &shapeB = B.shape;

    // 4) Vetor‑1D de tamanho 1 → trate como escalar
    if (shapeB.size() == 1 && shapeB[0] == 1) {
        float scalar = B.data.data()[0];
        A.scalar_divide(scalar);
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
        return;
    }

    // 5) Agora o bloco try/catch para os casos de tensor×tensor e broadcast
    try {
        // 5.1) Mesmos formatos → divisão direta
        if (shapeA == shapeB) {
            A.divide(B);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 5.2) Broadcast 2D×1D: A [M×N] ÷ B [N]
        if (shapeA.size() == 2 && shapeB.size() == 1 && shapeB[0] == shapeA[1]) {
            size_t M = shapeA[0], N = shapeA[1];
            ZTensor C(shapeA);
            float *cdat = C.data.data();
            const float *bdat = B.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cdat + i * N, bdat, N * sizeof(float));
            }
            A.divide(C);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 5.3) Broadcast inverso: A [N], B [M×N]
        if (shapeA.size() == 1 && shapeB.size() == 2 && shapeA[0] == shapeB[1]) {
            size_t M = shapeB[0], N = shapeB[1];
            ZTensor C(shapeB);
            float *cdat = C.data.data();
            const float *adat = A.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cdat + i * N, adat, N * sizeof(float));
            }
            C.divide(B); // A (broadcastado) ÷ B
            A = C;
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 5.4) Outros casos incompatíveis
        zend_throw_exception(zend_ce_exception,
            "Incompatible shapes for divide() with broadcasting", 0);
        RETURN_THROWS();
    }
    catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, pow)
{
    double exponent;
    ZEND_PARSE_PARAMETERS_START(1,1) Z_PARAM_DOUBLE(exponent) ZEND_PARSE_PARAMETERS_END();
    zmatrix_ztensor_object *self = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED,0); RETURN_THROWS(); }
    try { self->tensor->pow((float)exponent); ZVAL_ZVAL(return_value,ZEND_THIS,1,0);} catch(const std::exception &e){ zend_throw_exception(zend_ce_exception,e.what(),0); RETURN_THROWS(); }

}

PHP_METHOD(ZTensor, relu)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { self_obj->tensor->relu(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
    catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, tanh)
{
   zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
   if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
   try { self_obj->tensor->tanh(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
   catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, exp)
{
    // Implementação similar a ReLU, usando .exp()
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if(!self->tensor){ zend_throw_exception(zend_ce_exception,ZMATRIX_ERR_NOT_INITIALIZED,0); RETURN_THROWS(); }
    try{ self->tensor->exp(); ZVAL_ZVAL(return_value,ZEND_THIS,1,0);}catch(const std::exception &e){ zend_throw_exception(zend_ce_exception,e.what(),0); RETURN_THROWS(); }

}

PHP_METHOD(ZTensor, log)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if(!self->tensor){ zend_throw_exception(zend_ce_exception,ZMATRIX_ERR_NOT_INITIALIZED,0); RETURN_THROWS(); }
    try{
        self->tensor->log(); // Chama o método void ZTensor::log()
        ZVAL_ZVAL(return_value,ZEND_THIS,1,0); // Retorna $this
    }catch(const std::exception &e){ zend_throw_exception(zend_ce_exception,e.what(),0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, sqrt)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if(!self->tensor){ zend_throw_exception(zend_ce_exception,ZMATRIX_ERR_NOT_INITIALIZED,0); RETURN_THROWS(); }
    try{
        self->tensor->sqrt(); // Chama o método void ZTensor::sqrt()
        ZVAL_ZVAL(return_value,ZEND_THIS,1,0); // Retorna $this
    }catch(const std::exception &e){ zend_throw_exception(zend_ce_exception,e.what(),0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, reluDerivative)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { self_obj->tensor->relu_derivative(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
    catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

// Tanh
PHP_METHOD(ZTensor, tanhDerivative)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { self_obj->tensor->tanh_derivative(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
    catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

// Leaky ReLU (com alpha = 0.01 por padrão)
PHP_METHOD(ZTensor, leakyRelu)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }

    double alpha = 0.01;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(alpha)
    ZEND_PARSE_PARAMETERS_END();

    try { self_obj->tensor->leaky_relu(static_cast<float>(alpha)); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
    catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, leakyReluDerivative)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }

    double alpha = 0.01;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(alpha)
    ZEND_PARSE_PARAMETERS_END();

    try { self_obj->tensor->leaky_relu_derivative(static_cast<float>(alpha)); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
    catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, softmaxDerivative)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) { zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0); RETURN_THROWS(); }
    try { self_obj->tensor->softmax_derivative(); ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); }
    catch (const std::exception &e) { zend_throw_exception(zend_ce_exception, e.what(), 0); RETURN_THROWS(); }
}

PHP_METHOD(ZTensor, softmax)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    try {
        self_obj->tensor->softmax();  // agora void
        ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);  // retorna o próprio objeto
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

/* PHP method implementation */
PHP_METHOD(ZTensor, reshape)
{
    zval *zshape;
    HashTable *ht_shape;
    zval *val;
    std::vector<size_t> new_shape;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(zshape)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    /* 1. Extrai HashTable do array PHP */
    ht_shape = Z_ARRVAL_P(zshape);

    /* 2. Converte cada elemento em size_t */
    ZEND_HASH_FOREACH_VAL(ht_shape, val) {
        if (Z_TYPE_P(val) != IS_LONG) {
            zend_throw_exception(zend_ce_exception, "All dimensions must be integers", 0);
            RETURN_THROWS();
        }
        zend_long dim = Z_LVAL_P(val);
        if (dim < 0) {
            zend_throw_exception(zend_ce_exception, "Dimensions must be non-negative", 0);
            RETURN_THROWS();
        }
        new_shape.push_back((size_t) dim);
    } ZEND_HASH_FOREACH_END();

    /* 3. Chama o método C++ reshape */
    ZTensor reshaped = self_obj->tensor->reshape(new_shape);

    /* 4. Empacota o resultado em um novo objeto PHP ZTensor */
    object_init_ex(return_value, zmatrix_ce_ZTensor);
    zmatrix_ztensor_object *res_obj = Z_MATRIX_ZTENSOR_P(return_value);
    res_obj->tensor = new ZTensor(std::move(reshaped));
}

PHP_METHOD(ZTensor, randn)
{
    zval *shape_zv;
    double mean = 0.0;
    double std_dev = 1.0;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ARRAY(shape_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(mean)
        Z_PARAM_DOUBLE(std_dev)
    ZEND_PARSE_PARAMETERS_END();

    if (std_dev < 0) {
        zend_throw_exception(zend_ce_exception, "Standard deviation (std_dev) cannot be negative for randn", 0);
        RETURN_THROWS();
    }

    std::vector<size_t> shape_vec;
    // Reutilize sua função zmatrix_zval_to_shape ou implemente a lógica aqui
    // Inicio da lógica de zmatrix_zval_to_shape (simplificado, adicione sua validação robusta)
    HashTable *ht_shape = Z_ARRVAL_P(shape_zv);
    zval *dim_val_zv;
    ZEND_HASH_FOREACH_VAL(ht_shape, dim_val_zv) {
        if (Z_TYPE_P(dim_val_zv) != IS_LONG || Z_LVAL_P(dim_val_zv) < 0) {
            zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_INVALID_SHAPE, 0);
            RETURN_THROWS();
        }
        shape_vec.push_back(Z_LVAL_P(dim_val_zv));
    } ZEND_HASH_FOREACH_END();
    if (shape_vec.empty() && zend_hash_num_elements(ht_shape) > 0) { // Caso de array não vazio mas sem longs válidos
         zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_INVALID_SHAPE, 0);
         RETURN_THROWS();
    }
    // Fim da lógica de zmatrix_zval_to_shape

    try {
        ZTensor result(shape_vec);
        size_t total_size = result.size();
        if (total_size > 0) {
            float* data_ptr = result.data.data();
            std::normal_distribution<float> dist(static_cast<float>(mean), static_cast<float>(std_dev));

            #if HAS_OPENMP
            if (total_size > ZMATRIX_PARALLEL_THRESHOLD) {
                // Cada thread precisa de seu próprio estado de gerador para paralelismo seguro de PRNG.
                // Uma abordagem simples para loops é ter um gerador por thread.
                #pragma omp parallel
                {
                    // Seed local para cada thread para evitar que todas produzam a mesma sequência
                    unsigned int seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^ (omp_get_thread_num() << 16);
                    std::mt19937 thread_local_gen(seed);
                    #pragma omp for schedule(static)
                    for (size_t i = 0; i < total_size; ++i) {
                        data_ptr[i] = dist(thread_local_gen);
                    }
                }
            } else {
                std::mt19937& main_gen = get_global_mt19937(); // Para o caso sequencial
                for (size_t i = 0; i < total_size; ++i) {
                    data_ptr[i] = dist(main_gen);
                }
            }
            #else
            std::mt19937& main_gen = get_global_mt19937();
            for (size_t i = 0; i < total_size; ++i) {
                data_ptr[i] = dist(main_gen);
            }
            #endif
        }
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, arange)
{
    double arg1_val; // Pode ser start ou stop
    zval *z_arg2 = nullptr; // Pode ser stop ou null
    zval *z_arg3 = nullptr; // Pode ser step ou null

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_DOUBLE(arg1_val)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_EX(z_arg2, 1, 0) // allow_null = true
        Z_PARAM_ZVAL_EX(z_arg3, 1, 0) // allow_null = true
    ZEND_PARSE_PARAMETERS_END();

    float start_val, stop_val, step_val;

    int argc = ZEND_NUM_ARGS();

    if (argc == 1) { // arange(stop)
        start_val = 0.0f;
        stop_val = static_cast<float>(arg1_val);
        step_val = 1.0f;
    } else if (argc == 2) { // arange(start, stop) ou arange(stop, step=null) - o segundo caso não é típico.
                            // A API PHP é arange(float $start_or_stop, ?float $stop = null, float $step = 1.0)
                            // Se z_arg2 (stop) é null, então arg1_val é stop, start=0, step=1 (coberto por argc=1 se for assim)
                            // Se z_arg2 (stop) NÃO é null, então arg1_val é start, z_arg2 é stop.
        start_val = static_cast<float>(arg1_val);
        stop_val = static_cast<float>(zval_get_double(z_arg2)); // z_arg2 deve ter sido passado
        step_val = 1.0f;
    } else { // argc == 3, arange(start, stop, step)
        start_val = static_cast<float>(arg1_val);
        stop_val = static_cast<float>(zval_get_double(z_arg2)); // z_arg2 deve ter sido passado
        step_val = static_cast<float>(zval_get_double(z_arg3)); // z_arg3 deve ter sido passado
    }


    if (step_val == 0.0f) {
        zend_throw_exception(zend_ce_exception, "Step cannot be zero for arange", 0);
        RETURN_THROWS();
    }

    std::vector<float> values;
    if (step_val > 0) {
        if (start_val < stop_val) { // Somente adiciona se o intervalo for válido
            for (float current_val = start_val; current_val < stop_val; current_val += step_val) {
                values.push_back(current_val);
            }
        }
    } else { // step_val < 0
        if (start_val > stop_val) { // Somente adiciona se o intervalo for válido
            for (float current_val = start_val; current_val > stop_val; current_val += step_val) {
                values.push_back(current_val);
            }
        }
    }

    size_t count = values.size();

    try {
        ZTensor result_tensor(std::vector<size_t>{count});
        if (count > 0) {
            float* data_ptr = result_tensor.data.data();
            // std::copy é geralmente eficiente para esta tarefa
            // A paralelização de std::copy ou um loop manual aqui para 'arange' pode não ser
            // tão benéfica quanto para operações mais intensas, dado que 'values' já foi construído.
            // Se 'count' for extremamente grande, e a construção de 'values' for o gargalo,
            // a lógica de cálculo de 'count' e preenchimento direto do tensor seria mais complexa
            // mas potencialmente paralelizável.
            std::copy(values.begin(), values.end(), data_ptr);
        }
        zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, linspace)
{
    double start_val, stop_val;
    zend_long num_l = 50; // Renomeado para evitar conflito com a variável num
    zend_bool endpoint_val = 1; // true

    ZEND_PARSE_PARAMETERS_START(2, 4)
        Z_PARAM_DOUBLE(start_val)
        Z_PARAM_DOUBLE(stop_val)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(num_l)
        Z_PARAM_BOOL(endpoint_val)
    ZEND_PARSE_PARAMETERS_END();

    if (num_l < 0) {
        zend_throw_exception(zend_ce_exception, "Number of samples (num) cannot be negative for linspace", 0);
        RETURN_THROWS();
    }

    size_t num = static_cast<size_t>(num_l); // Usar size_t para o tamanho

    if (num == 0) {
        ZTensor result_tensor(std::vector<size_t>{0});
        zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
        return;
    }

    try {
        ZTensor result_tensor(std::vector<size_t>{num});
        // result.size() já retornaria num se num > 0, ou 0 se num == 0 (coberto acima)
        // Se num > 0, data_ptr será válido.

        if (num > 0) { // Garante que só acessamos data_ptr se houver elementos
            float* data_ptr = result_tensor.data.data();

            if (num == 1) {
                data_ptr[0] = static_cast<float>(start_val);
            } else {
                float step;
                if (endpoint_val) {
                    step = (static_cast<float>(stop_val) - static_cast<float>(start_val)) / (num - 1);
                } else {
                    step = (static_cast<float>(stop_val) - static_cast<float>(start_val)) / num;
                }

                #if HAS_OPENMP
                if (num > ZMATRIX_PARALLEL_THRESHOLD) {
                    #pragma omp parallel for schedule(static) // Removido simd por enquanto
                    for (size_t i = 0; i < num; ++i) {
                        data_ptr[i] = static_cast<float>(start_val) + i * step;
                    }
                } else {
                    for (size_t i = 0; i < num; ++i) {
                        data_ptr[i] = static_cast<float>(start_val) + i * step;
                    }
                }
                #else
                for (size_t i = 0; i < num; ++i) {
                    data_ptr[i] = static_cast<float>(start_val) + i * step;
                }
                #endif
                if (endpoint_val && num > 1) { // Garante que o último ponto seja exatamente stop_val
                     data_ptr[num - 1] = static_cast<float>(stop_val);
                }
            }
        }
        zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, logspace)
{
    double start_val, stop_val, base_val = 10.0;
    zend_long num_l = 50; // Renomeado
    zend_bool endpoint_val = 1; // true

    ZEND_PARSE_PARAMETERS_START(2, 5)
        Z_PARAM_DOUBLE(start_val)
        Z_PARAM_DOUBLE(stop_val)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(num_l)
        Z_PARAM_BOOL(endpoint_val)
        Z_PARAM_DOUBLE(base_val)
    ZEND_PARSE_PARAMETERS_END();

    if (num_l < 0) {
        zend_throw_exception(zend_ce_exception, "Number of samples (num) cannot be negative for logspace", 0);
        RETURN_THROWS();
    }

    size_t num = static_cast<size_t>(num_l); // Usar size_t

    if (num == 0) {
        ZTensor result_tensor(std::vector<size_t>{0});
        zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
        return;
    }

    try {
        ZTensor result_tensor(std::vector<size_t>{num});

        if (num > 0) { // Garante que só acessamos data_ptr se houver elementos
            float* data_ptr = result_tensor.data.data();

            if (num == 1) {
                 data_ptr[0] = std::pow(static_cast<float>(base_val), static_cast<float>(start_val));
            } else {
                float step; // Expoente step
                if (endpoint_val) {
                    step = (static_cast<float>(stop_val) - static_cast<float>(start_val)) / (num - 1);
                } else {
                    step = (static_cast<float>(stop_val) - static_cast<float>(start_val)) / num;
                }

                #if HAS_OPENMP
                if (num > ZMATRIX_PARALLEL_THRESHOLD) {
                    #pragma omp parallel for schedule(static) // Removido simd por enquanto
                    for (size_t i = 0; i < num; ++i) {
                        data_ptr[i] = std::pow(static_cast<float>(base_val), static_cast<float>(start_val) + i * step);
                    }
                } else {
                    for (size_t i = 0; i < num; ++i) {
                        data_ptr[i] = std::pow(static_cast<float>(base_val), static_cast<float>(start_val) + i * step);
                    }
                }
                #else
                for (size_t i = 0; i < num; ++i) {
                    data_ptr[i] = std::pow(static_cast<float>(base_val), static_cast<float>(start_val) + i * step);
                }
                #endif
                 if (endpoint_val && num > 1) { // Garante que o último ponto seja exatamente base^stop
                     data_ptr[num - 1] = std::pow(static_cast<float>(base_val), static_cast<float>(stop_val));
                }
            }
        }
        zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, eye)
{
    zend_long N_val, M_val_opt = -1, k_val = 0; // M_val_opt = -1 para indicar não fornecido
    zval *M_zv = nullptr;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_LONG(N_val)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_EX(M_zv, 1, 0) // Permite null
        Z_PARAM_LONG(k_val)
    ZEND_PARSE_PARAMETERS_END();

    if (N_val < 0) {
        zend_throw_exception(zend_ce_exception, "Number of rows N cannot be negative for eye", 0);
        RETURN_THROWS();
    }

    size_t rows = static_cast<size_t>(N_val);
    size_t cols;

    if (M_zv == nullptr || Z_TYPE_P(M_zv) == IS_NULL) {
        cols = rows; // Matriz quadrada se M não for fornecido
    } else {
        M_val_opt = zval_get_long(M_zv);
        if (M_val_opt < 0) {
            zend_throw_exception(zend_ce_exception, "Number of columns M cannot be negative for eye", 0);
            RETURN_THROWS();
        }
        cols = static_cast<size_t>(M_val_opt);
    }

    try {
        // Construtor ZTensor já preenche com zeros
        ZTensor result_tensor({rows, cols});
        size_t total_size_eye = result_tensor.size();

        if (rows > 0 && cols > 0 && total_size_eye > 0) { // Só preenche a diagonal se a matriz não for vazia
            float* data_ptr = result_tensor.data.data();
            // O número de elementos a serem setados na diagonal k é no máximo min(rows, cols)
            // A paralelização aqui pode ter mais overhead do que benefício a menos que rows/cols sejam muito grandes.
            // Usaremos um threshold menor ou específico se necessário, ou o ZMATRIX_PARALLEL_THRESHOLD.
            // O loop itera 'rows' vezes, mas a condição interna limita as escritas.
            #if HAS_OPENMP
            if (rows > ZMATRIX_PARALLEL_THRESHOLD / (cols > 0 ? std::min((size_t)100, cols) : 100) ) { // Heurística para paralelizar
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < rows; ++i) {
                    zend_long j_long = static_cast<zend_long>(i) + k_val; // k_val pode ser negativo
                    if (j_long >= 0 && static_cast<size_t>(j_long) < cols) {
                        data_ptr[i * cols + static_cast<size_t>(j_long)] = 1.0f;
                    }
                }
            } else {
                for (size_t i = 0; i < rows; ++i) {
                    zend_long j_long = static_cast<zend_long>(i) + k_val;
                    if (j_long >= 0 && static_cast<size_t>(j_long) < cols) {
                        data_ptr[i * cols + static_cast<size_t>(j_long)] = 1.0f;
                    }
                }
            }
            #else
            for (size_t i = 0; i < rows; ++i) {
                zend_long j_long = static_cast<zend_long>(i) + k_val;
                if (j_long >= 0 && static_cast<size_t>(j_long) < cols) {
                    data_ptr[i * cols + static_cast<size_t>(j_long)] = 1.0f;
                }
            }
            #endif
        }
        zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, dot)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    ZTensor *tensor_A = self_obj->tensor;
    ZTensor *tensor_B_ptr = nullptr;
    ZTensor tmp_tensor_B;

    if (!zmatrix_get_tensor_ptr(other_zv, tensor_B_ptr, tmp_tensor_B, zmatrix_ce_ZTensor)) { //
        RETURN_THROWS();
    }
    ZTensor& tensor_B = *tensor_B_ptr;

    try {
        size_t a_ndim = tensor_A->shape.size();
        size_t b_ndim = tensor_B.shape.size();

        // Caso 1: Ambos 1D (produto interno de vetores)
        if (a_ndim == 1 && b_ndim == 1) {
            if (tensor_A->shape[0] == 0 && tensor_B.shape[0] == 0) { // Ambos vazios
                 RETURN_DOUBLE(0.0);
            }
            if (tensor_A->shape[0] != tensor_B.shape[0]) {
                throw std::runtime_error("1D vectors with incompatible shapes for dot product");
            }
            if (tensor_A->shape[0] == 0) { // Ambos tamanho 0 devido à condição anterior
                 RETURN_DOUBLE(0.0);
            }

            float sum_product = 0.0f;
            const float* a_data = tensor_A->data.data();
            const float* b_data = tensor_B.data.data();
            size_t N = tensor_A->shape[0];

            // A redução para float pode ser menos precisa, mas ZTensor usa float.
            // Se precisão dupla for crítica, um acumulador double seria melhor aqui.
            #if HAS_OPENMP
            if (N > ZMATRIX_PARALLEL_THRESHOLD) {
                double omp_sum_product = 0.0; // Use double para redução paralela para precisão
                #pragma omp parallel for reduction(+:omp_sum_product) schedule(static)
                for (size_t i = 0; i < N; ++i) {
                    omp_sum_product += static_cast<double>(a_data[i]) * static_cast<double>(b_data[i]);
                }
                sum_product = static_cast<float>(omp_sum_product);
            } else {
                for (size_t i = 0; i < N; ++i) {
                    sum_product += a_data[i] * b_data[i];
                }
            }
            #else
            for (size_t i = 0; i < N; ++i) {
                sum_product += a_data[i] * b_data[i];
            }
            #endif
            RETURN_DOUBLE(static_cast<double>(sum_product));
        }
        // Caso 2: A é 2D, B é 2D (multiplicação de matrizes)
        else if (a_ndim == 2 && b_ndim == 2) {
            // Delega para o método matmul existente (que já faz validação de shape)
            ZTensor result_tensor = tensor_A->matmul(tensor_B); // matmul já retorna ZTensor
            zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
        }
        // Caso 3: A é 2D, B é 1D (produto matriz-vetor)
        else if (a_ndim == 2 && b_ndim == 1) {
            if (tensor_A->shape.empty() || tensor_B.shape.empty()) {
                 throw std::runtime_error("Empty tensor cannot be used in matrix-vector product");
            }
            if (tensor_A->shape[1] != tensor_B.shape[0]) {
                throw std::runtime_error("Incompatible shapes for matrix-vector product (A.cols != B.rows)");
            }
            if (tensor_A->shape[1] == 0) { // Caso onde A é Mx0 e B é 0x1, resultado é Mx1 de zeros
                ZTensor result_tensor({tensor_A->shape[0]}); // Já zerado pelo construtor
                zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
                return;
            }


            size_t M = tensor_A->shape[0]; // Linhas de A
            size_t K = tensor_A->shape[1]; // Colunas de A (e tamanho de B)

            ZTensor result_tensor({M}); // Resultado é um vetor de tamanho M (já zerado)
            if (M == 0) { // Se A é 0xK, resultado é vetor de tamanho 0
                 zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
                 return;
            }


            const float* a_data = tensor_A->data.data();
            const float* b_data = tensor_B.data.data();
            float* c_data = result_tensor.data.data();

            // C[i] = sum_j (A[i,j] * B[j])
            // Pode usar cblas_sgemv se CBLAS estiver disponível e configurado
            // #ifdef HAVE_CBLAS (ou similar)
            // cblas_sgemv(CblasRowMajor, CblasNoTrans, M, K, 1.0f, a_data, K, b_data, 1, 0.0f, c_data, 1);
            // #else
            // Loop manual se CBLAS não estiver disponível/usado para sgemv
            #if HAS_OPENMP
            if (M * K > ZMATRIX_PARALLEL_THRESHOLD) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < M; ++i) {
                    float row_sum = 0.0f;
                    for (size_t j = 0; j < K; ++j) {
                        row_sum += a_data[i * K + j] * b_data[j];
                    }
                    c_data[i] = row_sum;
                }
            } else {
                 for (size_t i = 0; i < M; ++i) {
                    float row_sum = 0.0f;
                    for (size_t j = 0; j < K; ++j) {
                        row_sum += a_data[i * K + j] * b_data[j];
                    }
                    c_data[i] = row_sum;
                }
            }
            #else
            for (size_t i = 0; i < M; ++i) {
                float row_sum = 0.0f;
                for (size_t j = 0; j < K; ++j) {
                    row_sum += a_data[i * K + j] * b_data[j];
                }
                c_data[i] = row_sum;
            }
            #endif
            // #endif // Fim do else para HAVE_CBLAS (se usado)
            zmatrix_return_tensor_obj(result_tensor, return_value, zmatrix_ce_ZTensor);
        }
        // TODO: Adicionar outros casos (ex: 1D . 2D) ou N-D se necessário
        else {
            throw std::runtime_error("Unsupported shape combination for dot product");
        }

    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, key)
{
    zval *indices_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(indices_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    /* Converte o array PHP de índices para std::vector<size_t> */
    HashTable   *ht_idx = Z_ARRVAL_P(indices_zv);
    zval        *idx_zv;
    std::vector<size_t> indices;
    zend_ulong   idx_key;
    zend_string *str_key;

    /* Evita warning “idx_key set but not used” */
    (void) idx_key;

    ZEND_HASH_FOREACH_KEY_VAL(ht_idx, idx_key, str_key, idx_zv) {
        /* Esperamos índices numéricos sequenciais (0,1,2,…) */
        if (str_key != nullptr) {
            zend_throw_exception(zend_ce_exception,
                "ZTensor::key() accepts only numerically indexed arrays", 0);
            RETURN_THROWS();
        }
        /* Cada valor deve ser inteiro >= 0 */
        if (Z_TYPE_P(idx_zv) != IS_LONG) {
            zend_throw_exception(zend_ce_exception,
                "ZTensor::key() expects each index to be an integer", 0);
            RETURN_THROWS();
        }
        zend_long l = Z_LVAL_P(idx_zv);
        if (l < 0) {
            zend_throw_exception(zend_ce_exception,
                "ZTensor::key() indices must be >= 0", 0);
            RETURN_THROWS();
        }
        indices.push_back((size_t) l);
    } ZEND_HASH_FOREACH_END();

    try {
        /* Acessa o elemento; at() lança std::out_of_range se fora de limites */
        float value = self_obj->tensor->at(indices);
        /* Retorna como double (PHP não tem float puro) */
        RETURN_DOUBLE((double) value);
    } catch (const std::out_of_range &e) {
        zend_throw_exception(zend_ce_exception, "Index out of tensor bounds", 0);
        RETURN_THROWS();
    } catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, ones)
{
    zval *shape_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(shape_zv)
    ZEND_PARSE_PARAMETERS_END();

    std::vector<size_t> shape;
    HashTable *ht = Z_ARRVAL_P(shape_zv);
    zval *dim_zv;
    ZEND_HASH_FOREACH_VAL(ht, dim_zv) {
        if (Z_TYPE_P(dim_zv) != IS_LONG || Z_LVAL_P(dim_zv) < 0) {
            zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_INVALID_SHAPE, 0);
            RETURN_THROWS();
        }
        shape.push_back(Z_LVAL_P(dim_zv));
    } ZEND_HASH_FOREACH_END();

    if (shape.empty()) {
        zend_throw_exception(zend_ce_exception, "Shape cannot be empty for ones", 0);
        RETURN_THROWS();
    }

    try {
        ZTensor result(shape);
        std::fill(result.data.begin(), result.data.end(), 1.0f);
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, clip)
{
    zval *input_zv;
    double min_val, max_val;

    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ZVAL(input_zv)
        Z_PARAM_DOUBLE(min_val)
        Z_PARAM_DOUBLE(max_val)
    ZEND_PARSE_PARAMETERS_END();

    ZTensor *input_tensor = nullptr;
    ZTensor tmp_tensor;

    if (!zmatrix_get_tensor_ptr(input_zv, input_tensor, tmp_tensor, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }

    try {
        ZTensor result = *input_tensor;  // cópia do tensor de entrada

        const size_t N = result.size();
        float * __restrict__ a = result.data.data();
        const float fmin = static_cast<float>(min_val);
        const float fmax = static_cast<float>(max_val);

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < N; ++i) {
                a[i] = std::max(fmin, std::min(fmax, a[i]));
            }
        } else {
            for (size_t i = 0; i < N; ++i) {
                a[i] = std::max(fmin, std::min(fmax, a[i]));
            }
        }
        #else
        for (size_t i = 0; i < N; ++i) {
            a[i] = std::max(fmin, std::min(fmax, a[i]));
        }
        #endif
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, sum)
{
    zval *other_zv;
    zval *axis_zv = nullptr;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(other_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(axis_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    ZTensor *other_ptr = nullptr;
    ZTensor tmp_other;

    if (!zmatrix_get_tensor_ptr(other_zv, other_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }

    try {
        if (!axis_zv || Z_TYPE_P(axis_zv) == IS_NULL) {
            // axis não fornecido → retorna soma total (escalares somados)
            double total = self_obj->tensor->sum();
            ZTensor scalar_tensor({1});
            scalar_tensor.data[0] = static_cast<float>(total);
            zmatrix_return_tensor_obj(scalar_tensor, return_value, zmatrix_ce_ZTensor);
        } else {
            zend_long axis = Z_LVAL_P(axis_zv);
            self_obj->tensor->soma(*other_ptr, static_cast<int>(axis));
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0); // retorno this
        }
    } catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, broadcast)
{
    zval *bias_zv;
        ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(bias_zv)
        ZEND_PARSE_PARAMETERS_END();

        // Obtém o ponteiro para o objeto atual (self) e verifica inicialização
        zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
        if (!self_obj->tensor) {
            zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
            RETURN_THROWS();
        }

        // Obtém o ponteiro para o tensor "bias" (pode ser array ou ZTensor)
        ZTensor *bias_ptr = nullptr;
        ZTensor tmp_bias;
        if (!zmatrix_get_tensor_ptr(bias_zv, bias_ptr, tmp_bias, zmatrix_ce_ZTensor)) {
            RETURN_THROWS();
        }

        try {
            ZTensor &self_tensor = *self_obj->tensor;
            ZTensor &bias_tensor = *bias_ptr;

            const std::vector<size_t> &shapeA = self_tensor.shape;
            const std::vector<size_t> &shapeB = bias_tensor.shape;

            // 1. Verifica compatibilidade de broadcast:
            //    shapes são comparados a partir do fim (direita):
            size_t ndimA = shapeA.size();
            size_t ndimB = shapeB.size();
            for (size_t i = 0; i < ndimB; ++i) {
                size_t dimA = shapeA[ndimA - 1 - i];
                size_t dimB = shapeB[ndimB - 1 - i];
                if (dimB != 1 && dimB != dimA) {
                    throw std::runtime_error("Incompatible for broadcast: dimension " +
                        std::to_string(dimB) + " x " + std::to_string(dimA));
                }
            }

            // 2. Cria o resultado com a forma de self_tensor
            ZTensor result(shapeA);
            size_t total_size = result.size();
            if (total_size == 0) {
                // Se for tensor vazio, basta retornar result (vazio)
                zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
                return;
            }

            // 3. Pré-computa strides do self e do bias
            const std::vector<size_t> &stridesA = self_tensor.strides;
            const std::vector<size_t> &stridesB = bias_tensor.strides;

            // 4. Para cada elemento em result, calcula o índice correspondente em bias
            //    e copia o valor. O critério:
            //    - Se bias_shape[d] == 1 → sempre usa índice 0 nessa dimensão
            //    - Senão → usa o índice em self naquela dimensão
            const float *dataB = bias_tensor.data.data();
            float *dataR = result.data.data();

            std::vector<size_t> indexA(ndimA), indexB(ndimB);
            for (size_t lin = 0; lin < total_size; ++lin) {
                // 4.1. Reconstrói o índice multidimensional de "lin" em shapeA
                size_t rem = lin;
                for (size_t d = 0; d < ndimA; ++d) {
                    indexA[d] = rem / stridesA[d];
                    rem = rem % stridesA[d];
                }

                // 4.2. Mapeia em indexB (direita-alinhado)
                //      Se bias tem menos dims, as dimensões altas de indexB são consideradas "travadas" em zero
                size_t offsetB = 0;
                if (ndimB == 0) {
                    // bias é escalar: sempre offsetB = 0
                    offsetB = 0;
                } else {
                    // calcula deslocamento do índice multidimensional de bias
                    // alinhando a direita:
                    size_t diff = ndimA - ndimB;
                    for (size_t db = 0; db < ndimB; ++db) {
                        size_t dimB = shapeB[db];
                        size_t dimIndexA = indexA[diff + db];
                        size_t idxB = (dimB == 1 ? 0 : dimIndexA);
                        offsetB += idxB * stridesB[db];
                    }
                }

                // 4.3. Copia do bias
                dataR[lin] = dataB[offsetB];
            }

            // 5. Retorna um novo objeto PHP contendo 'result'
            zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
        } catch (const std::exception &e) {
            zend_throw_exception(zend_ce_exception, e.what(), 0);
            RETURN_THROWS();
        }
}

// --- PHP_METHOD para greater ---
PHP_METHOD(ZTensor, greater)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    // Converte $this
    zmatrix_ztensor_object *intern = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!intern->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }
    ZTensor &A = *intern->tensor;

    // Converte o argumento para ZTensor*
    ZTensor tmp_other, *B_ptr = nullptr;
    if (!zmatrix_get_tensor_ptr(other_zv, B_ptr, tmp_other, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }
    ZTensor &B = *B_ptr;

    // Shape-mismatch → RuntimeException
    if (!A.same_shape(B)) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_SHAPE_MISMATCH, 0);
        RETURN_THROWS();
    }

    try {
        size_t N = A.size();
        ZTensor result(A.shape);
        float *r_data = result.data.data();
        const float *a_data = A.data.data();
        const float *b_data = B.data.data();

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                r_data[i] = (a_data[i] > b_data[i]) ? 1.0f : 0.0f;
            }
        } else
        #endif
        {
            for (size_t i = 0; i < N; ++i) {
                r_data[i] = (a_data[i] > b_data[i]) ? 1.0f : 0.0f;
            }
        }

        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    }
    catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}
// --- Método estático minimum ---
PHP_METHOD(ZTensor, minimum)
{
    zval *a_zv;
    double b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(a_zv)
        Z_PARAM_DOUBLE(b)
    ZEND_PARSE_PARAMETERS_END();

    // Converte 'a' para ZTensor*
    ZTensor *A_ptr = nullptr;
    ZTensor tmpA;
    if (!zmatrix_get_tensor_ptr(a_zv, A_ptr, tmpA, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }

    try {
        ZTensor &A = *A_ptr;
        size_t N = A.size();
        ZTensor result(A.shape);
        float *res_data = result.data.data();
        const float *a_data = A.data.data();
        float scalar = static_cast<float>(b);

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                res_data[i] = (a_data[i] < scalar ? a_data[i] : scalar);
            }
        } else {
            for (size_t i = 0; i < N; ++i) {
                res_data[i] = (a_data[i] < scalar ? a_data[i] : scalar);
            }
        }
        #else
        for (size_t i = 0; i < N; ++i) {
            res_data[i] = (a_data[i] < scalar ? a_data[i] : scalar);
        }
        #endif

        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

// --- Método estático maximum ---
PHP_METHOD(ZTensor, maximum)
{
    zval *a_zv;
    double b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(a_zv)
        Z_PARAM_DOUBLE(b)
    ZEND_PARSE_PARAMETERS_END();

    // Converte 'a' para ZTensor*
    ZTensor *A_ptr = nullptr;
    ZTensor tmpA;
    if (!zmatrix_get_tensor_ptr(a_zv, A_ptr, tmpA, zmatrix_ce_ZTensor)) {
        RETURN_THROWS();
    }

    try {
        ZTensor &A = *A_ptr;
        size_t N = A.size();
        ZTensor result(A.shape);
        float *res_data = result.data.data();
        const float *a_data = A.data.data();
        float scalar = static_cast<float>(b);

        #if HAS_OPENMP
        if (N > ZMATRIX_PARALLEL_THRESHOLD) {
#pragma omp parallel for simd schedule(static)
            for (size_t i = 0; i < N; ++i) {
                res_data[i] = (a_data[i] > scalar ? a_data[i] : scalar);
            }
        } else {
            for (size_t i = 0; i < N; ++i) {
                res_data[i] = (a_data[i] > scalar ? a_data[i] : scalar);
            }
        }
        #else
        for (size_t i = 0; i < N; ++i) {
            res_data[i] = (a_data[i] > scalar ? a_data[i] : scalar);
        }
        #endif

        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, scalarDivide)
{
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    try {
        ZTensor &A = *self_obj->tensor;

        // 1) Se for escalar (int ou float), divide elemento a elemento
        if (Z_TYPE_P(other_zv) == IS_LONG || Z_TYPE_P(other_zv) == IS_DOUBLE) {
            float scalar = (Z_TYPE_P(other_zv) == IS_LONG)
                ? (float)Z_LVAL_P(other_zv)
                : (float)Z_DVAL_P(other_zv);
            A.scalar_divide(scalar);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 2) Caso tensor/array: converter para ZTensor*
        ZTensor *B_ptr = nullptr;
        ZTensor tmpB;
        if (!zmatrix_get_tensor_ptr(other_zv, B_ptr, tmpB, zmatrix_ce_ZTensor)) {
            RETURN_THROWS();
        }
        ZTensor &B = *B_ptr;

        // 3) Mesmos formatos → divisão direta
        if (A.shape == B.shape) {
            A.divide(B);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 4) Broadcast linha: A [M×N], B [N]
        const auto &shapeA = A.shape;
        const auto &shapeB = B.shape;
        if (shapeA.size() == 2 && shapeB.size() == 1 && shapeB[0] == shapeA[1]) {
            size_t M = shapeA[0], N = shapeA[1];
            ZTensor C(shapeA);
            float *cd = C.data.data();
            const float *bd = B.data.data();
            for (size_t i = 0; i < M; ++i) {
                memcpy(cd + i*N, bd, N * sizeof(float));
            }
            A.divide(C);
            ZVAL_ZVAL(return_value, ZEND_THIS, 1, 0);
            return;
        }

        // 5) Broadcast inverso não suportado
        if (shapeA.size() == 1 && shapeB.size() == 2 && shapeA[0] == shapeB[1]) {
            throw std::runtime_error("For reverse broadcasting, call B->scalarDivide(A)");
        }

        throw std::invalid_argument(ZMATRIX_ERR_SHAPE_MISMATCH);

    } catch (const std::exception &e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, copy)
{
    ZEND_PARSE_PARAMETERS_NONE();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, "Tensor not initialized", 0);
        RETURN_THROWS();
    }

    try {
        ZTensor& A = *self_obj->tensor;
        object_init_ex(return_value, Z_OBJCE_P(ZEND_THIS)); // cria novo objeto do mesmo tipo
        zmatrix_ztensor_object *new_obj = Z_MATRIX_ZTENSOR_P(return_value);
        new_obj->tensor = new ZTensor(A); // copia
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, safe)
{
    zval *input_zv;
    zend_bool copy = 1;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(input_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(copy)
    ZEND_PARSE_PARAMETERS_END();

    try {
        if (Z_TYPE_P(input_zv) == IS_ARRAY) {
            ZTensor tensor = php_array_to_tensor(input_zv);
            zmatrix_return_tensor_obj(tensor, return_value, zmatrix_ce_ZTensor);
            return;
        }

        if (Z_TYPE_P(input_zv) == IS_OBJECT && instanceof_function(Z_OBJCE_P(input_zv), zmatrix_ce_ZTensor)) {
            zmatrix_ztensor_object *obj = Z_MATRIX_ZTENSOR_P(input_zv);
            if (!obj->tensor) {
                zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
                RETURN_THROWS();
            }

            if (copy) {
                ZTensor tensorCopy = ZTensor(*obj->tensor); // deep copy
                zmatrix_return_tensor_obj(tensorCopy, return_value, zmatrix_ce_ZTensor);
            } else {
                ZVAL_ZVAL(return_value, input_zv, 1, 0); // shallow, reusa
            }
            return;
        }

        zend_throw_exception(zend_ce_exception, "ZTensor::safe() expects an array or ZTensor", 0);
        RETURN_THROWS();

    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, tile)
{
    zval *tensor_zv;
    zend_long times;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, zmatrix_ce_ZTensor)
        Z_PARAM_LONG(times)
    ZEND_PARSE_PARAMETERS_END();

    if (times <= 0) {
        zend_throw_exception(zend_ce_exception, "tile(): parameter times must be >= 1", 0);
        RETURN_THROWS();
    }

    zmatrix_ztensor_object *obj = Z_MATRIX_ZTENSOR_P(tensor_zv);
    if (!obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    try {
        const ZTensor &input = *obj->tensor;
        const auto &inShape = input.shape;

        if (inShape.empty()) {
            zend_throw_exception(zend_ce_exception, "tile(): scalar tensor cannot be repeated", 0);
            RETURN_THROWS();
        }

        size_t rows = inShape[0];
        size_t cols = (inShape.size() == 2) ? inShape[1] : 1;
        std::vector<size_t> outShape = {rows * (size_t)times};
        if (inShape.size() == 2) outShape.push_back(cols);

        ZTensor result(outShape);
        const float* src = input.data.data();
        float* dst = result.data.data();

        size_t blockSize = rows * cols;
        for (size_t i = 0; i < (size_t)times; ++i) {
            memcpy(dst + i * blockSize, src, blockSize * sizeof(float));
        }

        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ztensor___tostring, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ZTensor, __toString)
{
    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);

    if (!self_obj->tensor) {
        RETURN_STRING("[ZTensor: (not initialized)]");
        return;
    }

    try {
        std::string result = self_obj->tensor->to_string();
        RETURN_STRING(result.c_str());
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_STRING("[ZTensor: error]");
    }
}

// TODO: Implementar métodos estáticos rand() e randn() similarmente

// ========== Autograd Methods ==========

PHP_METHOD(ZTensor, requiresGrad)
{
    zend_bool requires_grad = 1; // Default: true
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(requires_grad)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    self_obj->tensor->requiresGrad(requires_grad);
    ZVAL_COPY(return_value, ZEND_THIS);
}

PHP_METHOD(ZTensor, isRequiresGrad)
{
    ZEND_PARSE_PARAMETERS_NONE();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    RETURN_BOOL(self_obj->tensor->isRequiresGrad());
}

PHP_METHOD(ZTensor, ensureGrad)
{
    ZEND_PARSE_PARAMETERS_NONE();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    self_obj->tensor->ensureGrad();
    RETURN_NULL();
}

PHP_METHOD(ZTensor, zeroGrad)
{
    ZEND_PARSE_PARAMETERS_NONE();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    self_obj->tensor->zeroGrad();
    RETURN_NULL();
}

PHP_METHOD(ZTensor, getGrad)
{
    ZEND_PARSE_PARAMETERS_NONE();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    auto grad = self_obj->tensor->getGrad();
    if (!grad) {
        RETURN_NULL();
    }

    zmatrix_return_tensor_obj(*grad, return_value, zmatrix_ce_ZTensor);
}

PHP_METHOD(ZTensor, backward)
{
    ZEND_PARSE_PARAMETERS_NONE();

    zmatrix_ztensor_object *self_obj = Z_MATRIX_ZTENSOR_P(ZEND_THIS);
    if (!self_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    try {
        self_obj->tensor->backward();
        RETURN_NULL();
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

// ==========================================================================
// Static Autograd Methods for ZTensor class
// ==========================================================================

PHP_METHOD(ZTensor, addAutograd)
{
    zval *a_zv, *b_zv;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT(a_zv)
        Z_PARAM_OBJECT(b_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *a_obj = Z_MATRIX_ZTENSOR_P(a_zv);
    zmatrix_ztensor_object *b_obj = Z_MATRIX_ZTENSOR_P(b_zv);

    if (!a_obj->tensor || !b_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    try {
        ZTensor result = ZTensor::add_autograd(*a_obj->tensor, *b_obj->tensor);
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, subAutograd)
{
    zval *a_zv, *b_zv;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT(a_zv)
        Z_PARAM_OBJECT(b_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *a_obj = Z_MATRIX_ZTENSOR_P(a_zv);
    zmatrix_ztensor_object *b_obj = Z_MATRIX_ZTENSOR_P(b_zv);

    if (!a_obj->tensor || !b_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    try {
        ZTensor result = ZTensor::sub_autograd(*a_obj->tensor, *b_obj->tensor);
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, mulAutograd)
{
    zval *a_zv, *b_zv;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT(a_zv)
        Z_PARAM_OBJECT(b_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *a_obj = Z_MATRIX_ZTENSOR_P(a_zv);
    zmatrix_ztensor_object *b_obj = Z_MATRIX_ZTENSOR_P(b_zv);

    if (!a_obj->tensor || !b_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    try {
        ZTensor result = ZTensor::mul_autograd(*a_obj->tensor, *b_obj->tensor);
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

PHP_METHOD(ZTensor, sumAutograd)
{
    zval *a_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a_zv)
    ZEND_PARSE_PARAMETERS_END();

    zmatrix_ztensor_object *a_obj = Z_MATRIX_ZTENSOR_P(a_zv);

    if (!a_obj->tensor) {
        zend_throw_exception(zend_ce_exception, ZMATRIX_ERR_NOT_INITIALIZED, 0);
        RETURN_THROWS();
    }

    try {
        ZTensor result = ZTensor::sum_autograd(*a_obj->tensor);
        zmatrix_return_tensor_obj(result, return_value, zmatrix_ce_ZTensor);
    } catch (const std::exception& e) {
        zend_throw_exception(zend_ce_exception, e.what(), 0);
        RETURN_THROWS();
    }
}

#endif /* ZMATRIX_METHODS_H */