#ifndef ZMATRIX_ARGINFO_H
#define ZMATRIX_ARGINFO_H
// --- Construtor ---
 ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_construct, 0, 0, 0)
     ZEND_ARG_INFO(0, dataOrShape)
 ZEND_END_ARG_INFO()

 // --- ARG_INFO para o método estático arr ---
 ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_arr, 0, 0, 1)
     ZEND_ARG_INFO(0, input_data) // Agora é um zval genérico, faremos a verificação de tipo manualmente
 ZEND_END_ARG_INFO()

// --- Métodos de Operação Binária (add, sub, mul - elementwise) ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_op_other, 0, 0, 1)
    ZEND_ARG_INFO(0, other) // Aceita ZTensor ou array
ZEND_END_ARG_INFO()

// --- Métodos de Operação Escalar ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_op_scalar, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, scalar, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()

// --- Métodos Unários (transpose, abs, sigmoid) ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_no_args, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ztensor_isOnGpu, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

// --- Autograd Methods ---
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ZMatrix_ZTensor_requiresGrad, 0, 0, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, requires_grad, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ZMatrix_ZTensor_isRequiresGrad, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ZMatrix_ZTensor_ensureGrad, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ZMatrix_ZTensor_zeroGrad, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_getGrad, 0, 0, ZMatrix\\ZTensor, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ZMatrix_ZTensor_backward, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

// --- Global Autograd Functions ---
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_add_autograd, 0, 2, "ZMatrix\\ZTensor", 0)
    ZEND_ARG_TYPE_INFO(0, a, IS_OBJECT, 0)
    ZEND_ARG_TYPE_INFO(0, b, IS_OBJECT, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_sub_autograd, 0, 2, "ZMatrix\\ZTensor", 0)
    ZEND_ARG_TYPE_INFO(0, a, IS_OBJECT, 0)
    ZEND_ARG_TYPE_INFO(0, b, IS_OBJECT, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_mul_autograd, 0, 2, "ZMatrix\\ZTensor", 0)
    ZEND_ARG_TYPE_INFO(0, a, IS_OBJECT, 0)
    ZEND_ARG_TYPE_INFO(0, b, IS_OBJECT, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_sum_autograd, 0, 1, "ZMatrix\\ZTensor", 0)
    ZEND_ARG_TYPE_INFO(0, tensor, IS_OBJECT, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_optional_float, 0, 0, 0)
    ZEND_ARG_TYPE_INFO(0, alpha, IS_DOUBLE, 1)
ZEND_END_ARG_INFO()


// --- Métodos Estáticos (Criação) ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_shape, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_shape_value, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
    ZEND_ARG_TYPE_INFO(0, value, IS_DOUBLE, 1) // value agora é opcional
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_identity, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, size, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_random, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
    ZEND_ARG_TYPE_INFO(0, min, IS_DOUBLE, 1)
    ZEND_ARG_TYPE_INFO(0, max, IS_DOUBLE, 1)
ZEND_END_ARG_INFO()

// === Divide ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_divide, 0, 0, 1)
    ZEND_ARG_INFO(0, other)
ZEND_END_ARG_INFO()

// === Pow ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_pow, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, exponent, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()

// === ReLU ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_relu, 0, 0, 0)
ZEND_END_ARG_INFO()

// === Tanh ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_tanh, 0, 0, 0)
ZEND_END_ARG_INFO()

// === Exp ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_exp, 0, 0, 0)
ZEND_END_ARG_INFO()

// === Log ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_log, 0, 0, 0)
ZEND_END_ARG_INFO()

// === Sqrt ===
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_sqrt, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_softmax, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_reshape, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
ZEND_END_ARG_INFO()

// --- ARG_INFO para métodos estáticos de criação adicionais ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_randn, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, mean, IS_DOUBLE, 0, "0.0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, std_dev, IS_DOUBLE, 0, "1.0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_arange, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, arg1, IS_DOUBLE, 0) // start_or_stop
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, arg2, IS_DOUBLE, 1, "null") // stop (pode ser null se não passado explicitamente)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, arg3, IS_DOUBLE, 0, "1.0") // step
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_linspace, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, start, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, stop, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, num, IS_LONG, 0, "50")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, endpoint, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_logspace, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, start, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, stop, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, num, IS_LONG, 0, "50")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, endpoint, _IS_BOOL, 0, "true")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, base, IS_DOUBLE, 0, "10.0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_eye, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, N, IS_LONG, 0)
    // Para M, como é opcional e pode ser nulo para indicar M=N.
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, M, IS_LONG, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, k, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_dot, 0, 0, 1)
 ZEND_ARG_INFO(0, other)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_key, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, indices, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_ones, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, shape, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_clip_range, 0, 0, 3)
    ZEND_ARG_INFO(0, input)
    ZEND_ARG_TYPE_INFO(0, min, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, max, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_sum_flex, 0, 0, 1)
    ZEND_ARG_INFO(0, other) // Ztensor ou array
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_broadcast, 0, 0, 1)
    ZEND_ARG_OBJ_INFO(0, bias, ZMatrix\\Ztensor, 0)
ZEND_END_ARG_INFO()

// --- ARG_INFO para greater (element-wise “>”) ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_greater, 0, 0, 1)
    ZEND_ARG_INFO(0, other) // ZTensor ou array
ZEND_END_ARG_INFO()

// --- ARG_INFO para minimum e maximum ---
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_minimum, 0, 0, 2)
    ZEND_ARG_INFO(0, a) // ZTensor ou array
    ZEND_ARG_INFO(0, b) // float
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_maximum, 0, 0, 2)
    ZEND_ARG_INFO(0, a) // ZTensor ou array
    ZEND_ARG_INFO(0, b) // float
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_scalarDivide, 0, 0, 1)
   ZEND_ARG_INFO(0, scalar) // pode ser float|int|ZTensor|array
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_copy, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_safe, 0, 0, 1)
    ZEND_ARG_INFO(0, input)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, copy, _IS_BOOL, 1, "true")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_static_tile, 0, 0, 2)
    ZEND_ARG_OBJ_INFO(0, tensor, ZMatrix\\ZTensor, 0)
    ZEND_ARG_TYPE_INFO(0, times, IS_LONG, 0)
ZEND_END_ARG_INFO()

// --- ARG_INFO para matmul ---
// Aceita 1 argumento: outro ZTensor ou array
ZEND_BEGIN_ARG_INFO_EX(arginfo_ztensor_matmul, 0, 0, 1)
    ZEND_ARG_INFO(0, other) // Aceita ZTensor ou array
    // ZEND_ARG_TYPE_INFO(0, use_blas, _IS_BOOL, 1) // Opcional para usar BLAS (não implementado aqui)
ZEND_END_ARG_INFO()

#endif /* ZMATRIX_ARGINFO_H */