/* This is a generated file, edit the .stub.php file instead.
 * Stub hash: d8c41d3ec021903124dea99e31abfc3e7c370a04 */

ZEND_BEGIN_ARG_INFO_EX(arginfo_class_ZMatrix_ZTensor___construct, 0, 0, 0)
	ZEND_ARG_OBJ_TYPE_MASK(0, dataOrShape, ZMatrix\\ZTensor, MAY_BE_ARRAY|MAY_BE_NULL, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_safe, 0, 1, ZMatrix\\ZTensor, 0)
	ZEND_ARG_OBJ_TYPE_MASK(0, arrayData, ZMatrix\\ZTensor, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_full, 0, 2, ZMatrix\\ZTensor, 0)
	ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
	ZEND_ARG_TYPE_MASK(0, value, MAY_BE_LONG|MAY_BE_DOUBLE, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_zeros, 0, 1, ZMatrix\\ZTensor, 0)
	ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_ones arginfo_class_ZMatrix_ZTensor_zeros

#define arginfo_class_ZMatrix_ZTensor_empty arginfo_class_ZMatrix_ZTensor_zeros

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_arange, 0, 1, ZMatrix\\ZTensor, 0)
	ZEND_ARG_TYPE_INFO(0, start_or_stop, IS_DOUBLE, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, stop, IS_DOUBLE, 1, "null")
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, step, IS_DOUBLE, 0, "1.0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_linspace, 0, 2, ZMatrix\\ZTensor, 0)
	ZEND_ARG_TYPE_INFO(0, start, IS_DOUBLE, 0)
	ZEND_ARG_TYPE_INFO(0, stop, IS_DOUBLE, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, num, IS_LONG, 0, "50")
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, endpoint, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_rand, 0, 1, ZMatrix\\ZTensor, 0)
	ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, min, IS_DOUBLE, 0, "0.0")
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, max, IS_DOUBLE, 0, "1.0")
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_randn arginfo_class_ZMatrix_ZTensor_zeros

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ZMatrix_ZTensor_toArray, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_getShape arginfo_class_ZMatrix_ZTensor_toArray

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ZMatrix_ZTensor_getSize, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_getNDim arginfo_class_ZMatrix_ZTensor_getSize

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ZMatrix_ZTensor_getDType, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_reshape arginfo_class_ZMatrix_ZTensor_zeros

#define arginfo_class_ZMatrix_ZTensor_reshape_inplace arginfo_class_ZMatrix_ZTensor_zeros

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_flatten, 0, 0, ZMatrix\\ZTensor, 0)
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_flatten_inplace arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_transpose arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_transpose_inplace arginfo_class_ZMatrix_ZTensor_flatten

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_add, 0, 1, ZMatrix\\ZTensor, 0)
	ZEND_ARG_OBJ_TYPE_MASK(0, other, ZMatrix\\ZTensor, MAY_BE_ARRAY, NULL)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, inplace, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_subtract arginfo_class_ZMatrix_ZTensor_add

#define arginfo_class_ZMatrix_ZTensor_multiply arginfo_class_ZMatrix_ZTensor_add

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_scalarAdd, 0, 1, ZMatrix\\ZTensor, 0)
	ZEND_ARG_TYPE_MASK(0, scalar, MAY_BE_LONG|MAY_BE_DOUBLE, NULL)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, inplace, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_scalarSubtract arginfo_class_ZMatrix_ZTensor_scalarAdd

#define arginfo_class_ZMatrix_ZTensor_scalarMultiply arginfo_class_ZMatrix_ZTensor_scalarAdd

#define arginfo_class_ZMatrix_ZTensor_scalarDivide arginfo_class_ZMatrix_ZTensor_scalarAdd

#define arginfo_class_ZMatrix_ZTensor_divide arginfo_class_ZMatrix_ZTensor_add

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_matmul, 0, 1, ZMatrix\\ZTensor, 0)
	ZEND_ARG_OBJ_TYPE_MASK(0, other, ZMatrix\\ZTensor, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_class_ZMatrix_ZTensor_sumtotal, 0, 0, MAY_BE_LONG|MAY_BE_DOUBLE)
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_mean arginfo_class_ZMatrix_ZTensor_sumtotal

#define arginfo_class_ZMatrix_ZTensor_std arginfo_class_ZMatrix_ZTensor_sumtotal

#define arginfo_class_ZMatrix_ZTensor_var arginfo_class_ZMatrix_ZTensor_sumtotal

#define arginfo_class_ZMatrix_ZTensor_min arginfo_class_ZMatrix_ZTensor_sumtotal

#define arginfo_class_ZMatrix_ZTensor_max arginfo_class_ZMatrix_ZTensor_sumtotal

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_sum, 0, 0, ZMatrix\\ZTensor, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_prod arginfo_class_ZMatrix_ZTensor_sum

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_cumsum, 0, 0, ZMatrix\\ZTensor, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_cumprod arginfo_class_ZMatrix_ZTensor_cumsum

#define arginfo_class_ZMatrix_ZTensor_argmin arginfo_class_ZMatrix_ZTensor_cumsum

#define arginfo_class_ZMatrix_ZTensor_argmax arginfo_class_ZMatrix_ZTensor_cumsum

#define arginfo_class_ZMatrix_ZTensor_exp arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_log arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_log10 arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_sqrt arginfo_class_ZMatrix_ZTensor_flatten

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_pow, 0, 1, ZMatrix\\ZTensor, 0)
	ZEND_ARG_TYPE_MASK(0, exponent, MAY_BE_LONG|MAY_BE_DOUBLE, NULL)
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_abs arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_sign arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_sin arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_cos arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_tan arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_sinh arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_cosh arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_tanh arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_relu arginfo_class_ZMatrix_ZTensor_flatten

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_leaky_relu, 0, 0, ZMatrix\\ZTensor, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, alpha, IS_DOUBLE, 0, "0.01")
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_sigmoid arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_tanh_activation arginfo_class_ZMatrix_ZTensor_flatten

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_softmax, 0, 0, ZMatrix\\ZTensor, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 0, "-1")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_clip, 0, 2, ZMatrix\\ZTensor, 0)
	ZEND_ARG_TYPE_MASK(0, min, MAY_BE_LONG|MAY_BE_DOUBLE, NULL)
	ZEND_ARG_TYPE_MASK(0, max, MAY_BE_LONG|MAY_BE_DOUBLE, NULL)
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_unique arginfo_class_ZMatrix_ZTensor_flatten

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_sort, 0, 0, ZMatrix\\ZTensor, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 0, "-1")
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, descending, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_concatenate, 0, 1, ZMatrix\\ZTensor, 0)
	ZEND_ARG_TYPE_INFO(0, tensors, IS_ARRAY, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_copy arginfo_class_ZMatrix_ZTensor_flatten

#define arginfo_class_ZMatrix_ZTensor_clone arginfo_class_ZMatrix_ZTensor_flatten

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ZMatrix_ZTensor_requiresGrad, 0, 0, IS_VOID, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, requires_grad, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ZMatrix_ZTensor_is_requires_grad, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ZMatrix_ZTensor_ensure_grad, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_zero_grad arginfo_class_ZMatrix_ZTensor_ensure_grad

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ZMatrix_ZTensor_get_grad, 0, 0, ZMatrix\\ZTensor, 1)
ZEND_END_ARG_INFO()

#define arginfo_class_ZMatrix_ZTensor_backward arginfo_class_ZMatrix_ZTensor_ensure_grad

