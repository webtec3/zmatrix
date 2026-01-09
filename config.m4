PHP_ARG_ENABLE(zmatrix, whether to enable ZMatrix support,
[  --enable-zmatrix   Enable ZMatrix support], no)

if test "$PHP_ZMATRIX" != "no"; then
  AC_LANG_PUSH([C++])
  AC_PROG_CXX([g++ clang++ c++])
  PHP_REQUIRE_CXX()
  AC_PROG_CXXCPP

  dnl Limpa variáveis
  ZMATRIX_CPPFLAGS=""
  ZMATRIX_CXXFLAGS=""
  ZMATRIX_SHARED_LIBADD=""
  ZMATRIX_NVCCFLAGS=""

  dnl Flags padrão de C++ com otimizações
  ZMATRIX_CXXFLAGS="$ZMATRIX_CXXFLAGS -Wall -std=c++17 -O3"

  dnl Flags de arquitetura (AVX etc.)
  ZMATRIX_ARCH_FLAGS="-march=native"
  ZMATRIX_CXXFLAGS="$ZMATRIX_CXXFLAGS $ZMATRIX_ARCH_FLAGS"
  AC_MSG_CHECKING([for AVX/SIMD flags])
  AC_MSG_RESULT([using $ZMATRIX_ARCH_FLAGS])

  dnl Detectar nvcc CUDA compiler
  AC_MSG_CHECKING([for nvcc CUDA compiler])
  AC_PATH_PROG([NVCC], [nvcc], [no])
  if test "$NVCC" != "no"; then
    AC_DEFINE([HAVE_CUDA], [1], [Define if CUDA toolkit is available])
    AC_MSG_RESULT([yes, $NVCC])
    HAVE_CUDA=1

    dnl Verificar diretórios CUDA comuns
    CUDA_DIRS="/usr/local/cuda /opt/cuda /usr/cuda"
    CUDA_FOUND=0
    for cuda_dir in $CUDA_DIRS; do
      if test -d "$cuda_dir/include" && test -d "$cuda_dir/lib64"; then
        CUDA_ROOT="$cuda_dir"
        CUDA_FOUND=1
        break
      fi
    done

    if test "$CUDA_FOUND" = "0"; then
      AC_MSG_WARN([CUDA directories not found in standard locations])
      CUDA_ROOT="/usr/local/cuda"
    fi

    AC_MSG_NOTICE([Using CUDA root directory: $CUDA_ROOT])

    dnl Verificar headers CUDA
    AC_MSG_CHECKING([for CUDA headers])
    AC_MSG_RESULT([searching...])
    if test -f "$CUDA_ROOT/include/cuda_runtime.h"; then
      AC_MSG_RESULT([yes in $CUDA_ROOT/include])
      ZMATRIX_CPPFLAGS="$ZMATRIX_CPPFLAGS -I$CUDA_ROOT/include"
    elif test -f "/usr/include/cuda_runtime.h"; then
      AC_MSG_RESULT([yes in /usr/include])
      ZMATRIX_CPPFLAGS="$ZMATRIX_CPPFLAGS -I/usr/include"
    else
      AC_MSG_ERROR([CUDA headers not found (tried $CUDA_ROOT/include and /usr/include)])
    fi

    dnl Verificar bibliotecas CUDA
    AC_MSG_CHECKING([for CUDA libraries])
    if test -f "$CUDA_ROOT/lib64/libcudart.so" && test -f "$CUDA_ROOT/lib64/libcublas.so"; then
      AC_MSG_RESULT([yes in $CUDA_ROOT/lib64])
      ZMATRIX_SHARED_LIBADD="$ZMATRIX_SHARED_LIBADD -L$CUDA_ROOT/lib64 -lcudart -lcublas -lcurand"
    elif test -f "/usr/lib/x86_64-linux-gnu/libcudart.so" && test -f "/usr/lib/x86_64-linux-gnu/libcublas.so"; then
      AC_MSG_RESULT([yes in /usr/lib/x86_64-linux-gnu])
      ZMATRIX_SHARED_LIBADD="$ZMATRIX_SHARED_LIBADD -L/usr/lib/x86_64-linux-gnu -lcudart -lcublas -lcurand"
    else
      AC_MSG_ERROR([CUDA libraries not found (tried $CUDA_ROOT/lib64 and /usr/lib/x86_64-linux-gnu)])
    fi

    dnl Flags de compilação CUDA
    ZMATRIX_CXXFLAGS="$ZMATRIX_CXXFLAGS -DHAVE_CUDA"
    ZMATRIX_NVCCFLAGS="-std=c++17 -O3 -Xcompiler -fPIC -I$CUDA_ROOT/include"

    dnl Detectar compute capability
    AC_MSG_CHECKING([for CUDA compute capability])
    if test -x "$CUDA_ROOT/extras/demo_suite/deviceQuery"; then
      COMPUTE_CAP=`$CUDA_ROOT/extras/demo_suite/deviceQuery | grep "CUDA Capability" | head -1 | sed \'s/.*CUDA Capability Major\\/Minor version number: *\\([[0-9]]\\+\\)\\.\\([[0-9]]\\+\\).*/\\1\\2/\'`
      if test -n "$COMPUTE_CAP"; then
        ZMATRIX_NVCCFLAGS="$ZMATRIX_NVCCFLAGS -arch=sm_$COMPUTE_CAP"
        AC_MSG_RESULT([sm_$COMPUTE_CAP])
      else
        ZMATRIX_NVCCFLAGS="$ZMATRIX_NVCCFLAGS -arch=sm_60"
        AC_MSG_RESULT([defaulting to sm_60])
      fi
    else
      ZMATRIX_NVCCFLAGS="$ZMATRIX_NVCCFLAGS -arch=sm_60"
      AC_MSG_RESULT([defaulting to sm_60])
    fi

    dnl Verificar se o arquivo .cu existe
    AC_CHECK_FILE([${srcdir}/src/gpu_kernels.cu], [], [
      AC_MSG_ERROR([Missing CUDA kernels ${srcdir}/src/gpu_kernels.cu])
    ])

  else
    AC_MSG_RESULT([no])
    HAVE_CUDA=0
    AC_MSG_NOTICE([CUDA support disabled - nvcc not found])
  fi

  dnl Verifica OpenBLAS
  AC_MSG_CHECKING([for OpenBLAS library])
  AC_SEARCH_LIBS([cblas_dgemm], [openblas], [
    PHP_ADD_LIBRARY(openblas, 1, ZMATRIX_SHARED_LIBADD)
    AC_DEFINE(HAVE_OPENBLAS, 1, [Define if OpenBLAS is available])
    AC_MSG_RESULT([yes])
  ], [
    AC_MSG_RESULT([no])
    AC_MSG_CHECKING([for generic CBLAS library])
    AC_SEARCH_LIBS([cblas_dgemm], [cblas], [
      PHP_ADD_LIBRARY(cblas, 1, ZMATRIX_SHARED_LIBADD)
      AC_DEFINE(HAVE_CBLAS, 1, [Define if generic CBLAS is available])
      AC_MSG_RESULT([yes])
    ], [
      AC_MSG_RESULT([no])
      AC_MSG_ERROR([BLAS library with cblas_dgemm not found (install libopenblas-dev or libblas-dev).])
    ])
  ])

  dnl Verifica OpenMP
  AC_MSG_CHECKING([for OpenMP support])
  AC_OPENMP
  if test "$ac_cv_prog_cxx_openmp" != "no"; then
    ZMATRIX_CXXFLAGS="$ZMATRIX_CXXFLAGS $OPENMP_CXXFLAGS"
    PHP_ADD_LIBRARY(gomp, 1, ZMATRIX_SHARED_LIBADD)
    AC_DEFINE(HAVE_OPENMP, 1, [Define if OpenMP is available])
    AC_MSG_RESULT([yes ($OPENMP_CXXFLAGS)])
  else
    AC_MSG_RESULT([no])
    AC_MSG_WARN([OpenMP support not found. Extension will run single-threaded.])
  fi

  dnl Aplica as flags corretamente
  PHP_SUBST(ZMATRIX_CPPFLAGS)
  PHP_SUBST(ZMATRIX_CXXFLAGS)
  PHP_SUBST(ZMATRIX_SHARED_LIBADD)
  PHP_SUBST(ZMATRIX_NVCCFLAGS)
  PHP_SUBST(NVCC)
  PHP_SUBST(HAVE_CUDA)

  dnl Registra a extensão
    if test "$HAVE_CUDA" = "1"; then
      PHP_NEW_EXTENSION(zmatrix, [src/zmatrix.cpp], $ext_shared, , $ZMATRIX_CXXFLAGS)
      PHP_ADD_MAKEFILE_FRAGMENT
      AC_MSG_NOTICE([ZMatrix extension will be built with CUDA support])
    else
      PHP_NEW_EXTENSION(zmatrix, [src/zmatrix.cpp], $ext_shared, , $ZMATRIX_CXXFLAGS)
      AC_MSG_NOTICE([ZMatrix extension will be built without CUDA support])
    fi

  AC_LANG_POP([C++])

  AC_MSG_NOTICE([Final ZMatrix CXXFLAGS: $ZMATRIX_CXXFLAGS])
  AC_MSG_NOTICE([Final ZMatrix SHARED_LIBADD: $ZMATRIX_SHARED_LIBADD])
  if test "$HAVE_CUDA" = "1"; then
    AC_MSG_NOTICE([Final ZMatrix NVCCFLAGS: $ZMATRIX_NVCCFLAGS])
    AC_MSG_NOTICE([NVCC path: $NVCC])
  fi
fi