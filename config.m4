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

  dnl Flags padrão de C++ com otimizações
  ZMATRIX_CXXFLAGS="$ZMATRIX_CXXFLAGS -Wall -std=c++17 -O3"

  dnl Flags de arquitetura (AVX etc.)
  ZMATRIX_ARCH_FLAGS="-march=native"
  ZMATRIX_CXXFLAGS="$ZMATRIX_CXXFLAGS $ZMATRIX_ARCH_FLAGS"
  AC_MSG_CHECKING([for AVX/SIMD flags])
  AC_MSG_RESULT([using $ZMATRIX_ARCH_FLAGS])

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

  dnl Registra a extensão, não usando LDFLAGS diretamente
  PHP_NEW_EXTENSION(zmatrix, src/zmatrix.cpp, $ext_shared, , $ZMATRIX_CXXFLAGS)

  AC_LANG_POP([C++])

  AC_MSG_NOTICE([Final ZMatrix CXXFLAGS: $ZMATRIX_CXXFLAGS])
  AC_MSG_NOTICE([Final ZMatrix SHARED_LIBADD: $ZMATRIX_SHARED_LIBADD])
fi
