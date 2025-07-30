AC_DEFUN([AX_DETECT_IMATH], [
AC_DEFINE([USE_IMATH_FOR_MP], [], [use imath to implement isl_int])

MP_CPPFLAGS="-I$srcdir/imath_wrap"
MP_LDFLAGS=""
MP_LIBS=""

SAVE_CPPFLAGS="$CPPFLAGS"
CPPFLAGS="$MP_CPPFLAGS $CPPFLAGS"
AC_CHECK_HEADER([imath.h], [], [AC_ERROR([imath.h header not found])])
AC_CHECK_HEADER([gmp_compat.h], [], [AC_ERROR([gmp_compat.h header not found])])
CPPFLAGS="$SAVE_CPPFLAGS"

AM_CONDITIONAL(NEED_GET_MEMORY_FUNCTIONS, test x = xfalse)
])
