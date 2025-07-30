# size_max.m4 serial 6
dnl Copyright (C) 2003, 2005-2006, 2010, 2012 Free Software Foundation, Inc.
dnl This file is free software; the Free Software Foundation
dnl gives unlimited permission to copy and/or distribute it,
dnl with or without modifications, as long as this notice is preserved.

dnl From Bruno Haible.
dnl Change by Vincent Lefevre: added <inttypes.h> for consistency with MPFR
dnl Update with autoupdate.

AC_DEFUN([gl_SIZE_MAX],
[
  AC_CHECK_HEADERS(stdint.h)
  dnl First test whether the system already has SIZE_MAX.
  AC_MSG_CHECKING([for SIZE_MAX])
  AC_CACHE_VAL([gl_cv_size_max], [
    gl_cv_size_max=
    AC_EGREP_CPP([Found it], [
#include <limits.h>
#ifdef HAVE_INTTYPES_H
# include <inttypes.h>
#endif
#if HAVE_STDINT_H
# include <stdint.h>
#endif
#ifdef SIZE_MAX
Found it
#endif
], gl_cv_size_max=yes)
    if test -z "$gl_cv_size_max"; then
      dnl Define it ourselves. Here we assume that the type 'size_t' is not wider
      dnl than the type 'unsigned long'. Try hard to find a definition that can
      dnl be used in a preprocessor #if, i.e. doesn't contain a cast.
      AC_COMPUTE_INT([size_t_bits_minus_1], [sizeof (size_t) * CHAR_BIT - 1],
        [#include <stddef.h>
#include <limits.h>], size_t_bits_minus_1=)
      AC_COMPUTE_INT([fits_in_uint], [sizeof (size_t) <= sizeof (unsigned int)],
        [#include <stddef.h>], fits_in_uint=)
      if test -n "$size_t_bits_minus_1" && test -n "$fits_in_uint"; then
        if test $fits_in_uint = 1; then
          dnl Even though SIZE_MAX fits in an unsigned int, it must be of type
          dnl 'unsigned long' if the type 'size_t' is the same as 'unsigned long'.
          AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <stddef.h>
            extern size_t foo;
            extern unsigned long foo;
            ]], [[]])], fits_in_uint=0)
        fi
        dnl We cannot use 'expr' to simplify this expression, because 'expr'
        dnl works only with 'long' integers in the host environment, while we
        dnl might be cross-compiling from a 32-bit platform to a 64-bit platform.
        if test $fits_in_uint = 1; then
          gl_cv_size_max="(((1U << $size_t_bits_minus_1) - 1) * 2 + 1)"
        else
          gl_cv_size_max="(((1UL << $size_t_bits_minus_1) - 1) * 2 + 1)"
        fi
      else
        dnl Shouldn't happen, but who knows...
        gl_cv_size_max='((size_t)~(size_t)0)'
      fi
    fi
  ])
  AC_MSG_RESULT([$gl_cv_size_max])
  if test "$gl_cv_size_max" != yes; then
    AC_DEFINE_UNQUOTED([SIZE_MAX], [$gl_cv_size_max],
      [Define as the maximum value of type 'size_t', if the system doesn't define it.])
  fi
])

dnl Autoconf >= 2.61 has AC_COMPUTE_INT built-in.
dnl Remove this when we can assume autoconf >= 2.61.
m4_ifdef([AC_COMPUTE_INT], [], [
  AC_DEFUN([AC_COMPUTE_INT], [_AC_COMPUTE_INT([$2],[$1],[$3],[$4])])
])
