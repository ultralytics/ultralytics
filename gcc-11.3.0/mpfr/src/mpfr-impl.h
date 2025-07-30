/* Utilities for MPFR developers, not exported.

Copyright 1999-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#ifndef __MPFR_IMPL_H__
#define __MPFR_IMPL_H__

/* Let's include some standard headers unconditionally as they are
   already needed by several source files or when some options are
   enabled/disabled, and it is easy to forget them (some configure
   options may hide the error).
   Note: If some source file must not have such a header included
   (which is very unlikely and probably means something broken in
   this source file), we should do that with some macro (that would
   also force to disable incompatible features). */
#if defined (__cplusplus)
#include <cstdio>
#include <cstring>
#else
#include <stdio.h>
#include <string.h>
#endif
#include <limits.h>

#if _MPFR_EXP_FORMAT == 4
/* mpfr_exp_t will be defined as intmax_t */
# include "mpfr-intmax.h"
#endif

/* Check if we are inside a build of MPFR or inside the test suite.
   This is needed in mpfr.h to export or import the functions.
   It matters only for Windows DLL */
#ifndef __MPFR_TEST_H__
# define __MPFR_WITHIN_MPFR 1
#endif

/******************************************************
 ****************** Include files *********************
 ******************************************************/

/* Include 'config.h' before using ANY configure macros if needed
   NOTE: It isn't MPFR 'config.h', but GMP's one! */
#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

/* For the definition of MPFR_THREAD_ATTR. GCC/ICC detection macros are
   no longer used, as they sometimes gave incorrect information about
   the support of thread-local variables. A configure check is now done. */
#include "mpfr-thread.h"

#ifdef  MPFR_HAVE_GMP_IMPL /* Build with gmp internals */

# ifndef __GMP_H__
#  include "gmp.h"
# endif
# ifndef __GMP_IMPL_H__
#  include "gmp-impl.h"
# endif
# ifdef MPFR_NEED_LONGLONG_H
#  include "longlong.h"
# endif
# ifndef __MPFR_H
#  include "mpfr.h"
# endif

#else /* Build without gmp internals */

# ifndef __GMP_H__
#  include "gmp.h"
# endif
# ifndef __MPFR_H
#  include "mpfr.h"
# endif
# ifndef __GMPFR_GMP_H__
#  include "mpfr-gmp.h"
# endif
# ifdef MPFR_NEED_LONGLONG_H
#  define LONGLONG_STANDALONE
#  include "mpfr-longlong.h"
# endif

#endif
#undef MPFR_NEED_LONGLONG_H

/* If a mpn_sqr_n macro is not defined, use mpn_mul. GMP 4.x defines a
   mpn_sqr_n macro in gmp-impl.h (and this macro disappeared in GMP 5),
   so that GMP's macro can only be used when MPFR has been configured
   with --with-gmp-build (and only with GMP 4.x). */
#ifndef mpn_sqr_n
# define mpn_sqr_n(dst,src,n) mpn_mul((dst),(src),(n),(src),(n))
#endif


/******************************************************
 ***************** Detection macros *******************
 ******************************************************/

/* Macros to detect STDC, GCC, GLIBC, GMP and ICC version */
#if defined(__STDC_VERSION__)
# define __MPFR_STDC(version) (__STDC_VERSION__>=(version))
#elif defined(__STDC__)
# define __MPFR_STDC(version) (0 == (version))
#else
# define __MPFR_STDC(version) 0
#endif

#if defined(_WIN32)
/* Under MS Windows (e.g. with VS2008 or VS2010), Intel's compiler doesn't
   support/enable extensions like the ones seen under GNU/Linux.
   https://sympa.inria.fr/sympa/arc/mpfr/2011-02/msg00032.html */
# define __MPFR_ICC(a,b,c) 0
#elif defined(__ICC)
# define __MPFR_ICC(a,b,c) (__ICC >= (a)*100+(b)*10+(c))
#elif defined(__INTEL_COMPILER)
# define __MPFR_ICC(a,b,c) (__INTEL_COMPILER >= (a)*100+(b)*10+(c))
#else
# define __MPFR_ICC(a,b,c) 0
#endif

#if defined(__GNUC__) && defined(__GNUC_MINOR__) && ! __MPFR_ICC(0,0,0)
# define __MPFR_GNUC(a,i) \
 (MPFR_VERSION_NUM(__GNUC__,__GNUC_MINOR__,0) >= MPFR_VERSION_NUM(a,i,0))
#else
# define __MPFR_GNUC(a,i) 0
#endif

#if defined(__GLIBC__) && defined(__GLIBC_MINOR__)
# define __MPFR_GLIBC(a,i) \
 (MPFR_VERSION_NUM(__GLIBC__,__GLIBC_MINOR__,0) >= MPFR_VERSION_NUM(a,i,0))
#else
# define __MPFR_GLIBC(a,i) 0
#endif

#if defined(__GNU_MP_VERSION) && \
    defined(__GNU_MP_VERSION_MINOR) && \
    defined(__GNU_MP_VERSION_PATCHLEVEL)
# define __MPFR_GMP(a,b,c) \
  (MPFR_VERSION_NUM(__GNU_MP_VERSION,__GNU_MP_VERSION_MINOR,__GNU_MP_VERSION_PATCHLEVEL) >= MPFR_VERSION_NUM(a,b,c))
#else
# define __MPFR_GMP(a,b,c) 0
#endif



/******************************************************
 ************* GMP Basic Pointer Types ****************
 ******************************************************/

typedef mp_limb_t *mpfr_limb_ptr;
typedef __gmp_const mp_limb_t *mpfr_limb_srcptr;



/******************************************************
 ****************** (U)INTMAX_MAX *********************
 ******************************************************/

/* Let's try to fix UINTMAX_MAX and INTMAX_MAX if these macros don't work
   (e.g. with gcc -ansi -pedantic-errors in 32-bit mode under GNU/Linux),
   see <http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=582698>. */
#ifdef _MPFR_H_HAVE_INTMAX_T
# ifdef MPFR_HAVE_INTMAX_MAX
#  define MPFR_UINTMAX_MAX UINTMAX_MAX
#  define MPFR_INTMAX_MAX INTMAX_MAX
#  define MPFR_INTMAX_MIN INTMAX_MIN
# else
#  define MPFR_UINTMAX_MAX ((uintmax_t) -1)
#  define MPFR_INTMAX_MAX ((intmax_t) (MPFR_UINTMAX_MAX >> 1))
#  define MPFR_INTMAX_MIN (INT_MIN + INT_MAX - MPFR_INTMAX_MAX)
# endif
#endif

#define MPFR_BYTES_PER_MP_LIMB (GMP_NUMB_BITS/CHAR_BIT)

/******************************************************
 ******************** Check GMP ***********************
 ******************************************************/

#if !__MPFR_GMP(4,1,0)
# error "GMP 4.1.0 or newer needed"
#endif

#if GMP_NAIL_BITS != 0
# error "MPFR doesn't support nonzero values of GMP_NAIL_BITS"
#endif

#if (GMP_NUMB_BITS<32) || (GMP_NUMB_BITS & (GMP_NUMB_BITS - 1))
# error "GMP_NUMB_BITS must be a power of 2, and >= 32"
#endif

#if GMP_NUMB_BITS == 16
# define MPFR_LOG2_GMP_NUMB_BITS 4
#elif GMP_NUMB_BITS == 32
# define MPFR_LOG2_GMP_NUMB_BITS 5
#elif GMP_NUMB_BITS == 64
# define MPFR_LOG2_GMP_NUMB_BITS 6
#elif GMP_NUMB_BITS == 128
# define MPFR_LOG2_GMP_NUMB_BITS 7
#elif GMP_NUMB_BITS == 256
# define MPFR_LOG2_GMP_NUMB_BITS 8
#else
# error "Can't compute log2(GMP_NUMB_BITS)"
#endif

#if __MPFR_GNUC(3,0) || __MPFR_ICC(8,1,0)
/* For the future: N1478: Supporting the 'noreturn' property in C1x
   http://www.open-std.org/JTC1/SC22/WG14/www/docs/n1478.htm */
# define MPFR_NORETURN_ATTR __attribute__ ((noreturn))
# define MPFR_CONST_ATTR    __attribute__ ((const))
#else
# define MPFR_NORETURN_ATTR
# define MPFR_CONST_ATTR
#endif

/******************************************************
 ************* Global Internal Variables **************
 ******************************************************/

#if defined (__cplusplus)
extern "C" {
#endif

/* Cache struct */
struct __gmpfr_cache_s {
  mpfr_t x;
  int inexact;
  int (*func)(mpfr_ptr, mpfr_rnd_t);
};
typedef struct __gmpfr_cache_s mpfr_cache_t[1];
typedef struct __gmpfr_cache_s *mpfr_cache_ptr;

#if __GMP_LIBGMP_DLL
# define MPFR_WIN_THREAD_SAFE_DLL 1
#endif

#if defined(__MPFR_WITHIN_MPFR) || !defined(MPFR_WIN_THREAD_SAFE_DLL)
extern MPFR_THREAD_ATTR unsigned int __gmpfr_flags;
extern MPFR_THREAD_ATTR mpfr_exp_t   __gmpfr_emin;
extern MPFR_THREAD_ATTR mpfr_exp_t   __gmpfr_emax;
extern MPFR_THREAD_ATTR mpfr_prec_t  __gmpfr_default_fp_bit_precision;
extern MPFR_THREAD_ATTR mpfr_rnd_t   __gmpfr_default_rounding_mode;
extern MPFR_THREAD_ATTR mpfr_cache_t __gmpfr_cache_const_euler;
extern MPFR_THREAD_ATTR mpfr_cache_t __gmpfr_cache_const_catalan;
# ifndef MPFR_USE_LOGGING
extern MPFR_THREAD_ATTR mpfr_cache_t __gmpfr_cache_const_pi;
extern MPFR_THREAD_ATTR mpfr_cache_t __gmpfr_cache_const_log2;
# else
/* Two constants are used by the logging functions (via mpfr_fprintf,
   then mpfr_log, for the base conversion): pi and log(2). Since the
   mpfr_cache function isn't re-entrant when working on the same cache,
   we need to define two caches for each constant. */
extern MPFR_THREAD_ATTR mpfr_cache_t   __gmpfr_normal_pi;
extern MPFR_THREAD_ATTR mpfr_cache_t   __gmpfr_normal_log2;
extern MPFR_THREAD_ATTR mpfr_cache_t   __gmpfr_logging_pi;
extern MPFR_THREAD_ATTR mpfr_cache_t   __gmpfr_logging_log2;
extern MPFR_THREAD_ATTR mpfr_cache_ptr __gmpfr_cache_const_pi;
extern MPFR_THREAD_ATTR mpfr_cache_ptr __gmpfr_cache_const_log2;
# endif
#endif

#ifdef MPFR_WIN_THREAD_SAFE_DLL
# define MPFR_MAKE_VARFCT(T,N) T * N ## _f (void) { return &N; }
__MPFR_DECLSPEC unsigned int * __gmpfr_flags_f (void);
__MPFR_DECLSPEC mpfr_exp_t *   __gmpfr_emin_f (void);
__MPFR_DECLSPEC mpfr_exp_t *   __gmpfr_emax_f (void);
__MPFR_DECLSPEC mpfr_prec_t *  __gmpfr_default_fp_bit_precision_f (void);
__MPFR_DECLSPEC mpfr_rnd_t *   __gmpfr_default_rounding_mode_f (void);
__MPFR_DECLSPEC mpfr_cache_t * __gmpfr_cache_const_euler_f (void);
__MPFR_DECLSPEC mpfr_cache_t * __gmpfr_cache_const_catalan_f (void);
# ifndef MPFR_USE_LOGGING
__MPFR_DECLSPEC mpfr_cache_t * __gmpfr_cache_const_pi_f (void);
__MPFR_DECLSPEC mpfr_cache_t * __gmpfr_cache_const_log2_f (void);
# else
__MPFR_DECLSPEC mpfr_cache_t *   __gmpfr_normal_pi_f (void);
__MPFR_DECLSPEC mpfr_cache_t *   __gmpfr_normal_log2_f (void);
__MPFR_DECLSPEC mpfr_cache_t *   __gmpfr_logging_pi_f (void);
__MPFR_DECLSPEC mpfr_cache_t *   __gmpfr_logging_log2_f (void);
__MPFR_DECLSPEC mpfr_cache_ptr * __gmpfr_cache_const_pi_f (void);
__MPFR_DECLSPEC mpfr_cache_ptr * __gmpfr_cache_const_log2_f (void);
# endif
# ifndef __MPFR_WITHIN_MPFR
#  define __gmpfr_flags                    (*__gmpfr_flags_f())
#  define __gmpfr_emin                     (*__gmpfr_emin_f())
#  define __gmpfr_emax                     (*__gmpfr_emax_f())
#  define __gmpfr_default_fp_bit_precision (*__gmpfr_default_fp_bit_precision_f())
#  define __gmpfr_default_rounding_mode    (*__gmpfr_default_rounding_mode_f())
#  define __gmpfr_cache_const_euler        (*__gmpfr_cache_const_euler_f())
#  define __gmpfr_cache_const_catalan      (*__gmpfr_cache_const_catalan_f())
#  ifndef MPFR_USE_LOGGING
#   define __gmpfr_cache_const_pi         (*__gmpfr_cache_const_pi_f())
#   define __gmpfr_cache_const_log2       (*__gmpfr_cache_const_log2_f())
#  else
#   define __gmpfr_normal_pi              (*__gmpfr_normal_pi_f())
#   define __gmpfr_logging_pi             (*__gmpfr_logging_pi_f())
#   define __gmpfr_logging_log2           (*__gmpfr_logging_log2_f())
#   define __gmpfr_cache_const_pi         (*__gmpfr_cache_const_pi_f())
#   define __gmpfr_cache_const_log2       (*__gmpfr_cache_const_log2_f())
#  endif
# endif
#else
# define MPFR_MAKE_VARFCT(T,N)
#endif

# define MPFR_THREAD_VAR(T,N,V)    \
  MPFR_THREAD_ATTR T N = (V);      \
  MPFR_MAKE_VARFCT (T,N)

#define BASE_MAX 62
__MPFR_DECLSPEC extern const __mpfr_struct __gmpfr_l2b[BASE_MAX-1][2];

/* Note: do not use the following values when they can be outside the
   current exponent range, e.g. when the exponent range has not been
   extended yet; under such a condition, they can be used only in
   mpfr_cmpabs. */
__MPFR_DECLSPEC extern const mpfr_t __gmpfr_one;
__MPFR_DECLSPEC extern const mpfr_t __gmpfr_two;
__MPFR_DECLSPEC extern const mpfr_t __gmpfr_four;


#if defined (__cplusplus)
 }
#endif

/* Flags of __gmpfr_flags */
#define MPFR_FLAGS_UNDERFLOW 1
#define MPFR_FLAGS_OVERFLOW 2
#define MPFR_FLAGS_NAN 4
#define MPFR_FLAGS_INEXACT 8
#define MPFR_FLAGS_ERANGE 16
#define MPFR_FLAGS_DIVBY0 32
#define MPFR_FLAGS_ALL 63

/* Replace some common functions for direct access to the global vars.
   The casts prevent these macros from being used as a lvalue (and this
   method makes sure that the expressions have the correct type). */
#define mpfr_get_emin() ((mpfr_exp_t) __gmpfr_emin)
#define mpfr_get_emax() ((mpfr_exp_t) __gmpfr_emax)
#define mpfr_get_default_rounding_mode() \
  ((mpfr_rnd_t) __gmpfr_default_rounding_mode)
#define mpfr_get_default_prec() \
  ((mpfr_prec_t) __gmpfr_default_fp_bit_precision)

#define mpfr_clear_flags() \
  ((void) (__gmpfr_flags = 0))
#define mpfr_clear_underflow() \
  ((void) (__gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_UNDERFLOW))
#define mpfr_clear_overflow() \
  ((void) (__gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_OVERFLOW))
#define mpfr_clear_nanflag() \
  ((void) (__gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_NAN))
#define mpfr_clear_inexflag() \
  ((void) (__gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_INEXACT))
#define mpfr_clear_erangeflag() \
  ((void) (__gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_ERANGE))
#define mpfr_clear_divby0() \
  ((void) (__gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_DIVBY0))
#define mpfr_underflow_p() \
  ((int) (__gmpfr_flags & MPFR_FLAGS_UNDERFLOW))
#define mpfr_overflow_p() \
  ((int) (__gmpfr_flags & MPFR_FLAGS_OVERFLOW))
#define mpfr_nanflag_p() \
  ((int) (__gmpfr_flags & MPFR_FLAGS_NAN))
#define mpfr_inexflag_p() \
  ((int) (__gmpfr_flags & MPFR_FLAGS_INEXACT))
#define mpfr_erangeflag_p() \
  ((int) (__gmpfr_flags & MPFR_FLAGS_ERANGE))
#define mpfr_divby0_p() \
  ((int) (__gmpfr_flags & MPFR_FLAGS_DIVBY0))

/* Testing an exception flag correctly is tricky. There are mainly two
   pitfalls: First, one needs to remember to clear the corresponding
   flag, in case it was set before the function call or during some
   intermediate computations (in practice, one can clear all the flags).
   Secondly, one needs to test the flag early enough, i.e. before it
   can be modified by another function. Moreover, it is quite difficult
   (if not impossible) to reliably check problems with "make check". To
   avoid these pitfalls, it is recommended to use the following macros.
   Other use of the exception-flag predicate functions/macros will be
   detected by mpfrlint.
   Note: _op can be either a statement or an expression.
   MPFR_BLOCK_EXCEP should be used only inside a block; it is useful to
   detect some exception in order to exit the block as soon as possible. */
#define MPFR_BLOCK_DECL(_flags) unsigned int _flags
/* The (void) (_flags) makes sure that _flags is read at least once (it
   makes sense to use MPFR_BLOCK while _flags will never be read in the
   source, so that we wish to avoid the corresponding warning). */
#define MPFR_BLOCK(_flags,_op)          \
  do                                    \
    {                                   \
      mpfr_clear_flags ();              \
      _op;                              \
      (_flags) = __gmpfr_flags;         \
      (void) (_flags);                  \
    }                                   \
  while (0)
#define MPFR_BLOCK_TEST(_flags,_f) MPFR_UNLIKELY ((_flags) & (_f))
#define MPFR_BLOCK_EXCEP (__gmpfr_flags & (MPFR_FLAGS_UNDERFLOW | \
                                           MPFR_FLAGS_OVERFLOW | \
                                           MPFR_FLAGS_DIVBY0 | \
                                           MPFR_FLAGS_NAN))
/* Let's use a MPFR_ prefix, because e.g. OVERFLOW is defined by glibc's
   math.h, though this is not a reserved identifier! */
#define MPFR_UNDERFLOW(_flags)  MPFR_BLOCK_TEST (_flags, MPFR_FLAGS_UNDERFLOW)
#define MPFR_OVERFLOW(_flags)   MPFR_BLOCK_TEST (_flags, MPFR_FLAGS_OVERFLOW)
#define MPFR_NANFLAG(_flags)    MPFR_BLOCK_TEST (_flags, MPFR_FLAGS_NAN)
#define MPFR_INEXFLAG(_flags)   MPFR_BLOCK_TEST (_flags, MPFR_FLAGS_INEXACT)
#define MPFR_ERANGEFLAG(_flags) MPFR_BLOCK_TEST (_flags, MPFR_FLAGS_ERANGE)
#define MPFR_DIVBY0(_flags)     MPFR_BLOCK_TEST (_flags, MPFR_FLAGS_DIVBY0)


/******************************************************
 ******************** Assertions **********************
 ******************************************************/

/* Compile with -DMPFR_WANT_ASSERT to check all assert statements */

/* Note: do not use GMP macros ASSERT_ALWAYS and ASSERT as they are not
   expressions, and as a consequence, they cannot be used in a for(),
   with a comma operator and so on. */

/* MPFR_ASSERTN(expr): assertions that should always be checked */
#define MPFR_ASSERTN(expr)  \
   ((void) ((MPFR_UNLIKELY(expr)) || MPFR_UNLIKELY( (ASSERT_FAIL(expr),0) )))

/* MPFR_ASSERTD(expr): assertions that should be checked when testing */
#ifdef MPFR_WANT_ASSERT
# define MPFR_EXP_CHECK 1
# define MPFR_ASSERTD(expr)  MPFR_ASSERTN (expr)
#else
# define MPFR_ASSERTD(expr)  ((void) 0)
#endif

/* Code to deal with impossible
   WARNING: It doesn't use do { } while (0) for Insure++*/
#define MPFR_RET_NEVER_GO_HERE()  {MPFR_ASSERTN(0); return 0;}


/******************************************************
 ******************** Warnings ************************
 ******************************************************/

/* MPFR_WARNING is no longer useful, but let's keep the macro in case
   it needs to be used again in the future. */

#ifdef MPFR_USE_WARNINGS
# include <stdlib.h>
# define MPFR_WARNING(W)                    \
  do                                        \
    {                                       \
      char *q = getenv ("MPFR_QUIET");      \
      if (q == NULL || *q == 0)             \
        fprintf (stderr, "MPFR: %s\n", W);  \
    }                                       \
  while (0)
#else
# define MPFR_WARNING(W)  ((void) 0)
#endif


/******************************************************
 ****************** double macros *********************
 ******************************************************/

/* Precision used for lower precision computations */
#define MPFR_SMALL_PRECISION 32

/* Definition of constants */
#define LOG2 0.69314718055994528622 /* log(2) rounded to zero on 53 bits */
#define ALPHA 4.3191365662914471407 /* a+2 = a*log(a), rounded to +infinity */
#define EXPM1 0.36787944117144227851 /* exp(-1), rounded to zero */

/* MPFR_DOUBLE_SPEC = 1 if the C type 'double' corresponds to IEEE-754
   double precision, 0 if it doesn't, and undefined if one doesn't know.
   On all the tested machines, MPFR_DOUBLE_SPEC = 1. To have this macro
   defined here, #include <float.h> is needed. If need be, other values
   could be defined for other specs (once they are known). */
#if !defined(MPFR_DOUBLE_SPEC) && defined(FLT_RADIX) && \
    defined(DBL_MANT_DIG) && defined(DBL_MIN_EXP) && defined(DBL_MAX_EXP)
# if FLT_RADIX == 2 && DBL_MANT_DIG == 53 && \
     DBL_MIN_EXP == -1021 && DBL_MAX_EXP == 1024
#  define MPFR_DOUBLE_SPEC 1
# else
#  define MPFR_DOUBLE_SPEC 0
# endif
#endif

/* Debug non IEEE floats */
#ifdef XDEBUG
# undef _GMP_IEEE_FLOATS
#endif
#ifndef _GMP_IEEE_FLOATS
# define _GMP_IEEE_FLOATS 0
#endif

#ifndef IEEE_DBL_MANT_DIG
#define IEEE_DBL_MANT_DIG 53
#endif
#define MPFR_LIMBS_PER_DOUBLE ((IEEE_DBL_MANT_DIG-1)/GMP_NUMB_BITS+1)

#ifndef IEEE_FLT_MANT_DIG
#define IEEE_FLT_MANT_DIG 24
#endif
#define MPFR_LIMBS_PER_FLT ((IEEE_FLT_MANT_DIG-1)/GMP_NUMB_BITS+1)

/* Visual C++ doesn't support +1.0/0.0, -1.0/0.0 and 0.0/0.0
   at compile time.
   Clang with -fsanitize=undefined is a bit similar due to a bug:
     http://llvm.org/bugs/show_bug.cgi?id=17381
   but even without its sanitizer, it may be better to use the
   double_zero version until IEEE 754 division by zero is properly
   supported:
     http://llvm.org/bugs/show_bug.cgi?id=17000
*/
#if (defined(_MSC_VER) && defined(_WIN32) && (_MSC_VER >= 1200)) || \
    defined(__clang__)
static double double_zero = 0.0;
# define DBL_NAN (double_zero/double_zero)
# define DBL_POS_INF ((double) 1.0/double_zero)
# define DBL_NEG_INF ((double)-1.0/double_zero)
# define DBL_NEG_ZERO (-double_zero)
#else
# define DBL_POS_INF ((double) 1.0/0.0)
# define DBL_NEG_INF ((double)-1.0/0.0)
# define DBL_NAN     ((double) 0.0/0.0)
# define DBL_NEG_ZERO (-0.0)
#endif

/* Note: In the past, there was specific code for _GMP_IEEE_FLOATS, which
   was based on NaN and Inf memory representations. This code was breaking
   the aliasing rules (see ISO C99, 6.5#6 and 6.5#7 on the effective type)
   and for this reason it did not behave correctly with GCC 4.5.0 20091119.
   The code needed a memory transfer and was probably not better than the
   macros below with a good compiler (a fix based on the NaN / Inf memory
   representation would be even worse due to C limitations), and this code
   could be selected only when MPFR was built with --with-gmp-build, thus
   introducing a difference (bad for maintaining/testing MPFR); therefore
   it has been removed. The old code required that the argument x be an
   lvalue of type double. We still require that, in case one would need
   to change the macros below, e.g. for some broken compiler. But the
   LVALUE(x) condition could be removed if really necessary. */
/* Below, the &(x) == &(x) || &(x) != &(x) allows to make sure that x
   is a lvalue without (probably) any warning from the compiler.  The
   &(x) != &(x) is needed to avoid a failure under Mac OS X 10.4.11
   (with Xcode 2.4.1, i.e. the latest one). */
#define LVALUE(x) (&(x) == &(x) || &(x) != &(x))
#define DOUBLE_ISINF(x) (LVALUE(x) && ((x) > DBL_MAX || (x) < -DBL_MAX))
/* The DOUBLE_ISNAN(x) macro is also valid on long double x
   (assuming that the compiler isn't too broken). */
#ifdef MPFR_NANISNAN
/* Avoid MIPSpro / IRIX64 / gcc -ffast-math (incorrect) optimizations.
   The + must not be replaced by a ||. With gcc -ffast-math, NaN is
   regarded as a positive number or something like that; the second
   test catches this case. */
# define DOUBLE_ISNAN(x) \
    (LVALUE(x) && !((((x) >= 0.0) + ((x) <= 0.0)) && -(x)*(x) <= 0.0))
#else
# define DOUBLE_ISNAN(x) (LVALUE(x) && (x) != (x))
#endif

/******************************************************
 *************** Long double macros *******************
 ******************************************************/

/* We try to get the exact value of the precision of long double
   (provided by the implementation) in order to provide correct
   rounding in this case (not guaranteed if the C implementation
   does not have an adequate long double arithmetic). Note that
   it may be lower than the precision of some numbers that can
   be represented in a long double; e.g. on FreeBSD/x86, it is
   53 because the processor is configured to round in double
   precision (even when using the long double type -- this is a
   limitation of the x87 arithmetic), and on Mac OS X, it is 106
   because the implementation is a double-double arithmetic.
   Otherwise (e.g. in base 10), we get an upper bound of the
   precision, and correct rounding isn't currently provided.
*/
#if defined(LDBL_MANT_DIG) && FLT_RADIX == 2
# define MPFR_LDBL_MANT_DIG LDBL_MANT_DIG
#else
# define MPFR_LDBL_MANT_DIG \
  (sizeof(long double)*GMP_NUMB_BITS/sizeof(mp_limb_t))
#endif
#define MPFR_LIMBS_PER_LONG_DOUBLE \
  ((sizeof(long double)-1)/sizeof(mp_limb_t)+1)

/* LONGDOUBLE_NAN_ACTION executes the code "action" if x is a NaN. */

/* On hppa2.0n-hp-hpux10 with the unbundled HP cc, the test x!=x on a NaN
   has been seen false, meaning NaNs are not detected.  This seemed to
   happen only after other comparisons, not sure what's really going on.  In
   any case we can pick apart the bytes to identify a NaN.  */
#ifdef HAVE_LDOUBLE_IEEE_QUAD_BIG
# define LONGDOUBLE_NAN_ACTION(x, action)                       \
  do {                                                          \
    union {                                                     \
      long double    ld;                                        \
      struct {                                                  \
        unsigned int sign : 1;                                  \
        unsigned int exp  : 15;                                 \
        unsigned int man3 : 16;                                 \
        unsigned int man2 : 32;                                 \
        unsigned int man1 : 32;                                 \
        unsigned int man0 : 32;                                 \
      } s;                                                      \
    } u;                                                        \
    u.ld = (x);                                                 \
    if (u.s.exp == 0x7FFFL                                      \
        && (u.s.man0 | u.s.man1 | u.s.man2 | u.s.man3) != 0)    \
      { action; }                                               \
  } while (0)
#endif

#ifdef HAVE_LDOUBLE_IEEE_QUAD_LITTLE
# define LONGDOUBLE_NAN_ACTION(x, action)                       \
  do {                                                          \
    union {                                                     \
      long double    ld;                                        \
      struct {                                                  \
        unsigned int man0 : 32;                                 \
        unsigned int man1 : 32;                                 \
        unsigned int man2 : 32;                                 \
        unsigned int man3 : 16;                                 \
        unsigned int exp  : 15;                                 \
        unsigned int sign : 1;                                  \
      } s;                                                      \
    } u;                                                        \
    u.ld = (x);                                                 \
    if (u.s.exp == 0x7FFFL                                      \
        && (u.s.man0 | u.s.man1 | u.s.man2 | u.s.man3) != 0)    \
      { action; }                                               \
  } while (0)
#endif

/* Under IEEE rules, NaN is not equal to anything, including itself.
   "volatile" here stops "cc" on mips64-sgi-irix6.5 from optimizing away
   x!=x. */
#ifndef LONGDOUBLE_NAN_ACTION
# define LONGDOUBLE_NAN_ACTION(x, action)               \
  do {                                                  \
    volatile long double __x = LONGDOUBLE_VOLATILE (x); \
    if ((x) != __x)                                     \
      { action; }                                       \
  } while (0)
# define WANT_LONGDOUBLE_VOLATILE 1
#endif

/* If we don't have a proper "volatile" then volatile is #defined to empty,
   in this case call through an external function to stop the compiler
   optimizing anything. */
#ifdef WANT_LONGDOUBLE_VOLATILE
# ifdef volatile
__MPFR_DECLSPEC long double __gmpfr_longdouble_volatile _MPFR_PROTO ((long double)) MPFR_CONST_ATTR;
#  define LONGDOUBLE_VOLATILE(x)  (__gmpfr_longdouble_volatile (x))
#  define WANT_GMPFR_LONGDOUBLE_VOLATILE 1
# else
#  define LONGDOUBLE_VOLATILE(x)  (x)
# endif
#endif

/* Some special case for IEEE_EXT Litle Endian */
#if HAVE_LDOUBLE_IEEE_EXT_LITTLE

typedef union {
  long double    ld;
  struct {
    unsigned int manl : 32;
    unsigned int manh : 32;
    unsigned int expl : 8 ;
    unsigned int exph : 7;
    unsigned int sign : 1;
  } s;
} mpfr_long_double_t;

/* #undef MPFR_LDBL_MANT_DIG */
#undef MPFR_LIMBS_PER_LONG_DOUBLE
/* #define MPFR_LDBL_MANT_DIG   64 */
#define MPFR_LIMBS_PER_LONG_DOUBLE ((64-1)/GMP_NUMB_BITS+1)

#endif

/******************************************************
 *************** _Decimal64 support *******************
 ******************************************************/

#ifdef MPFR_WANT_DECIMAL_FLOATS
/* to cast between binary64 and decimal64 */
union ieee_double_decimal64 { double d; _Decimal64 d64; };
#endif

/******************************************************
 **************** mpfr_t properties *******************
 ******************************************************/

/* In the following macro, p is usually a mpfr_prec_t, but this macro
   works with other integer types (without integer overflow). Checking
   that p >= 1 in debug mode is useful here because this macro can be
   used on a computed precision (in particular, this formula does not
   work for a degenerate case p = 0, and could give different results
   on different platforms). But let us not use an assertion checking
   in the MPFR_LAST_LIMB() and MPFR_LIMB_SIZE() macros below to avoid
   too much expansion for assertions (in practice, this should be a
   problem just when testing MPFR with the --enable-assert configure
   option and the -ansi -pedantic-errors gcc compiler flags). */
#define MPFR_PREC2LIMBS(p) \
  (MPFR_ASSERTD ((p) >= 1), ((p) - 1) / GMP_NUMB_BITS + 1)

#define MPFR_PREC(x)      ((x)->_mpfr_prec)
#define MPFR_EXP(x)       ((x)->_mpfr_exp)
#define MPFR_MANT(x)      ((x)->_mpfr_d)
#define MPFR_LAST_LIMB(x) ((MPFR_PREC (x) - 1) / GMP_NUMB_BITS)
#define MPFR_LIMB_SIZE(x) (MPFR_LAST_LIMB (x) + 1)


/******************************************************
 **************** exponent properties *****************
 ******************************************************/

/* Limits of the mpfr_exp_t type (NOT those of valid exponent values).
   These macros can be used in preprocessor directives. */
#if   _MPFR_EXP_FORMAT == 1
# define MPFR_EXP_MAX (SHRT_MAX)
# define MPFR_EXP_MIN (SHRT_MIN)
#elif _MPFR_EXP_FORMAT == 2
# define MPFR_EXP_MAX (INT_MAX)
# define MPFR_EXP_MIN (INT_MIN)
#elif _MPFR_EXP_FORMAT == 3
# define MPFR_EXP_MAX (LONG_MAX)
# define MPFR_EXP_MIN (LONG_MIN)
#elif _MPFR_EXP_FORMAT == 4
# define MPFR_EXP_MAX (MPFR_INTMAX_MAX)
# define MPFR_EXP_MIN (MPFR_INTMAX_MIN)
#else
# error "Invalid MPFR Exp format"
#endif

/* Before doing a cast to mpfr_uexp_t, make sure that the value is
   nonnegative. */
#define MPFR_UEXP(X) (MPFR_ASSERTD ((X) >= 0), (mpfr_uexp_t) (X))

#if MPFR_EXP_MIN >= LONG_MIN && MPFR_EXP_MAX <= LONG_MAX
typedef long int mpfr_eexp_t;
# define mpfr_get_exp_t(x,r) mpfr_get_si((x),(r))
# define mpfr_set_exp_t(x,e,r) mpfr_set_si((x),(e),(r))
# define MPFR_EXP_FSPEC "l"
#elif defined (_MPFR_H_HAVE_INTMAX_T)
typedef intmax_t mpfr_eexp_t;
# define mpfr_get_exp_t(x,r) mpfr_get_sj((x),(r))
# define mpfr_set_exp_t(x,e,r) mpfr_set_sj((x),(e),(r))
# define MPFR_EXP_FSPEC "j"
#else
# error "Cannot define mpfr_get_exp_t and mpfr_set_exp_t"
#endif

/* Invalid exponent value (to track bugs...) */
#define MPFR_EXP_INVALID \
 ((mpfr_exp_t) 1 << (GMP_NUMB_BITS*sizeof(mpfr_exp_t)/sizeof(mp_limb_t)-2))

/* Definition of the exponent limits for MPFR numbers.
 * These limits are chosen so that if e is such an exponent, then 2e-1 and
 * 2e+1 are representable. This is useful for intermediate computations,
 * in particular the multiplication.
 */
#undef MPFR_EMIN_MIN
#undef MPFR_EMIN_MAX
#undef MPFR_EMAX_MIN
#undef MPFR_EMAX_MAX
#define MPFR_EMIN_MIN (1-MPFR_EXP_INVALID)
#define MPFR_EMIN_MAX (MPFR_EXP_INVALID-1)
#define MPFR_EMAX_MIN (1-MPFR_EXP_INVALID)
#define MPFR_EMAX_MAX (MPFR_EXP_INVALID-1)

/* Use MPFR_GET_EXP and MPFR_SET_EXP instead of MPFR_EXP directly,
   unless when the exponent may be out-of-range, for instance when
   setting the exponent before calling mpfr_check_range.
   MPFR_EXP_CHECK is defined when MPFR_WANT_ASSERT is defined, but if you
   don't use MPFR_WANT_ASSERT (for speed reasons), you can still define
   MPFR_EXP_CHECK by setting -DMPFR_EXP_CHECK in $CFLAGS. */

#ifdef MPFR_EXP_CHECK
# define MPFR_GET_EXP(x)          (mpfr_get_exp) (x)
# define MPFR_SET_EXP(x, exp)     MPFR_ASSERTN (!mpfr_set_exp ((x), (exp)))
# define MPFR_SET_INVALID_EXP(x)  ((void) (MPFR_EXP (x) = MPFR_EXP_INVALID))
#else
# define MPFR_GET_EXP(x)          MPFR_EXP (x)
# define MPFR_SET_EXP(x, exp)     ((void) (MPFR_EXP (x) = (exp)))
# define MPFR_SET_INVALID_EXP(x)  ((void) 0)
#endif



/******************************************************
 ********** Singular Values (NAN, INF, ZERO) **********
 ******************************************************/

/* Enum special value of exponent. */
# define MPFR_EXP_ZERO (MPFR_EXP_MIN+1)
# define MPFR_EXP_NAN  (MPFR_EXP_MIN+2)
# define MPFR_EXP_INF  (MPFR_EXP_MIN+3)

#define MPFR_IS_NAN(x)   (MPFR_EXP(x) == MPFR_EXP_NAN)
#define MPFR_SET_NAN(x)  (MPFR_EXP(x) =  MPFR_EXP_NAN)
#define MPFR_IS_INF(x)   (MPFR_EXP(x) == MPFR_EXP_INF)
#define MPFR_SET_INF(x)  (MPFR_EXP(x) =  MPFR_EXP_INF)
#define MPFR_IS_ZERO(x)  (MPFR_EXP(x) == MPFR_EXP_ZERO)
#define MPFR_SET_ZERO(x) (MPFR_EXP(x) =  MPFR_EXP_ZERO)
#define MPFR_NOTZERO(x)  (MPFR_EXP(x) != MPFR_EXP_ZERO)

#define MPFR_IS_FP(x)       (!MPFR_IS_NAN(x) && !MPFR_IS_INF(x))
#define MPFR_IS_SINGULAR(x) (MPFR_EXP(x) <= MPFR_EXP_INF)
#define MPFR_IS_PURE_FP(x)  (!MPFR_IS_SINGULAR(x) && \
  (MPFR_ASSERTD ((MPFR_MANT(x)[MPFR_LAST_LIMB(x)]  \
                  & MPFR_LIMB_HIGHBIT) != 0), 1))

#define MPFR_ARE_SINGULAR(x,y) \
  (MPFR_UNLIKELY(MPFR_IS_SINGULAR(x)) || MPFR_UNLIKELY(MPFR_IS_SINGULAR(y)))

#define MPFR_IS_POWER_OF_2(x) \
  (mpfr_cmp_ui_2exp ((x), 1, MPFR_GET_EXP (x) - 1) == 0)


/******************************************************
 ********************* Sign Macros ********************
 ******************************************************/

#define MPFR_SIGN_POS (1)
#define MPFR_SIGN_NEG (-1)

#define MPFR_IS_STRICTPOS(x)  (MPFR_NOTZERO((x)) && MPFR_SIGN(x) > 0)
#define MPFR_IS_STRICTNEG(x)  (MPFR_NOTZERO((x)) && MPFR_SIGN(x) < 0)

#define MPFR_IS_NEG(x) (MPFR_SIGN(x) < 0)
#define MPFR_IS_POS(x) (MPFR_SIGN(x) > 0)

#define MPFR_SET_POS(x) (MPFR_SIGN(x) = MPFR_SIGN_POS)
#define MPFR_SET_NEG(x) (MPFR_SIGN(x) = MPFR_SIGN_NEG)

#define MPFR_CHANGE_SIGN(x) (MPFR_SIGN(x) = -MPFR_SIGN(x))
#define MPFR_SET_SAME_SIGN(x, y) (MPFR_SIGN(x) = MPFR_SIGN(y))
#define MPFR_SET_OPPOSITE_SIGN(x, y) (MPFR_SIGN(x) = -MPFR_SIGN(y))
#define MPFR_ASSERT_SIGN(s) \
 (MPFR_ASSERTD((s) == MPFR_SIGN_POS || (s) == MPFR_SIGN_NEG))
#define MPFR_SET_SIGN(x, s) \
  (MPFR_ASSERT_SIGN(s), MPFR_SIGN(x) = s)
#define MPFR_IS_POS_SIGN(s1) ((s1) > 0)
#define MPFR_IS_NEG_SIGN(s1) ((s1) < 0)
#define MPFR_MULT_SIGN(s1, s2) ((s1) * (s2))
/* Transform a sign to 1 or -1 */
#define MPFR_FROM_SIGN_TO_INT(s) (s)
#define MPFR_INT_SIGN(x) MPFR_FROM_SIGN_TO_INT(MPFR_SIGN(x))



/******************************************************
 ***************** Ternary Value Macros ***************
 ******************************************************/

/* Special inexact value */
#define MPFR_EVEN_INEX 2

/* Macros for functions returning two inexact values in an 'int' */
#define INEXPOS(y) ((y) == 0 ? 0 : (((y) > 0) ? 1 : 2))
#define INEX(y,z) (INEXPOS(y) | (INEXPOS(z) << 2))

/* When returning the ternary inexact value, ALWAYS use one of the
   following two macros, unless the flag comes from another function
   returning the ternary inexact value */
#define MPFR_RET(I) return \
  (I) != 0 ? ((__gmpfr_flags |= MPFR_FLAGS_INEXACT), (I)) : 0
#define MPFR_RET_NAN return (__gmpfr_flags |= MPFR_FLAGS_NAN), 0

#define MPFR_SET_ERANGE() (__gmpfr_flags |= MPFR_FLAGS_ERANGE)

#define SIGN(I) ((I) < 0 ? -1 : (I) > 0)
#define SAME_SIGN(I1,I2) (SIGN (I1) == SIGN (I2))



/******************************************************
 ************** Rounding mode macros  *****************
 ******************************************************/

/* MPFR_RND_MAX gives the number of supported rounding modes by all functions.
 * Once faithful rounding is implemented, MPFR_RNDA should be changed
 * to MPFR_RNDF. But this will also require more changes in the tests.
 */
#define MPFR_RND_MAX ((mpfr_rnd_t)((MPFR_RNDA)+1))

/* We want to test this :
 *  (rnd == MPFR_RNDU && test) || (rnd == RNDD && !test)
 * ie it transforms RNDU or RNDD to Away or Zero according to the sign */
#define MPFR_IS_RNDUTEST_OR_RNDDNOTTEST(rnd, test) \
  (((rnd) + (test)) == MPFR_RNDD)

/* We want to test if rnd = Zero, or Away.
   'test' is 1 if negative, and 0 if positive. */
#define MPFR_IS_LIKE_RNDZ(rnd, test) \
  ((rnd) == MPFR_RNDZ || MPFR_IS_RNDUTEST_OR_RNDDNOTTEST (rnd, test))

#define MPFR_IS_LIKE_RNDU(rnd, sign)                    \
  (((rnd) == MPFR_RNDU) ||                              \
   ((rnd) == MPFR_RNDZ && MPFR_IS_NEG_SIGN (sign)) ||   \
   ((rnd) == MPFR_RNDA && MPFR_IS_POS_SIGN (sign)))

#define MPFR_IS_LIKE_RNDD(rnd, sign)                    \
  (((rnd) == MPFR_RNDD) ||                              \
   ((rnd) == MPFR_RNDZ && MPFR_IS_POS_SIGN (sign)) ||   \
   ((rnd) == MPFR_RNDA && MPFR_IS_NEG_SIGN (sign)))

/* Invert a rounding mode, RNDN, RNDZ and RNDA are unchanged */
#define MPFR_INVERT_RND(rnd) ((rnd) == MPFR_RNDU ? MPFR_RNDD :          \
                              (rnd) == MPFR_RNDD ? MPFR_RNDU : (rnd))

/* Transform RNDU and RNDD to RNDZ according to test */
#define MPFR_UPDATE_RND_MODE(rnd, test)                             \
  do {                                                              \
    if (MPFR_UNLIKELY(MPFR_IS_RNDUTEST_OR_RNDDNOTTEST(rnd, test)))  \
      rnd = MPFR_RNDZ;                                              \
  } while (0)

/* Transform RNDU and RNDD to RNDZ or RNDA according to sign,
   leave the other modes unchanged */
#define MPFR_UPDATE2_RND_MODE(rnd, sign)                       \
  do {                                                         \
    if (rnd == MPFR_RNDU)                                      \
      rnd = MPFR_IS_POS_SIGN (sign) ? MPFR_RNDA : MPFR_RNDZ;   \
    else if (rnd == MPFR_RNDD)                                 \
      rnd = MPFR_IS_NEG_SIGN (sign) ? MPFR_RNDA : MPFR_RNDZ;   \
  } while (0)


/******************************************************
 ******************* Limb Macros **********************
 ******************************************************/

 /* Definition of MPFR_LIMB_HIGHBIT */
#if defined(GMP_LIMB_HIGHBIT)
# define MPFR_LIMB_HIGHBIT GMP_LIMB_HIGHBIT
#elif defined(MP_LIMB_T_HIGHBIT)
# define MPFR_LIMB_HIGHBIT MP_LIMB_T_HIGHBIT
#else
# error "Neither GMP_LIMB_HIGHBIT nor MP_LIMB_T_HIGHBIT defined in GMP"
#endif

/* Mask to get the Most Significant Bit of a limb */
#define MPFR_LIMB_MSB(l) ((l)&MPFR_LIMB_HIGHBIT)

/* Definition of MPFR_LIMB_ONE & MPFR_LIMB_ZERO */
#ifdef CNST_LIMB
# define MPFR_LIMB_ONE  CNST_LIMB(1)
# define MPFR_LIMB_ZERO CNST_LIMB(0)
#else
# define MPFR_LIMB_ONE  ((mp_limb_t) 1L)
# define MPFR_LIMB_ZERO ((mp_limb_t) 0L)
#endif

/* Mask for the low 's' bits of a limb */
#define MPFR_LIMB_MASK(s) ((MPFR_LIMB_ONE<<(s))-MPFR_LIMB_ONE)



/******************************************************
 ********************** Memory ************************
 ******************************************************/

/* Heap Memory gestion */
typedef union { mp_size_t s; mp_limb_t l; } mpfr_size_limb_t;
#define MPFR_GET_ALLOC_SIZE(x) \
 ( ((mp_size_t*) MPFR_MANT(x))[-1] + 0)
#define MPFR_SET_ALLOC_SIZE(x, n) \
 ( ((mp_size_t*) MPFR_MANT(x))[-1] = n)
#define MPFR_MALLOC_SIZE(s) \
  ( sizeof(mpfr_size_limb_t) + MPFR_BYTES_PER_MP_LIMB * ((size_t) s) )
#define MPFR_SET_MANT_PTR(x,p) \
   (MPFR_MANT(x) = (mp_limb_t*) ((mpfr_size_limb_t*) p + 1))
#define MPFR_GET_REAL_PTR(x) \
   ((mp_limb_t*) ((mpfr_size_limb_t*) MPFR_MANT(x) - 1))

/* Temporary memory gestion */
#ifndef TMP_SALLOC
/* GMP 4.1.x or below or internals */
#define MPFR_TMP_DECL TMP_DECL
#define MPFR_TMP_MARK TMP_MARK
#define MPFR_TMP_ALLOC TMP_ALLOC
#define MPFR_TMP_FREE TMP_FREE
#else
#define MPFR_TMP_DECL(x) TMP_DECL
#define MPFR_TMP_MARK(x) TMP_MARK
#define MPFR_TMP_ALLOC(s) TMP_ALLOC(s)
#define MPFR_TMP_FREE(x) TMP_FREE
#endif

/* This code is experimental: don't use it */
#ifdef MPFR_USE_OWN_MPFR_TMP_ALLOC
extern unsigned char *mpfr_stack;
#undef MPFR_TMP_DECL
#undef MPFR_TMP_MARK
#undef MPFR_TMP_ALLOC
#undef MPFR_TMP_FREE
#define MPFR_TMP_DECL(_x) unsigned char *(_x)
#define MPFR_TMP_MARK(_x) ((_x) = mpfr_stack)
#define MPFR_TMP_ALLOC(_s) (mpfr_stack += (_s), mpfr_stack - (_s))
#define MPFR_TMP_FREE(_x) (mpfr_stack = (_x))
#endif

#define MPFR_TMP_LIMBS_ALLOC(N) \
  ((mp_limb_t *) MPFR_TMP_ALLOC ((size_t) (N) * MPFR_BYTES_PER_MP_LIMB))

/* temporary allocate 1 limb at xp, and initialize mpfr variable x */
/* The temporary var doesn't have any size field, but it doesn't matter
 * since only functions dealing with the Heap care about it */
#define MPFR_TMP_INIT1(xp, x, p)                                     \
 ( MPFR_PREC(x) = (p),                                               \
   MPFR_MANT(x) = (xp),                                              \
   MPFR_SET_POS(x),                                                  \
   MPFR_SET_INVALID_EXP(x))

#define MPFR_TMP_INIT(xp, x, p, s)                                   \
  (xp = MPFR_TMP_LIMBS_ALLOC(s),                                     \
   MPFR_TMP_INIT1(xp, x, p))

#define MPFR_TMP_INIT_ABS(d, s)                                      \
 ( MPFR_PREC(d) = MPFR_PREC(s),                                      \
   MPFR_MANT(d) = MPFR_MANT(s),                                      \
   MPFR_SET_POS(d),                                                  \
   MPFR_EXP(d)  = MPFR_EXP(s))



/******************************************************
 *****************  Cache macros **********************
 ******************************************************/

#define mpfr_const_pi(_d,_r)    mpfr_cache(_d, __gmpfr_cache_const_pi,_r)
#define mpfr_const_log2(_d,_r)  mpfr_cache(_d, __gmpfr_cache_const_log2, _r)
#define mpfr_const_euler(_d,_r) mpfr_cache(_d, __gmpfr_cache_const_euler, _r)
#define mpfr_const_catalan(_d,_r) mpfr_cache(_d,__gmpfr_cache_const_catalan,_r)

#define MPFR_DECL_INIT_CACHE(_cache,_func)                           \
  MPFR_THREAD_ATTR mpfr_cache_t _cache =                             \
    {{{{0,MPFR_SIGN_POS,0,(mp_limb_t*)0}},0,_func}};                 \
  MPFR_MAKE_VARFCT (mpfr_cache_t,_cache)



/******************************************************
 *******************  Threshold ***********************
 ******************************************************/

#include "mparam.h"

/******************************************************
 *****************  Useful macros *********************
 ******************************************************/

/* Theses macros help the compiler to determine if a test is
   likely or unlikely. The !! is necessary in case x is larger
   than a long. */
#if __MPFR_GNUC(3,0) || __MPFR_ICC(8,1,0)
# define MPFR_LIKELY(x) (__builtin_expect(!!(x),1))
# define MPFR_UNLIKELY(x) (__builtin_expect(!!(x),0))
#else
# define MPFR_LIKELY(x) (x)
# define MPFR_UNLIKELY(x) (x)
#endif

/* Declare that some variable is initialized before being used (without a
   dummy initialization) in order to avoid some compiler warnings. Use the
   VAR = VAR trick (see http://gcc.gnu.org/bugzilla/show_bug.cgi?id=36296)
   only with gcc as this is undefined behavior, and we don't know what
   other compilers do (they may also be smarter). This trick could be
   disabled with future gcc versions. */
#if defined(__GNUC__)
# define INITIALIZED(VAR) VAR = VAR
#else
# define INITIALIZED(VAR) VAR
#endif

/* Ceil log 2: If GCC, uses a GCC extension, otherwise calls a function */
/* Warning:
 *   Needs to define MPFR_NEED_LONGLONG.
 *   Computes ceil(log2(x)) only for x integer (unsigned long)
 *   Undefined if x is 0 */
#if __MPFR_GNUC(2,95) || __MPFR_ICC(8,1,0)
# define MPFR_INT_CEIL_LOG2(x)                            \
    (MPFR_UNLIKELY ((x) == 1) ? 0 :                       \
     __extension__ ({ int _b; mp_limb_t _limb;            \
      MPFR_ASSERTN ((x) > 1);                             \
      _limb = (x) - 1;                                    \
      MPFR_ASSERTN (_limb == (x) - 1);                    \
      count_leading_zeros (_b, _limb);                    \
      (GMP_NUMB_BITS - _b); }))
#else
# define MPFR_INT_CEIL_LOG2(x) (__gmpfr_int_ceil_log2(x))
#endif

/* Add two integers with overflow handling */
/* Example: MPFR_SADD_OVERFLOW (c, a, b, long, unsigned long,
 *                              LONG_MIN, LONG_MAX,
 *                              goto overflow, goto underflow); */
#define MPFR_UADD_OVERFLOW(c,a,b,ACTION_IF_OVERFLOW)                  \
 do {                                                                 \
  (c) = (a) + (b);                                                    \
  if ((c) < (a)) ACTION_IF_OVERFLOW;                                  \
 } while (0)

#define MPFR_SADD_OVERFLOW(c,a,b,STYPE,UTYPE,MIN,MAX,ACTION_IF_POS_OVERFLOW,ACTION_IF_NEG_OVERFLOW) \
  do {                                                                \
  if ((a) >= 0 && (b) >= 0) {                                         \
         UTYPE uc,ua,ub;                                              \
         ua = (UTYPE) (a); ub = (UTYPE) (b);                          \
         MPFR_UADD_OVERFLOW (uc, ua, ub, ACTION_IF_POS_OVERFLOW);     \
         if (uc > (UTYPE)(MAX)) ACTION_IF_POS_OVERFLOW;               \
         else (c) = (STYPE) uc;                                       \
  } else if ((a) < 0 && (b) < 0) {                                    \
         UTYPE uc,ua,ub;                                              \
         ua = -(UTYPE) (a); ub = -(UTYPE) (b);                        \
         MPFR_UADD_OVERFLOW (uc, ua, ub, ACTION_IF_NEG_OVERFLOW);     \
         if (uc >= -(UTYPE)(MIN) || uc > (UTYPE)(MAX)) {              \
           if (uc ==  -(UTYPE)(MIN)) (c) = (MIN);                     \
           else  ACTION_IF_NEG_OVERFLOW; }                            \
         else (c) = -(STYPE) uc;                                      \
  } else (c) = (a) + (b);                                             \
 } while (0)


/* Set a number to 1 (Fast) - It doesn't check if 1 is in the exponent range */
#define MPFR_SET_ONE(x)                                               \
do {                                                                  \
  mp_size_t _size = MPFR_LAST_LIMB(x);                                \
  MPFR_SET_POS(x);                                                    \
  MPFR_EXP(x) = 1;                                                    \
  MPN_ZERO ( MPFR_MANT(x), _size);                                    \
  MPFR_MANT(x)[_size] = MPFR_LIMB_HIGHBIT;                            \
} while (0)

/* Compute s = (-a) % GMP_NUMB_BITS as unsigned */
#define MPFR_UNSIGNED_MINUS_MODULO(s, a)                              \
  do                                                                  \
    {                                                                 \
      if (IS_POW2 (GMP_NUMB_BITS))                                    \
        (s) = (- (unsigned int) (a)) % GMP_NUMB_BITS;                 \
      else                                                            \
        {                                                             \
          (s) = (a) % GMP_NUMB_BITS;                                  \
          if ((s) != 0)                                               \
            (s) = GMP_NUMB_BITS - (s);                                \
        }                                                             \
      MPFR_ASSERTD ((s) >= 0 && (s) < GMP_NUMB_BITS);                 \
    }                                                                 \
  while (0)

/* Use it only for debug reasons */
/*   MPFR_TRACE (operation) : execute operation iff DEBUG flag is set */
/*   MPFR_DUMP (x) : print x (a mpfr_t) on stdout */
#ifdef DEBUG
# define MPFR_TRACE(x) x
#else
# define MPFR_TRACE(x) (void) 0
#endif
#define MPFR_DUMP(x) ( printf(#x"="), mpfr_dump(x) )

/* Test if X (positive) is a power of 2 */
#define IS_POW2(X) (((X) & ((X) - 1)) == 0)
#define NOT_POW2(X) (((X) & ((X) - 1)) != 0)

/* Safe absolute value (to avoid possible integer overflow) */
/* type is the target (unsigned) type */
#define SAFE_ABS(type,x) ((x) >= 0 ? (type)(x) : -(type)(x))

#define mpfr_get_d1(x) mpfr_get_d(x,__gmpfr_default_rounding_mode)

/* Store in r the size in bits of the mpz_t z */
#define MPFR_MPZ_SIZEINBASE2(r, z)              \
  do {                                          \
   int _cnt;                                    \
   mp_size_t _size;                             \
   MPFR_ASSERTD (mpz_sgn (z) != 0);             \
   _size = ABSIZ(z);                            \
   count_leading_zeros (_cnt, PTR(z)[_size-1]); \
   (r) = _size * GMP_NUMB_BITS - _cnt;       \
  } while (0)

/* MPFR_LCONV_DPTS can also be forced to 0 or 1 by the user. */
#ifndef MPFR_LCONV_DPTS
# if defined(HAVE_LOCALE_H) && \
     defined(HAVE_STRUCT_LCONV_DECIMAL_POINT) && \
     defined(HAVE_STRUCT_LCONV_THOUSANDS_SEP)
#  define MPFR_LCONV_DPTS 1
# else
#  define MPFR_LCONV_DPTS 0
# endif
#endif

/* FIXME: Add support for multibyte decimal_point and thousands_sep since
   this can be found in practice: https://reviews.llvm.org/D27167 says:
   "I found this problem on FreeBSD 11, where thousands_sep in fr_FR.UTF-8
   is a no-break space (U+00A0)."
   Note, however, that this is not allowed by the C standard, which just
   says "character" and not "multibyte character".
   In the mean time, in case of non-single-byte character, revert to the
   default value. */
#if MPFR_LCONV_DPTS
#include <locale.h>
/* Warning! In case of signed char, the value of MPFR_DECIMAL_POINT may
   be negative (the ISO C99 does not seem to forbid negative values). */
#define MPFR_DECIMAL_POINT                      \
  (localeconv()->decimal_point[1] != '\0' ?     \
   (char) '.' : localeconv()->decimal_point[0])
#define MPFR_THOUSANDS_SEPARATOR                \
  (localeconv()->thousands_sep[0] == '\0' ||    \
   localeconv()->thousands_sep[1] != '\0' ?     \
   (char) '\0' : localeconv()->thousands_sep[0])
#else
#define MPFR_DECIMAL_POINT ((char) '.')
#define MPFR_THOUSANDS_SEPARATOR ((char) '\0')
#endif


/* Set y to s*significand(x)*2^e, for example MPFR_ALIAS(y,x,1,MPFR_EXP(x))
   sets y to |x|, and MPFR_ALIAS(y,x,MPFR_SIGN(x),0) sets y to x*2^f such
   that 1/2 <= |y| < 1. Does not check y is in the valid exponent range.
   WARNING! x and y share the same mantissa. So, some operations are
   not valid if x has been provided via an argument, e.g., trying to
   modify the mantissa of y, even temporarily, or calling mpfr_clear on y.
*/
#define MPFR_ALIAS(y,x,s,e)                     \
  do                                            \
    {                                           \
      MPFR_PREC(y) = MPFR_PREC(x);              \
      MPFR_SIGN(y) = (s);                       \
      MPFR_EXP(y) = (e);                        \
      MPFR_MANT(y) = MPFR_MANT(x);              \
    } while (0)


/******************************************************
 **************  Save exponent macros  ****************
 ******************************************************/

/* See README.dev for details on how to use the macros.
   They are used to set the exponent range to the maximum
   temporarily */

typedef struct {
  unsigned int saved_flags;
  mpfr_exp_t saved_emin;
  mpfr_exp_t saved_emax;
} mpfr_save_expo_t;

/* Minimum and maximum exponents of the extended exponent range. */
#define MPFR_EXT_EMIN MPFR_EMIN_MIN
#define MPFR_EXT_EMAX MPFR_EMAX_MAX

#define MPFR_SAVE_EXPO_DECL(x) mpfr_save_expo_t x
#define MPFR_SAVE_EXPO_MARK(x)     \
 ((x).saved_flags = __gmpfr_flags, \
  (x).saved_emin = __gmpfr_emin,   \
  (x).saved_emax = __gmpfr_emax,   \
  __gmpfr_emin = MPFR_EXT_EMIN,    \
  __gmpfr_emax = MPFR_EXT_EMAX)
#define MPFR_SAVE_EXPO_FREE(x)     \
 (__gmpfr_flags = (x).saved_flags, \
  __gmpfr_emin = (x).saved_emin,   \
  __gmpfr_emax = (x).saved_emax)
#define MPFR_SAVE_EXPO_UPDATE_FLAGS(x, flags)  \
  (x).saved_flags |= (flags)

/* Speed up final checking */
#define mpfr_check_range(x,t,r) \
 (MPFR_LIKELY (MPFR_EXP (x) >= __gmpfr_emin && MPFR_EXP (x) <= __gmpfr_emax) \
  ? ((t) ? (__gmpfr_flags |= MPFR_FLAGS_INEXACT, (t)) : 0)                   \
  : mpfr_check_range(x,t,r))


/******************************************************
 *****************  Inline Rounding *******************
 ******************************************************/

/*
 * Note: due to the labels, one cannot use a macro MPFR_RNDRAW* more than
 * once in a function (otherwise these labels would not be unique).
 */

/*
 * Round mantissa (srcp, sprec) to mpfr_t dest using rounding mode rnd
 * assuming dest's sign is sign.
 * In rounding to nearest mode, execute MIDDLE_HANDLER when the value
 * is the middle of two consecutive numbers in dest precision.
 * Execute OVERFLOW_HANDLER in case of overflow when rounding.
 */
#define MPFR_RNDRAW_GEN(inexact, dest, srcp, sprec, rnd, sign,              \
                        MIDDLE_HANDLER, OVERFLOW_HANDLER)                   \
  do {                                                                      \
    mp_size_t _dests, _srcs;                                                \
    mp_limb_t *_destp;                                                      \
    mpfr_prec_t _destprec, _srcprec;                                        \
                                                                            \
    /* Check Trivial Case when Dest Mantissa has more bits than source */   \
    _srcprec = (sprec);                                                     \
    _destprec = MPFR_PREC (dest);                                           \
    _destp = MPFR_MANT (dest);                                              \
    if (MPFR_UNLIKELY (_destprec >= _srcprec))                              \
      {                                                                     \
        _srcs  = MPFR_PREC2LIMBS (_srcprec);                                \
        _dests = MPFR_PREC2LIMBS (_destprec) - _srcs;                       \
        MPN_COPY (_destp + _dests, srcp, _srcs);                            \
        MPN_ZERO (_destp, _dests);                                          \
        inexact = 0;                                                        \
      }                                                                     \
    else                                                                    \
      {                                                                     \
        /* Non trivial case: rounding needed */                             \
        mpfr_prec_t _sh;                                                    \
        mp_limb_t *_sp;                                                     \
        mp_limb_t _rb, _sb, _ulp;                                           \
                                                                            \
        /* Compute Position and shift */                                    \
        _srcs  = MPFR_PREC2LIMBS (_srcprec);                                \
        _dests = MPFR_PREC2LIMBS (_destprec);                               \
        MPFR_UNSIGNED_MINUS_MODULO (_sh, _destprec);                        \
        _sp = (srcp) + _srcs - _dests;                                      \
                                                                            \
        /* General case when prec % GMP_NUMB_BITS != 0 */                   \
        if (MPFR_LIKELY (_sh != 0))                                         \
          {                                                                 \
            mp_limb_t _mask;                                                \
            /* Compute Rounding Bit and Sticky Bit */                       \
            /* Note: in directed rounding modes, if the rounding bit */     \
            /* is 1, the behavior does not depend on the sticky bit; */     \
            /* thus we will not try to compute it in this case (this */     \
            /* can be much faster and avoids to read uninitialized   */     \
            /* data in the current mpfr_mul implementation). We just */     \
            /* make sure that _sb is initialized.                    */     \
            _mask = MPFR_LIMB_ONE << (_sh - 1);                             \
            _rb = _sp[0] & _mask;                                           \
            _sb = _sp[0] & (_mask - 1);                                     \
            if (MPFR_UNLIKELY (_sb == 0) &&                                 \
                ((rnd) == MPFR_RNDN || _rb == 0))                           \
              { /* TODO: Improve it */                                      \
                mp_limb_t *_tmp;                                            \
                mp_size_t _n;                                               \
                for (_tmp = _sp, _n = _srcs - _dests ;                      \
                     _n != 0 && _sb == 0 ; _n--)                            \
                  _sb = *--_tmp;                                            \
              }                                                             \
            _ulp = 2 * _mask;                                               \
          }                                                                 \
        else /* _sh == 0 */                                                 \
          {                                                                 \
            MPFR_ASSERTD (_dests < _srcs);                                  \
            /* Compute Rounding Bit and Sticky Bit - see note above */      \
            _rb = _sp[-1] & MPFR_LIMB_HIGHBIT;                              \
            _sb = _sp[-1] & (MPFR_LIMB_HIGHBIT-1);                          \
            if (MPFR_UNLIKELY (_sb == 0) &&                                 \
                ((rnd) == MPFR_RNDN || _rb == 0))                           \
              {                                                             \
                mp_limb_t *_tmp;                                            \
                mp_size_t _n;                                               \
                for (_tmp = _sp - 1, _n = _srcs - _dests - 1 ;              \
                     _n != 0 && _sb == 0 ; _n--)                            \
                  _sb = *--_tmp;                                            \
              }                                                             \
            _ulp = MPFR_LIMB_ONE;                                           \
          }                                                                 \
        /* Rounding */                                                      \
        if (MPFR_LIKELY (rnd == MPFR_RNDN))                                 \
          {                                                                 \
            if (_rb == 0)                                                   \
              {                                                             \
              trunc:                                                        \
                inexact = MPFR_LIKELY ((_sb | _rb) != 0) ? -sign : 0;       \
              trunc_doit:                                                   \
                MPN_COPY (_destp, _sp, _dests);                             \
                _destp[0] &= ~(_ulp - 1);                                   \
              }                                                             \
            else if (MPFR_UNLIKELY (_sb == 0))                              \
              { /* Middle of two consecutive representable numbers */       \
                MIDDLE_HANDLER;                                             \
              }                                                             \
            else                                                            \
              {                                                             \
                if (0)                                                      \
                  goto addoneulp_doit; /* dummy code to avoid warning */    \
              addoneulp:                                                    \
                inexact = sign;                                             \
              addoneulp_doit:                                               \
                if (MPFR_UNLIKELY (mpn_add_1 (_destp, _sp, _dests, _ulp)))  \
                  {                                                         \
                    _destp[_dests - 1] = MPFR_LIMB_HIGHBIT;                 \
                    OVERFLOW_HANDLER;                                       \
                  }                                                         \
                _destp[0] &= ~(_ulp - 1);                                   \
              }                                                             \
          }                                                                 \
        else                                                                \
          { /* Directed rounding mode */                                    \
            if (MPFR_LIKELY (MPFR_IS_LIKE_RNDZ (rnd,                        \
                                                MPFR_IS_NEG_SIGN (sign))))  \
              goto trunc;                                                   \
             else if (MPFR_UNLIKELY ((_sb | _rb) == 0))                     \
               {                                                            \
                 inexact = 0;                                               \
                 goto trunc_doit;                                           \
               }                                                            \
             else                                                           \
              goto addoneulp;                                               \
          }                                                                 \
      }                                                                     \
  } while (0)

/*
 * Round mantissa (srcp, sprec) to mpfr_t dest using rounding mode rnd
 * assuming dest's sign is sign.
 * Execute OVERFLOW_HANDLER in case of overflow when rounding.
 */
#define MPFR_RNDRAW(inexact, dest, srcp, sprec, rnd, sign, OVERFLOW_HANDLER) \
   MPFR_RNDRAW_GEN (inexact, dest, srcp, sprec, rnd, sign,                   \
     if ((_sp[0] & _ulp) == 0)                                               \
       {                                                                     \
         inexact = -sign;                                                    \
         goto trunc_doit;                                                    \
       }                                                                     \
     else                                                                    \
       goto addoneulp;                                                       \
     , OVERFLOW_HANDLER)

/*
 * Round mantissa (srcp, sprec) to mpfr_t dest using rounding mode rnd
 * assuming dest's sign is sign.
 * Execute OVERFLOW_HANDLER in case of overflow when rounding.
 * Set inexact to +/- MPFR_EVEN_INEX in case of even rounding.
 */
#define MPFR_RNDRAW_EVEN(inexact, dest, srcp, sprec, rnd, sign, \
                         OVERFLOW_HANDLER)                      \
   MPFR_RNDRAW_GEN (inexact, dest, srcp, sprec, rnd, sign,      \
     if ((_sp[0] & _ulp) == 0)                                  \
       {                                                        \
         inexact = -MPFR_EVEN_INEX * sign;                      \
         goto trunc_doit;                                       \
       }                                                        \
     else                                                       \
       {                                                        \
         inexact = MPFR_EVEN_INEX * sign;                       \
         goto addoneulp_doit;                                   \
       }                                                        \
     , OVERFLOW_HANDLER)

/* Return TRUE if b is non singular and we can round it to precision 'prec'
   and determine the ternary value, with rounding mode 'rnd', and with
   error at most 'error' */
#define MPFR_CAN_ROUND(b,err,prec,rnd)                                       \
 (!MPFR_IS_SINGULAR (b) && mpfr_round_p (MPFR_MANT (b), MPFR_LIMB_SIZE (b),  \
                                         (err), (prec) + ((rnd)==MPFR_RNDN)))

/* Copy the sign and the significand, and handle the exponent in exp. */
#define MPFR_SETRAW(inexact,dest,src,exp,rnd)                           \
  if (MPFR_UNLIKELY (dest != src))                                      \
    {                                                                   \
      MPFR_SET_SIGN (dest, MPFR_SIGN (src));                            \
      if (MPFR_LIKELY (MPFR_PREC (dest) == MPFR_PREC (src)))            \
        {                                                               \
          MPN_COPY (MPFR_MANT (dest), MPFR_MANT (src),                  \
                    MPFR_LIMB_SIZE (src));                              \
          inexact = 0;                                                  \
        }                                                               \
      else                                                              \
        {                                                               \
          MPFR_RNDRAW (inexact, dest, MPFR_MANT (src), MPFR_PREC (src), \
                       rnd, MPFR_SIGN (src), exp++);                    \
        }                                                               \
    }                                                                   \
  else                                                                  \
    inexact = 0;

/* TODO: fix this description (see round_near_x.c). */
/* Assuming that the function has a Taylor expansion which looks like:
    y=o(f(x)) = o(v + g(x)) with |g(x)| <= 2^(EXP(v)-err)
   we can quickly set y to v if x is small (ie err > prec(y)+1) in most
   cases. It assumes that f(x) is not representable exactly as a FP number.
   v must not be a singular value (NAN, INF or ZERO); usual values are
   v=1 or v=x.

   y is the destination (a mpfr_t), v the value to set (a mpfr_t),
   err1+err2 with err2 <= 3 the error term (mpfr_exp_t's), dir (an int) is
   the direction of the committed error (if dir = 0, it rounds toward 0,
   if dir=1, it rounds away from 0), rnd the rounding mode.

   It returns from the function a ternary value in case of success.
   If you want to free something, you must fill the "extra" field
   in consequences, otherwise put nothing in it.

   The test is less restrictive than necessary, but the function
   will finish the check itself.

   Note: err1 + err2 is allowed to overflow as mpfr_exp_t, but it must give
   its real value as mpfr_uexp_t.
*/
#define MPFR_FAST_COMPUTE_IF_SMALL_INPUT(y,v,err1,err2,dir,rnd,extra)   \
  do {                                                                  \
    mpfr_ptr _y = (y);                                                  \
    mpfr_exp_t _err1 = (err1);                                          \
    mpfr_exp_t _err2 = (err2);                                          \
    if (_err1 > 0)                                                      \
      {                                                                 \
        mpfr_uexp_t _err = (mpfr_uexp_t) _err1 + _err2;                 \
        if (MPFR_UNLIKELY (_err > MPFR_PREC (_y) + 1))                  \
          {                                                             \
            int _inexact = mpfr_round_near_x (_y,(v),_err,(dir),(rnd)); \
            if (_inexact != 0)                                          \
              {                                                         \
                extra;                                                  \
                return _inexact;                                        \
              }                                                         \
          }                                                             \
      }                                                                 \
  } while (0)

/* Variant, to be called somewhere after MPFR_SAVE_EXPO_MARK. This variant
   is needed when there are some computations before or when some non-zero
   real constant is used, such as __gmpfr_one for mpfr_cos. */
#define MPFR_SMALL_INPUT_AFTER_SAVE_EXPO(y,v,err1,err2,dir,rnd,expo,extra) \
  do {                                                                  \
    mpfr_ptr _y = (y);                                                  \
    mpfr_exp_t _err1 = (err1);                                          \
    mpfr_exp_t _err2 = (err2);                                          \
    if (_err1 > 0)                                                      \
      {                                                                 \
        mpfr_uexp_t _err = (mpfr_uexp_t) _err1 + _err2;                 \
        if (MPFR_UNLIKELY (_err > MPFR_PREC (_y) + 1))                  \
          {                                                             \
            int _inexact;                                               \
            mpfr_clear_flags ();                                        \
            _inexact = mpfr_round_near_x (_y,(v),_err,(dir),(rnd));     \
            if (_inexact != 0)                                          \
              {                                                         \
                extra;                                                  \
                MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);      \
                MPFR_SAVE_EXPO_FREE (expo);                             \
                return mpfr_check_range (_y, _inexact, (rnd));          \
              }                                                         \
          }                                                             \
      }                                                                 \
  } while (0)

/******************************************************
 ***************  Ziv Loop Macro  *********************
 ******************************************************/

#ifndef MPFR_USE_LOGGING

#define MPFR_ZIV_DECL(_x) mpfr_prec_t _x
#define MPFR_ZIV_INIT(_x, _p) (_x) = GMP_NUMB_BITS
#define MPFR_ZIV_NEXT(_x, _p) ((_p) += (_x), (_x) = (_p)/2)
#define MPFR_ZIV_FREE(x)

#else

/* The following test on glibc is there mainly for Darwin (Mac OS X), to
   obtain a better error message. The real test should have been a test
   concerning nested functions in gcc, which are disabled by default on
   Darwin; but it is not possible to do that without a configure test. */
# if defined (__cplusplus) || !(__MPFR_GNUC(3,0) && __MPFR_GLIBC(2,0))
#  error "Logging not supported (needs gcc >= 3.0 and GNU C Library >= 2.0)."
# endif

/* Use LOGGING */

/* Note: the mpfr_log_level >= 0 below avoids to take into account
   Ziv loops used by the MPFR functions called by the mpfr_fprintf
   in LOG_PRINT. */

#define MPFR_ZIV_DECL(_x)                                               \
  mpfr_prec_t _x;                                                       \
  int _x ## _cpt = 1;                                                   \
  static unsigned long  _x ## _loop = 0, _x ## _bad = 0;                \
  static const char *_x ## _fname = __func__;                           \
  auto void __attribute__ ((destructor)) x ## _f  (void);               \
  void __attribute__ ((destructor)) x ## _f  (void) {                   \
    if (_x ## _loop != 0 && (MPFR_LOG_STAT_F & mpfr_log_type))          \
      fprintf (mpfr_log_file,                                           \
               "%s: Ziv failed %2.2f%% (%lu bad cases / %lu calls)\n",  \
               _x ## _fname, (double) 100.0 * _x ## _bad / _x ## _loop, \
               _x ## _bad, _x ## _loop ); }

#define MPFR_ZIV_INIT(_x, _p)                                           \
  do                                                                    \
    {                                                                   \
      (_x) = GMP_NUMB_BITS;                                             \
      if (mpfr_log_level >= 0)                                          \
        _x ## _loop ++;                                                 \
      if ((MPFR_LOG_BADCASE_F & mpfr_log_type) &&                       \
          (mpfr_log_current <= mpfr_log_level))                         \
        LOG_PRINT ("%s:ZIV 1st prec=%Pd\n",                             \
                   __func__, (mpfr_prec_t) (_p));                       \
    }                                                                   \
  while (0)

#define MPFR_ZIV_NEXT(_x, _p)                                           \
  do                                                                    \
    {                                                                   \
      (_p) += (_x);                                                     \
      (_x) = (_p) / 2;                                                  \
      if (mpfr_log_level >= 0)                                          \
        _x ## _bad += (_x ## _cpt == 1);                                \
      _x ## _cpt ++;                                                    \
      if ((MPFR_LOG_BADCASE_F & mpfr_log_type) &&                       \
          (mpfr_log_current <= mpfr_log_level))                         \
        LOG_PRINT ("%s:ZIV new prec=%Pd\n",                             \
                   __func__, (mpfr_prec_t) (_p));                       \
    }                                                                   \
  while (0)

#define MPFR_ZIV_FREE(_x)                                               \
  do                                                                    \
    {                                                                   \
      if ((MPFR_LOG_BADCASE_F & mpfr_log_type) && _x ## _cpt > 1 &&     \
          (mpfr_log_current <= mpfr_log_level))                         \
        fprintf (mpfr_log_file, "%s:ZIV %d loops\n",                    \
                 __func__, _x ## _cpt);                                 \
    }                                                                   \
  while (0)

#endif


/******************************************************
 ***************  Logging Macros  *********************
 ******************************************************/

/* The different kind of LOG */
#define MPFR_LOG_INPUT_F    1
#define MPFR_LOG_OUTPUT_F   2
#define MPFR_LOG_INTERNAL_F 4
#define MPFR_LOG_TIME_F     8
#define MPFR_LOG_BADCASE_F  16
#define MPFR_LOG_MSG_F      32
#define MPFR_LOG_STAT_F     64

#ifdef MPFR_USE_LOGGING

/* Check if we can support this feature */
# ifdef MPFR_USE_THREAD_SAFE
#  error "Enable either `Logging' or `thread-safe', not both"
# endif
# if !__MPFR_GNUC(3,0)
#  error "Logging not supported (GCC >= 3.0)"
# endif

#if defined (__cplusplus)
extern "C" {
#endif

__MPFR_DECLSPEC extern FILE *mpfr_log_file;
__MPFR_DECLSPEC extern int   mpfr_log_type;
__MPFR_DECLSPEC extern int   mpfr_log_level;
__MPFR_DECLSPEC extern int   mpfr_log_current;
__MPFR_DECLSPEC extern mpfr_prec_t mpfr_log_prec;

#if defined (__cplusplus)
 }
#endif

/* LOG_PRINT calls mpfr_fprintf on mpfr_log_file with logging disabled
   (recursive logging is not wanted and freezes MPFR). */
#define LOG_PRINT(format, ...)                                          \
  do                                                                    \
    {                                                                   \
      int old_level = mpfr_log_level;                                   \
      mpfr_log_level = -1;  /* disable logging in mpfr_fprintf */       \
      __gmpfr_cache_const_pi = __gmpfr_logging_pi;                      \
      __gmpfr_cache_const_log2 = __gmpfr_logging_log2;                  \
      mpfr_fprintf (mpfr_log_file, format, __VA_ARGS__);                \
      mpfr_log_level = old_level;                                       \
      __gmpfr_cache_const_pi = __gmpfr_normal_pi;                       \
      __gmpfr_cache_const_log2 = __gmpfr_normal_log2;                   \
    }                                                                   \
  while (0)

#define MPFR_LOG_VAR(x)                                                 \
  do                                                                    \
    if ((MPFR_LOG_INTERNAL_F & mpfr_log_type) &&                        \
        (mpfr_log_current <= mpfr_log_level))                           \
      LOG_PRINT ("%s.%d:%s[%#Pu]=%.*Rg\n", __func__, __LINE__,          \
                 #x, mpfr_get_prec (x), mpfr_log_prec, x);              \
  while (0)

#define MPFR_LOG_MSG2(format, ...)                                      \
  do                                                                    \
    if ((MPFR_LOG_MSG_F & mpfr_log_type) &&                             \
        (mpfr_log_current <= mpfr_log_level))                           \
      LOG_PRINT ("%s.%d: "format, __func__, __LINE__, __VA_ARGS__);     \
  while (0)
#define MPFR_LOG_MSG(x) MPFR_LOG_MSG2 x

#define MPFR_LOG_BEGIN2(format, ...)                                    \
  mpfr_log_current ++;                                                  \
  if ((MPFR_LOG_INPUT_F & mpfr_log_type) &&                             \
      (mpfr_log_current <= mpfr_log_level))                             \
    LOG_PRINT ("%s:IN  "format"\n", __func__, __VA_ARGS__);             \
  if ((MPFR_LOG_TIME_F & mpfr_log_type) &&                              \
      (mpfr_log_current <= mpfr_log_level))                             \
    __gmpfr_log_time = mpfr_get_cputime ();
#define MPFR_LOG_BEGIN(x)                                               \
  int __gmpfr_log_time = 0;                                             \
  MPFR_LOG_BEGIN2 x

#define MPFR_LOG_END2(format, ...)                                      \
  if ((MPFR_LOG_TIME_F & mpfr_log_type) &&                              \
      (mpfr_log_current <= mpfr_log_level))                             \
    fprintf (mpfr_log_file, "%s:TIM %dms\n", __mpfr_log_fname,          \
             mpfr_get_cputime () - __gmpfr_log_time);                   \
  if ((MPFR_LOG_OUTPUT_F & mpfr_log_type) &&                            \
      (mpfr_log_current <= mpfr_log_level))                             \
    LOG_PRINT ("%s:OUT "format"\n", __mpfr_log_fname, __VA_ARGS__);     \
  mpfr_log_current --;
#define MPFR_LOG_END(x)                                                 \
  static const char *__mpfr_log_fname = __func__;                       \
  MPFR_LOG_END2 x

#define MPFR_LOG_FUNC(begin,end)                                        \
  static const char *__mpfr_log_fname = __func__;                       \
  auto void __mpfr_log_cleanup (int *time);                             \
  void __mpfr_log_cleanup (int *time) {                                 \
    int __gmpfr_log_time = *time;                                       \
    MPFR_LOG_END2 end; }                                                \
  int __gmpfr_log_time __attribute__ ((cleanup (__mpfr_log_cleanup)));  \
  __gmpfr_log_time = 0;                                                 \
  MPFR_LOG_BEGIN2 begin

#else /* MPFR_USE_LOGGING */

/* Define void macro for logging */

#define MPFR_LOG_VAR(x)
#define MPFR_LOG_BEGIN(x)
#define MPFR_LOG_END(x)
#define MPFR_LOG_MSG(x)
#define MPFR_LOG_FUNC(x,y)

#endif /* MPFR_USE_LOGGING */


/**************************************************************
 ************  Group Initialize Functions Macros  *************
 **************************************************************/

#ifndef MPFR_GROUP_STATIC_SIZE
# define MPFR_GROUP_STATIC_SIZE 16
#endif

struct mpfr_group_t {
  size_t     alloc;
  mp_limb_t *mant;
  mp_limb_t  tab[MPFR_GROUP_STATIC_SIZE];
};

#define MPFR_GROUP_DECL(g) struct mpfr_group_t g
#define MPFR_GROUP_CLEAR(g) do {                                 \
 MPFR_LOG_MSG (("GROUP_CLEAR: ptr = 0x%lX, size = %lu\n",        \
                (unsigned long) (g).mant,                        \
                (unsigned long) (g).alloc));                     \
 if (MPFR_UNLIKELY ((g).alloc != 0)) {                           \
   MPFR_ASSERTD ((g).mant != (g).tab);                           \
   (*__gmp_free_func) ((g).mant, (g).alloc);                     \
 }} while (0)

#define MPFR_GROUP_INIT_TEMPLATE(g, prec, num, handler) do {            \
 mpfr_prec_t _prec = (prec);                                            \
 mp_size_t _size;                                                       \
 MPFR_ASSERTD (_prec >= MPFR_PREC_MIN);                                 \
 if (MPFR_UNLIKELY (_prec > MPFR_PREC_MAX))                             \
   mpfr_abort_prec_max ();                                              \
 _size = MPFR_PREC2LIMBS (_prec);                                       \
 if (MPFR_UNLIKELY (_size * (num) > MPFR_GROUP_STATIC_SIZE))            \
   {                                                                    \
     (g).alloc = (num) * _size * sizeof (mp_limb_t);                    \
     (g).mant = (mp_limb_t *) (*__gmp_allocate_func) ((g).alloc);       \
   }                                                                    \
 else                                                                   \
   {                                                                    \
     (g).alloc = 0;                                                     \
     (g).mant = (g).tab;                                                \
   }                                                                    \
 MPFR_LOG_MSG (("GROUP_INIT: ptr = 0x%lX, size = %lu\n",                \
                (unsigned long) (g).mant, (unsigned long) (g).alloc));  \
 handler;                                                               \
 } while (0)
#define MPFR_GROUP_TINIT(g, n, x)                       \
  MPFR_TMP_INIT1 ((g).mant + _size * (n), x, _prec)

#define MPFR_GROUP_INIT_1(g, prec, x)                            \
 MPFR_GROUP_INIT_TEMPLATE(g, prec, 1, MPFR_GROUP_TINIT(g, 0, x))
#define MPFR_GROUP_INIT_2(g, prec, x, y)                         \
 MPFR_GROUP_INIT_TEMPLATE(g, prec, 2,                            \
   MPFR_GROUP_TINIT(g, 0, x);MPFR_GROUP_TINIT(g, 1, y))
#define MPFR_GROUP_INIT_3(g, prec, x, y, z)                      \
 MPFR_GROUP_INIT_TEMPLATE(g, prec, 3,                            \
   MPFR_GROUP_TINIT(g, 0, x);MPFR_GROUP_TINIT(g, 1, y);          \
   MPFR_GROUP_TINIT(g, 2, z))
#define MPFR_GROUP_INIT_4(g, prec, x, y, z, t)                   \
 MPFR_GROUP_INIT_TEMPLATE(g, prec, 4,                            \
   MPFR_GROUP_TINIT(g, 0, x);MPFR_GROUP_TINIT(g, 1, y);          \
   MPFR_GROUP_TINIT(g, 2, z);MPFR_GROUP_TINIT(g, 3, t))
#define MPFR_GROUP_INIT_5(g, prec, x, y, z, t, a)                \
 MPFR_GROUP_INIT_TEMPLATE(g, prec, 5,                            \
   MPFR_GROUP_TINIT(g, 0, x);MPFR_GROUP_TINIT(g, 1, y);          \
   MPFR_GROUP_TINIT(g, 2, z);MPFR_GROUP_TINIT(g, 3, t);          \
   MPFR_GROUP_TINIT(g, 4, a))
#define MPFR_GROUP_INIT_6(g, prec, x, y, z, t, a, b)             \
 MPFR_GROUP_INIT_TEMPLATE(g, prec, 6,                            \
   MPFR_GROUP_TINIT(g, 0, x);MPFR_GROUP_TINIT(g, 1, y);          \
   MPFR_GROUP_TINIT(g, 2, z);MPFR_GROUP_TINIT(g, 3, t);          \
   MPFR_GROUP_TINIT(g, 4, a);MPFR_GROUP_TINIT(g, 5, b))

#define MPFR_GROUP_REPREC_TEMPLATE(g, prec, num, handler) do {          \
 mpfr_prec_t _prec = (prec);                                            \
 size_t    _oalloc = (g).alloc;                                         \
 mp_size_t _size;                                                       \
 MPFR_LOG_MSG (("GROUP_REPREC: oldptr = 0x%lX, oldsize = %lu\n",        \
                (unsigned long) (g).mant, (unsigned long) _oalloc));    \
 MPFR_ASSERTD (_prec >= MPFR_PREC_MIN);                                 \
 if (MPFR_UNLIKELY (_prec > MPFR_PREC_MAX))                             \
   mpfr_abort_prec_max ();                                              \
 _size = MPFR_PREC2LIMBS (_prec);                                       \
 (g).alloc = (num) * _size * sizeof (mp_limb_t);                        \
 if (MPFR_LIKELY (_oalloc == 0))                                        \
   (g).mant = (mp_limb_t *) (*__gmp_allocate_func) ((g).alloc);         \
 else                                                                   \
   (g).mant = (mp_limb_t *)                                             \
     (*__gmp_reallocate_func) ((g).mant, _oalloc, (g).alloc);           \
 MPFR_LOG_MSG (("GROUP_REPREC: newptr = 0x%lX, newsize = %lu\n",        \
                (unsigned long) (g).mant, (unsigned long) (g).alloc));  \
 handler;                                                               \
 } while (0)

#define MPFR_GROUP_REPREC_1(g, prec, x)                          \
 MPFR_GROUP_REPREC_TEMPLATE(g, prec, 1, MPFR_GROUP_TINIT(g, 0, x))
#define MPFR_GROUP_REPREC_2(g, prec, x, y)                       \
 MPFR_GROUP_REPREC_TEMPLATE(g, prec, 2,                          \
   MPFR_GROUP_TINIT(g, 0, x);MPFR_GROUP_TINIT(g, 1, y))
#define MPFR_GROUP_REPREC_3(g, prec, x, y, z)                    \
 MPFR_GROUP_REPREC_TEMPLATE(g, prec, 3,                          \
   MPFR_GROUP_TINIT(g, 0, x);MPFR_GROUP_TINIT(g, 1, y);          \
   MPFR_GROUP_TINIT(g, 2, z))
#define MPFR_GROUP_REPREC_4(g, prec, x, y, z, t)                 \
 MPFR_GROUP_REPREC_TEMPLATE(g, prec, 4,                          \
   MPFR_GROUP_TINIT(g, 0, x);MPFR_GROUP_TINIT(g, 1, y);          \
   MPFR_GROUP_TINIT(g, 2, z);MPFR_GROUP_TINIT(g, 3, t))
#define MPFR_GROUP_REPREC_5(g, prec, x, y, z, t, a)              \
 MPFR_GROUP_REPREC_TEMPLATE(g, prec, 5,                          \
   MPFR_GROUP_TINIT(g, 0, x);MPFR_GROUP_TINIT(g, 1, y);          \
   MPFR_GROUP_TINIT(g, 2, z);MPFR_GROUP_TINIT(g, 3, t);          \
   MPFR_GROUP_TINIT(g, 4, a))
#define MPFR_GROUP_REPREC_6(g, prec, x, y, z, t, a, b)           \
 MPFR_GROUP_REPREC_TEMPLATE(g, prec, 6,                          \
   MPFR_GROUP_TINIT(g, 0, x);MPFR_GROUP_TINIT(g, 1, y);          \
   MPFR_GROUP_TINIT(g, 2, z);MPFR_GROUP_TINIT(g, 3, t);          \
   MPFR_GROUP_TINIT(g, 4, a);MPFR_GROUP_TINIT(g, 5, b))


/******************************************************
 ***************  Internal Functions  *****************
 ******************************************************/

#if defined (__cplusplus)
extern "C" {
#endif

__MPFR_DECLSPEC int mpfr_underflow _MPFR_PROTO ((mpfr_ptr, mpfr_rnd_t, int));
__MPFR_DECLSPEC int mpfr_overflow _MPFR_PROTO ((mpfr_ptr, mpfr_rnd_t, int));

__MPFR_DECLSPEC int mpfr_add1 _MPFR_PROTO ((mpfr_ptr, mpfr_srcptr,
                                            mpfr_srcptr, mpfr_rnd_t));
__MPFR_DECLSPEC int mpfr_sub1 _MPFR_PROTO ((mpfr_ptr, mpfr_srcptr,
                                            mpfr_srcptr, mpfr_rnd_t));
__MPFR_DECLSPEC int mpfr_add1sp _MPFR_PROTO ((mpfr_ptr, mpfr_srcptr,
                                              mpfr_srcptr, mpfr_rnd_t));
__MPFR_DECLSPEC int mpfr_sub1sp _MPFR_PROTO ((mpfr_ptr, mpfr_srcptr,
                                              mpfr_srcptr, mpfr_rnd_t));
__MPFR_DECLSPEC int mpfr_can_round_raw _MPFR_PROTO ((const mp_limb_t *,
             mp_size_t, int, mpfr_exp_t, mpfr_rnd_t, mpfr_rnd_t, mpfr_prec_t));

__MPFR_DECLSPEC int mpfr_cmp2 _MPFR_PROTO ((mpfr_srcptr, mpfr_srcptr,
                                            mpfr_prec_t *));

__MPFR_DECLSPEC long          __gmpfr_ceil_log2     _MPFR_PROTO ((double));
__MPFR_DECLSPEC long          __gmpfr_floor_log2    _MPFR_PROTO ((double));
__MPFR_DECLSPEC double        __gmpfr_ceil_exp2     _MPFR_PROTO ((double));
__MPFR_DECLSPEC unsigned long __gmpfr_isqrt     _MPFR_PROTO ((unsigned long));
__MPFR_DECLSPEC unsigned long __gmpfr_cuberoot  _MPFR_PROTO ((unsigned long));
__MPFR_DECLSPEC int       __gmpfr_int_ceil_log2 _MPFR_PROTO ((unsigned long));

__MPFR_DECLSPEC mpfr_exp_t mpfr_ceil_mul _MPFR_PROTO ((mpfr_exp_t, int, int));

__MPFR_DECLSPEC int mpfr_exp_2 _MPFR_PROTO ((mpfr_ptr, mpfr_srcptr,mpfr_rnd_t));
__MPFR_DECLSPEC int mpfr_exp_3 _MPFR_PROTO ((mpfr_ptr, mpfr_srcptr,mpfr_rnd_t));
__MPFR_DECLSPEC int mpfr_powerof2_raw _MPFR_PROTO ((mpfr_srcptr));
__MPFR_DECLSPEC int mpfr_powerof2_raw2 (const mp_limb_t *, mp_size_t);

__MPFR_DECLSPEC int mpfr_pow_general _MPFR_PROTO ((mpfr_ptr, mpfr_srcptr,
                           mpfr_srcptr, mpfr_rnd_t, int, mpfr_save_expo_t *));

__MPFR_DECLSPEC void mpfr_setmax _MPFR_PROTO ((mpfr_ptr, mpfr_exp_t));
__MPFR_DECLSPEC void mpfr_setmin _MPFR_PROTO ((mpfr_ptr, mpfr_exp_t));

__MPFR_DECLSPEC long mpfr_mpn_exp _MPFR_PROTO ((mp_limb_t *, mpfr_exp_t *, int,
                                                mpfr_exp_t, size_t));

#ifdef _MPFR_H_HAVE_FILE
__MPFR_DECLSPEC void mpfr_fprint_binary _MPFR_PROTO ((FILE *, mpfr_srcptr));
#endif
__MPFR_DECLSPEC void mpfr_print_binary _MPFR_PROTO ((mpfr_srcptr));
__MPFR_DECLSPEC void mpfr_print_mant_binary _MPFR_PROTO ((const char*,
                                          const mp_limb_t*, mpfr_prec_t));
__MPFR_DECLSPEC void mpfr_set_str_binary _MPFR_PROTO((mpfr_ptr, const char*));

__MPFR_DECLSPEC int mpfr_round_raw _MPFR_PROTO ((mp_limb_t *,
       const mp_limb_t *, mpfr_prec_t, int, mpfr_prec_t, mpfr_rnd_t, int *));
__MPFR_DECLSPEC int mpfr_round_raw_2 _MPFR_PROTO ((const mp_limb_t *,
             mpfr_prec_t, int, mpfr_prec_t, mpfr_rnd_t));
/* No longer defined (see round_prec.c).
   Uncomment if it needs to be defined again.
__MPFR_DECLSPEC int mpfr_round_raw_3 _MPFR_PROTO ((const mp_limb_t *,
             mpfr_prec_t, int, mpfr_prec_t, mpfr_rnd_t, int *));
*/
__MPFR_DECLSPEC int mpfr_round_raw_4 _MPFR_PROTO ((mp_limb_t *,
       const mp_limb_t *, mpfr_prec_t, int, mpfr_prec_t, mpfr_rnd_t));

#define mpfr_round_raw2(xp, xn, neg, r, prec) \
  mpfr_round_raw_2((xp),(xn)*GMP_NUMB_BITS,(neg),(prec),(r))

__MPFR_DECLSPEC int mpfr_check _MPFR_PROTO ((mpfr_srcptr));

__MPFR_DECLSPEC int mpfr_sum_sort _MPFR_PROTO ((mpfr_srcptr *const,
                                                unsigned long, mpfr_srcptr *,
                                                mpfr_prec_t *));

__MPFR_DECLSPEC int mpfr_get_cputime _MPFR_PROTO ((void));

__MPFR_DECLSPEC void mpfr_nexttozero _MPFR_PROTO ((mpfr_ptr));
__MPFR_DECLSPEC void mpfr_nexttoinf _MPFR_PROTO ((mpfr_ptr));

__MPFR_DECLSPEC int mpfr_const_pi_internal _MPFR_PROTO ((mpfr_ptr,mpfr_rnd_t));
__MPFR_DECLSPEC int mpfr_const_log2_internal _MPFR_PROTO((mpfr_ptr,mpfr_rnd_t));
__MPFR_DECLSPEC int mpfr_const_euler_internal _MPFR_PROTO((mpfr_ptr, mpfr_rnd_t));
__MPFR_DECLSPEC int mpfr_const_catalan_internal _MPFR_PROTO((mpfr_ptr, mpfr_rnd_t));

#if 0
__MPFR_DECLSPEC void mpfr_init_cache _MPFR_PROTO ((mpfr_cache_t,
                                           int(*)(mpfr_ptr,mpfr_rnd_t)));
#endif
__MPFR_DECLSPEC void mpfr_clear_cache _MPFR_PROTO ((mpfr_cache_t));
__MPFR_DECLSPEC int  mpfr_cache _MPFR_PROTO ((mpfr_ptr, mpfr_cache_t,
                                              mpfr_rnd_t));

__MPFR_DECLSPEC void mpfr_mulhigh_n _MPFR_PROTO ((mpfr_limb_ptr,
                        mpfr_limb_srcptr, mpfr_limb_srcptr, mp_size_t));
__MPFR_DECLSPEC void mpfr_mullow_n  _MPFR_PROTO ((mpfr_limb_ptr,
                        mpfr_limb_srcptr, mpfr_limb_srcptr, mp_size_t));
__MPFR_DECLSPEC void mpfr_sqrhigh_n _MPFR_PROTO ((mpfr_limb_ptr,
                        mpfr_limb_srcptr, mp_size_t));
__MPFR_DECLSPEC mp_limb_t mpfr_divhigh_n _MPFR_PROTO ((mpfr_limb_ptr,
                        mpfr_limb_ptr, mpfr_limb_ptr, mp_size_t));

__MPFR_DECLSPEC int mpfr_round_p _MPFR_PROTO ((mp_limb_t *, mp_size_t,
                                               mpfr_exp_t, mpfr_prec_t));

__MPFR_DECLSPEC void mpfr_dump_mant _MPFR_PROTO ((const mp_limb_t *,
                                                  mpfr_prec_t, mpfr_prec_t,
                                                  mpfr_prec_t));

__MPFR_DECLSPEC int mpfr_round_near_x _MPFR_PROTO ((mpfr_ptr, mpfr_srcptr,
                                                    mpfr_uexp_t, int,
                                                    mpfr_rnd_t));
__MPFR_DECLSPEC void mpfr_abort_prec_max _MPFR_PROTO ((void))
       MPFR_NORETURN_ATTR;

__MPFR_DECLSPEC void mpfr_rand_raw _MPFR_PROTO((mpfr_limb_ptr, gmp_randstate_t,
                                                mpfr_prec_t));

__MPFR_DECLSPEC mpz_t* mpfr_bernoulli_internal _MPFR_PROTO((mpz_t*,
                                                            unsigned long));

__MPFR_DECLSPEC int mpfr_sincos_fast _MPFR_PROTO((mpfr_t, mpfr_t,
                                                  mpfr_srcptr, mpfr_rnd_t));

__MPFR_DECLSPEC double mpfr_scale2 _MPFR_PROTO((double, int));

__MPFR_DECLSPEC void mpfr_div_ui2 _MPFR_PROTO((mpfr_ptr, mpfr_srcptr,
                                               unsigned long int, unsigned long int,
                                               mpfr_rnd_t));

__MPFR_DECLSPEC void mpfr_gamma_one_and_two_third _MPFR_PROTO((mpfr_ptr, mpfr_ptr, mpfr_prec_t));

#if defined (__cplusplus)
}
#endif

#endif
