/* Interface to replace gmp-impl.h

Copyright 2004-2017 Free Software Foundation, Inc.
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

#ifndef __GMPFR_GMP_H__
#define __GMPFR_GMP_H__

#ifndef __MPFR_IMPL_H__
# error  "mpfr-impl.h not included"
#endif

#include <limits.h> /* For INT_MAX, ... */
#include <string.h> /* For memcpy, memset and memmove */

/* The following tries to get a good version of alloca.
   See gmp-impl.h for implementation details and original version */
/* FIXME: the autoconf manual gives a different piece of code under the
   documentation of the AC_FUNC_ALLOCA macro. Should we switch to it? */
#ifndef alloca
# if defined ( __GNUC__ )
#  define alloca __builtin_alloca
# elif defined (__DECC)
#  define alloca(x) __ALLOCA(x)
# elif defined (_MSC_VER)
#  include <malloc.h>
#  define alloca _alloca
# elif defined (HAVE_ALLOCA_H)
#  include <alloca.h>
# elif defined (_AIX) || defined (_IBMR2)
#  pragma alloca
# else
void *alloca (size_t);
# endif
#endif

#if defined (__cplusplus)
extern "C" {
#endif

/* Define GMP_NUMB_BITS
   Can't use sizeof(mp_limb_t) since it should be a preprocessor constant */
#if defined(GMP_NUMB_BITS) /* GMP 4.1.2 or above */
#ifndef GMP_NUMB_BITS
# define GMP_NUMB_BITS  (GMP_NUMB_BITS+GMP_NAIL_BITS)
#endif
#elif defined (__GMP_GMP_NUMB_BITS) /* Older versions 4.x.x */
# define GMP_NUMB_BITS  __GMP_GMP_NUMB_BITS
# define GMP_NUMB_BITS GMP_NUMB_BITS
# ifndef GMP_NAIL_BITS
#  define GMP_NAIL_BITS 0
# endif
#else
# error "Could not detect GMP_NUMB_BITS. Try with gmp internal files."
#endif

/* Define some macros */

#define MP_LIMB_T_MAX (~(mp_limb_t)0)

#define ULONG_HIGHBIT (ULONG_MAX ^ ((unsigned long) ULONG_MAX >> 1))
#define UINT_HIGHBIT  (UINT_MAX ^ ((unsigned) UINT_MAX >> 1))
#define USHRT_HIGHBIT ((unsigned short) (USHRT_MAX ^ ((unsigned short) USHRT_MAX >> 1)))

#define GMP_LIMB_HIGHBIT (MP_LIMB_T_MAX ^ (MP_LIMB_T_MAX >> 1))


#if __GMP_MP_SIZE_T_INT
#define MP_SIZE_T_MAX      INT_MAX
#define MP_SIZE_T_MIN      INT_MIN
#else
#define MP_SIZE_T_MAX      LONG_MAX
#define MP_SIZE_T_MIN      LONG_MIN
#endif

#define LONG_HIGHBIT       LONG_MIN
#define INT_HIGHBIT        INT_MIN
#define SHRT_HIGHBIT       SHRT_MIN

/* MP_LIMB macros */
#define MPN_ZERO(dst, n) memset((dst), 0, (n)*MPFR_BYTES_PER_MP_LIMB)
#define MPN_COPY_DECR(dst,src,n) memmove((dst),(src),(n)*MPFR_BYTES_PER_MP_LIMB)
#define MPN_COPY_INCR(dst,src,n) memmove((dst),(src),(n)*MPFR_BYTES_PER_MP_LIMB)
#define MPN_COPY(dst,src,n) \
  do                                                                  \
    {                                                                 \
      if ((dst) != (src))                                             \
        {                                                             \
          MPFR_ASSERTD ((char *) (dst) >= (char *) (src) +            \
                                     (n) * MPFR_BYTES_PER_MP_LIMB ||  \
                        (char *) (src) >= (char *) (dst) +            \
                                     (n) * MPFR_BYTES_PER_MP_LIMB);   \
          memcpy ((dst), (src), (n) * MPFR_BYTES_PER_MP_LIMB);        \
        }                                                             \
    }                                                                 \
  while (0)

/* MPN macros taken from gmp-impl.h */
#define MPN_NORMALIZE(DST, NLIMBS) \
  do {                                        \
    while (NLIMBS > 0)                        \
      {                                       \
        if ((DST)[(NLIMBS) - 1] != 0)         \
          break;                              \
        NLIMBS--;                             \
      }                                       \
  } while (0)
#define MPN_NORMALIZE_NOT_ZERO(DST, NLIMBS)     \
  do {                                          \
    MPFR_ASSERTD ((NLIMBS) >= 1);               \
    while (1)                                   \
      {                                         \
        if ((DST)[(NLIMBS) - 1] != 0)           \
          break;                                \
        NLIMBS--;                               \
      }                                         \
  } while (0)
#define MPN_OVERLAP_P(xp, xsize, yp, ysize) \
  ((xp) + (xsize) > (yp) && (yp) + (ysize) > (xp))
#define MPN_SAME_OR_INCR2_P(dst, dsize, src, ssize)             \
  ((dst) <= (src) || ! MPN_OVERLAP_P (dst, dsize, src, ssize))
#define MPN_SAME_OR_INCR_P(dst, src, size)      \
  MPN_SAME_OR_INCR2_P(dst, size, src, size)
#define MPN_SAME_OR_DECR2_P(dst, dsize, src, ssize)             \
  ((dst) >= (src) || ! MPN_OVERLAP_P (dst, dsize, src, ssize))
#define MPN_SAME_OR_DECR_P(dst, src, size)      \
  MPN_SAME_OR_DECR2_P(dst, size, src, size)

/* If mul_basecase or mpn_sqr_basecase are not exported, used mpn_mul instead */
#ifndef mpn_mul_basecase
# define mpn_mul_basecase(dst,s1,n1,s2,n2) mpn_mul((dst),(s1),(n1),(s2),(n2))
#endif
#ifndef mpn_sqr_basecase
# define mpn_sqr_basecase(dst,src,n) mpn_mul((dst),(src),(n),(src),(n))
#endif

/* ASSERT */
__MPFR_DECLSPEC void mpfr_assert_fail _MPFR_PROTO((const char *, int,
                                                   const char *));

#define ASSERT_FAIL(expr)  mpfr_assert_fail (__FILE__, __LINE__, #expr)
#define ASSERT(expr)       MPFR_ASSERTD(expr)

/* Access fields of GMP struct */
#define SIZ(x) ((x)->_mp_size)
#define ABSIZ(x) ABS (SIZ (x))
#define PTR(x) ((x)->_mp_d)
#define EXP(x) ((x)->_mp_exp)
#define PREC(x) ((x)->_mp_prec)
#define ALLOC(x) ((x)->_mp_alloc)
#define MPZ_REALLOC(z,n) ((n) > ALLOC(z) ? _mpz_realloc(z,n) : PTR(z))

/* Non IEEE float supports -- needs to detect them with proper configure */
#undef  XDEBUG
#define XDEBUG

/* For longlong.h */
#ifdef HAVE_ATTRIBUTE_MODE
typedef unsigned int UQItype    __attribute__ ((mode (QI)));
typedef          int SItype     __attribute__ ((mode (SI)));
typedef unsigned int USItype    __attribute__ ((mode (SI)));
typedef          int DItype     __attribute__ ((mode (DI)));
typedef unsigned int UDItype    __attribute__ ((mode (DI)));
#else
typedef unsigned char UQItype;
typedef          long SItype;
typedef unsigned long USItype;
#ifdef HAVE_LONG_LONG
typedef long long int DItype;
typedef unsigned long long int UDItype;
#else /* Assume `long' gives us a wide enough type.  Needed for hppa2.0w.  */
typedef long int DItype;
typedef unsigned long int UDItype;
#endif
#endif
typedef mp_limb_t UWtype;
typedef unsigned int UHWtype;
#define W_TYPE_SIZE GMP_NUMB_BITS

/* Remap names of internal mpn functions (for longlong.h).  */
#undef  __clz_tab
#define __clz_tab               mpfr_clz_tab

/* Use (4.0 * ...) instead of (2.0 * ...) to work around buggy compilers
   that don't convert ulong->double correctly (eg. SunOS 4 native cc).  */
#undef MP_BASE_AS_DOUBLE
#define MP_BASE_AS_DOUBLE (4.0 * ((mp_limb_t) 1 << (GMP_NUMB_BITS - 2)))

/* Structure for conversion between internal binary format and
   strings in base 2..36.  */
struct bases
{
  /* log(2)/log(conversion_base) */
  double chars_per_bit_exactly;
};
#undef  __mp_bases
#define __mp_bases mpfr_bases
__MPFR_DECLSPEC extern const struct bases mpfr_bases[257];

/* Standard macros */
#undef ABS
#undef MIN
#undef MAX
#undef numberof
#define ABS(x) ((x) >= 0 ? (x) : -(x))
#define MIN(l,o) ((l) < (o) ? (l) : (o))
#define MAX(h,i) ((h) > (i) ? (h) : (i))
#define numberof(x)  (sizeof (x) / sizeof ((x)[0]))

/* Random */
#undef  __gmp_rands_initialized
#undef  __gmp_rands
#define __gmp_rands_initialized mpfr_rands_initialized
#define __gmp_rands             mpfr_rands

__MPFR_DECLSPEC extern char             mpfr_rands_initialized;
__MPFR_DECLSPEC extern gmp_randstate_t  mpfr_rands;

#undef RANDS
#define RANDS                                   \
  ((__gmp_rands_initialized ? 0                 \
    : (__gmp_rands_initialized = 1,             \
       gmp_randinit_default (__gmp_rands), 0)), \
   __gmp_rands)

#undef RANDS_CLEAR
#define RANDS_CLEAR()                   \
  do {                                  \
    if (__gmp_rands_initialized)        \
      {                                 \
        __gmp_rands_initialized = 0;    \
        gmp_randclear (__gmp_rands);    \
      }                                 \
  } while (0)

/* Allocate func are defined in gmp-impl.h */

/* In newer GMP, there aren't anymore __gmp_allocate_func,
   __gmp_reallocate_func & __gmp_free_func in gmp.h
   Just getting the correct value by calling mp_get_memory_functions */
#ifdef mp_get_memory_functions

#undef __gmp_allocate_func
#undef __gmp_reallocate_func
#undef __gmp_free_func
#define MPFR_GET_MEMFUNC                                        \
  ((void) (MPFR_LIKELY (mpfr_allocate_func != 0) ||             \
           (mp_get_memory_functions(&mpfr_allocate_func,        \
                                    &mpfr_reallocate_func,      \
                                    &mpfr_free_func), 1)))
#define __gmp_allocate_func   (MPFR_GET_MEMFUNC, mpfr_allocate_func)
#define __gmp_reallocate_func (MPFR_GET_MEMFUNC, mpfr_reallocate_func)
#define __gmp_free_func       (MPFR_GET_MEMFUNC, mpfr_free_func)
__MPFR_DECLSPEC extern MPFR_THREAD_ATTR void * (*mpfr_allocate_func)   _MPFR_PROTO ((size_t));
__MPFR_DECLSPEC extern MPFR_THREAD_ATTR void * (*mpfr_reallocate_func) _MPFR_PROTO ((void *, size_t, size_t));
__MPFR_DECLSPEC extern MPFR_THREAD_ATTR void   (*mpfr_free_func)       _MPFR_PROTO ((void *, size_t));

#endif

#undef __gmp_default_allocate
#undef __gmp_default_reallocate
#undef __gmp_default_free
#define __gmp_default_allocate   mpfr_default_allocate
#define __gmp_default_reallocate mpfr_default_reallocate
#define __gmp_default_free       mpfr_default_free
__MPFR_DECLSPEC void *__gmp_default_allocate _MPFR_PROTO ((size_t));
__MPFR_DECLSPEC void *__gmp_default_reallocate _MPFR_PROTO ((void *, size_t,
                                                             size_t));
__MPFR_DECLSPEC void __gmp_default_free _MPFR_PROTO ((void *, size_t));

#if defined(WANT_GMP_INTERNALS) && defined(HAVE___GMPN_ROOTREM)
#ifndef __gmpn_rootrem
  __MPFR_DECLSPEC mp_size_t __gmpn_rootrem _MPFR_PROTO ((mp_limb_t*,
                    mp_limb_t*, mp_limb_t*, mp_size_t, mp_limb_t));
#endif
#endif

#if defined(WANT_GMP_INTERNALS) && defined(HAVE___GMPN_SBPI1_DIVAPPR_Q)
#ifndef __gmpn_sbpi1_divappr_q
  __MPFR_DECLSPEC mp_limb_t __gmpn_sbpi1_divappr_q _MPFR_PROTO ((mp_limb_t*,
                mp_limb_t*, mp_size_t, mp_limb_t*, mp_size_t, mp_limb_t));
#endif
#endif

/* Temp memory allocate */

struct tmp_marker
{
  void *ptr;
  size_t size;
  struct tmp_marker *next;
};

__MPFR_DECLSPEC void *mpfr_tmp_allocate _MPFR_PROTO ((struct tmp_marker **,
                                                      size_t));
__MPFR_DECLSPEC void mpfr_tmp_free _MPFR_PROTO ((struct tmp_marker *));

/* Do not define TMP_SALLOC (see the test in mpfr-impl.h)! */
#define TMP_ALLOC(n) (MPFR_LIKELY ((n) < 16384) ?       \
                      alloca (n) : mpfr_tmp_allocate (&tmp_marker, (n)))
#define TMP_DECL(m) struct tmp_marker *tmp_marker
#define TMP_MARK(m) (tmp_marker = 0)
#define TMP_FREE(m) mpfr_tmp_free (tmp_marker)

/* invert_limb macro, copied from GMP 5.0.2, file gmp-impl.h.
   It returns invxl = floor((B^2-1)/xl)-B, where B=2^BITS_PER_LIMB,
   assuming the most significant bit of xl is set. */
#undef invert_limb
#define invert_limb(invxl,xl)                             \
  do {                                                    \
    mp_limb_t dummy;                                      \
    MPFR_ASSERTD ((xl) != 0);                             \
    udiv_qrnnd (invxl, dummy, ~(xl), ~(mp_limb_t)0, xl);  \
  } while (0)

typedef struct {mp_limb_t inv32;} mpfr_pi1_t; /* We changed gmp_pi1_t into
                                                 mpfr_pi1_t to avoid using
                                                 GMP's namespace. */
/* invert_pi1 macro, adapted from GMP 5.0.2, file gmp-impl.h.
   It returns dinv = floor((B^3-1)/(d1*B+d0))-B, where B=2^BITS_PER_LIMB,
   assuming the most significant bit of d1 is set. */
#undef invert_pi1
#define invert_pi1(dinv, d1, d0)                                \
  do {                                                          \
    mp_limb_t _v, _p, _t1, _t0, _mask;                          \
    invert_limb (_v, d1);                                       \
    _p = d1 * _v;                                               \
    _p += d0;                                                   \
    if (_p < d0)                                                \
      {                                                         \
        _v--;                                                   \
        _mask = -(_p >= d1);                                    \
        _p -= d1;                                               \
        _v += _mask;                                            \
        _p -= _mask & d1;                                       \
      }                                                         \
    umul_ppmm (_t1, _t0, d0, _v);                               \
    _p += _t1;                                                  \
    if (_p < _t1)                                               \
      {                                                         \
        _v--;                                                   \
        if (MPFR_UNLIKELY (_p >= d1))                           \
          {                                                     \
            if (_p > d1 || _t0 >= d0)                           \
              _v--;                                             \
          }                                                     \
      }                                                         \
    (dinv).inv32 = _v;                                          \
  } while (0)

/* udiv_qr_3by2 macro, adapted from GMP 5.0.2, file gmp-impl.h.
   Compute quotient the quotient and remainder for n / d. Requires d
   >= B^2 / 2 and n < d B. dinv is the inverse

     floor ((B^3 - 1) / (d0 + d1 B)) - B.

   NOTE: Output variables are updated multiple times. Only some inputs
   and outputs may overlap.
*/
#undef udiv_qr_3by2
#define udiv_qr_3by2(q, r1, r0, n2, n1, n0, d1, d0, dinv)               \
  do {                                                                  \
    mp_limb_t _q0, _t1, _t0, _mask;                                     \
    umul_ppmm ((q), _q0, (n2), (dinv));                                 \
    add_ssaaaa ((q), _q0, (q), _q0, (n2), (n1));                        \
                                                                        \
    /* Compute the two most significant limbs of n - q'd */             \
    (r1) = (n1) - (d1) * (q);                                           \
    (r0) = (n0);                                                        \
    sub_ddmmss ((r1), (r0), (r1), (r0), (d1), (d0));                    \
    umul_ppmm (_t1, _t0, (d0), (q));                                    \
    sub_ddmmss ((r1), (r0), (r1), (r0), _t1, _t0);                      \
    (q)++;                                                              \
                                                                        \
    /* Conditionally adjust q and the remainders */                     \
    _mask = - (mp_limb_t) ((r1) >= _q0);                                \
    (q) += _mask;                                                       \
    add_ssaaaa ((r1), (r0), (r1), (r0), _mask & (d1), _mask & (d0));    \
    if (MPFR_UNLIKELY ((r1) >= (d1)))                                   \
      {                                                                 \
        if ((r1) > (d1) || (r0) >= (d0))                                \
          {                                                             \
            (q)++;                                                      \
            sub_ddmmss ((r1), (r0), (r1), (r0), (d1), (d0));            \
          }                                                             \
      }                                                                 \
  } while (0)

#if defined (__cplusplus)
}
#endif

#endif /* Gmp internal emulator */
