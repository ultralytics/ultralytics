/* auxiliary functions for MPFR tests.

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

#ifndef __MPFR_TEST_H__
#define __MPFR_TEST_H__

#include <stdio.h>

#include "mpfr-impl.h"

/* generates a random long int, a random double,
   and corresponding seed initializing */
#define DBL_RAND() ((double) randlimb() / (double) MP_LIMB_T_MAX)

#define MINNORM 2.2250738585072013831e-308 /* 2^(-1022), smallest normalized */
#define MAXNORM 1.7976931348623157081e308 /* 2^(1023)*(2-2^(-52)) */

/* Generates a random rounding mode */
#define RND_RAND() ((mpfr_rnd_t) (randlimb() % MPFR_RND_MAX))

/* Generates a random sign */
#define SIGN_RAND() ( (randlimb()%2) ? MPFR_SIGN_POS : MPFR_SIGN_NEG)

/* Loop for all rounding modes */
#define RND_LOOP(_r) for((_r) = 0 ; (_r) < MPFR_RND_MAX ; (_r)++)

/* Test whether two floating-point data have the same value,
   seen as an element of the set of the floating-point data
   (Level 2 in the IEEE 754-2008 standard). */
#define SAME_VAL(X,Y)                                                   \
  ((MPFR_IS_NAN (X) && MPFR_IS_NAN (Y)) ||                              \
   (mpfr_equal_p ((X), (Y)) && MPFR_INT_SIGN (X) == MPFR_INT_SIGN (Y)))

/* The MAX, MIN and ABS macros may already be defined if gmp-impl.h has
   been included. They have the same semantics as in gmp-impl.h, but the
   expressions may be slightly different. So, it's better to undefine
   them first, as required by the ISO C standard. */
#undef MAX
#undef MIN
#undef ABS
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define ABS(x) (((x)>0) ? (x) : -(x))

#define FLIST mpfr_ptr, mpfr_srcptr, mpfr_rnd_t

#if defined (__cplusplus)
extern "C" {
#endif

int test_version _MPFR_PROTO ((void));

void tests_memory_start _MPFR_PROTO ((void));
void tests_memory_end _MPFR_PROTO ((void));

void tests_start_mpfr _MPFR_PROTO ((void));
void tests_end_mpfr _MPFR_PROTO ((void));

int mpfr_set_machine_rnd_mode _MPFR_PROTO ((mpfr_rnd_t));
void mpfr_test_init _MPFR_PROTO ((void));
mp_limb_t randlimb _MPFR_PROTO ((void));
void randseed _MPFR_PROTO ((unsigned int));
void mpfr_random2 _MPFR_PROTO ((mpfr_ptr, mp_size_t, mpfr_exp_t, gmp_randstate_t));
int ulp _MPFR_PROTO ((double, double));
double dbl _MPFR_PROTO ((double, int));
double Ulp _MPFR_PROTO ((double));
int Isnan _MPFR_PROTO ((double));
void d_trace _MPFR_PROTO ((const char *, double));
void ld_trace _MPFR_PROTO ((const char *, long double));

FILE *src_fopen _MPFR_PROTO ((const char *, const char *));
void set_emin _MPFR_PROTO ((mpfr_exp_t));
void set_emax _MPFR_PROTO ((mpfr_exp_t));
void tests_default_random _MPFR_PROTO ((mpfr_ptr, int, mpfr_exp_t, mpfr_exp_t,
                                        int));
void data_check _MPFR_PROTO ((const char *, int (*) (FLIST), const char *));
void bad_cases _MPFR_PROTO ((int (*)(FLIST), int (*)(FLIST),
                             const char *, int, mpfr_exp_t, mpfr_exp_t,
                             mpfr_prec_t, mpfr_prec_t, mpfr_prec_t, int));
void flags_out _MPFR_PROTO ((unsigned int));

int mpfr_cmp_str _MPFR_PROTO ((mpfr_srcptr x, const char *, int, mpfr_rnd_t));
#define mpfr_cmp_str1(x,s) mpfr_cmp_str(x,s,10,MPFR_RNDN)
#define mpfr_set_str1(x,s) mpfr_set_str(x,s,10,MPFR_RNDN)

#define mpfr_cmp0(x,y) (MPFR_ASSERTN (!MPFR_IS_NAN (x) && !MPFR_IS_NAN (y)), mpfr_cmp (x,y))
#define mpfr_cmp_ui0(x,i) (MPFR_ASSERTN (!MPFR_IS_NAN (x)), mpfr_cmp_ui (x,i))

/* Allocation */
void *tests_allocate _MPFR_PROTO ((size_t));
void *tests_reallocate _MPFR_PROTO ((void *, size_t, size_t));
void tests_free _MPFR_PROTO ((void *, size_t));

#if defined (__cplusplus)
}
#endif

/* define CHECK_EXTERNAL if you want to check mpfr against another library
   with correct rounding. You'll probably have to modify mpfr_print_raw()
   and/or test_add() below:
   * mpfr_print_raw() prints each number as "p m e" where p is the precision,
     m the mantissa (as a binary integer with sign), and e the exponent.
     The corresponding number is m*2^e. Example: "2 10 -6" represents
     2*2^(-6) with a precision of 2 bits.
   * test_add() outputs "b c a" on one line, for each addition a <- b + c.
     Currently it only prints such a line for rounding to nearest, when
     the inputs b and c are not NaN and/or Inf.
*/
#ifdef CHECK_EXTERNAL
static void
mpfr_print_raw (mpfr_srcptr x)
{
  printf ("%lu ", MPFR_PREC (x));
  if (MPFR_IS_NAN (x))
    {
      printf ("@NaN@");
      return;
    }

  if (MPFR_SIGN (x) < 0)
    printf ("-");

  if (MPFR_IS_INF (x))
    printf ("@Inf@");
  else if (MPFR_IS_ZERO (x))
    printf ("0 0");
  else
    {
      mp_limb_t *mx;
      mpfr_prec_t px;
      mp_size_t n;

      mx = MPFR_MANT (x);
      px = MPFR_PREC (x);

      for (n = (px - 1) / GMP_NUMB_BITS; ; n--)
        {
          mp_limb_t wd, t;

          MPFR_ASSERTN (n >= 0);
          wd = mx[n];
          for (t = MPFR_LIMB_HIGHBIT; t != 0; t >>= 1)
            {
              printf ((wd & t) == 0 ? "0" : "1");
              if (--px == 0)
                {
                  mpfr_exp_t ex;

                  ex = MPFR_GET_EXP (x);
                  MPFR_ASSERTN (ex >= LONG_MIN && ex <= LONG_MAX);
                  printf (" %ld", (long) ex - (long) MPFR_PREC (x));
                  return;
                }
            }
        }
    }
}
#endif

#endif
