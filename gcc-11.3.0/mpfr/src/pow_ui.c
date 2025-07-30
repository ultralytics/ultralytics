/* mpfr_pow_ui-- compute the power of a floating-point
                                  by a machine integer

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

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

/* sets y to x^n, and return 0 if exact, non-zero otherwise */
int
mpfr_pow_ui (mpfr_ptr y, mpfr_srcptr x, unsigned long int n, mpfr_rnd_t rnd)
{
  unsigned long m;
  mpfr_t res;
  mpfr_prec_t prec, err;
  int inexact;
  mpfr_rnd_t rnd1;
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_ZIV_DECL (loop);
  MPFR_BLOCK_DECL (flags);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg n=%lu rnd=%d",
      mpfr_get_prec (x), mpfr_log_prec, x, n, rnd),
     ("y[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (y), mpfr_log_prec, y, inexact));

  /* x^0 = 1 for any x, even a NaN */
  if (MPFR_UNLIKELY (n == 0))
    return mpfr_set_ui (y, 1, rnd);

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (x))
        {
          /* Inf^n = Inf, (-Inf)^n = Inf for n even, -Inf for n odd */
          if (MPFR_IS_NEG (x) && (n & 1) == 1)
            MPFR_SET_NEG (y);
          else
            MPFR_SET_POS (y);
          MPFR_SET_INF (y);
          MPFR_RET (0);
        }
      else /* x is zero */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (x));
          /* 0^n = 0 for any n */
          MPFR_SET_ZERO (y);
          if (MPFR_IS_POS (x) || (n & 1) == 0)
            MPFR_SET_POS (y);
          else
            MPFR_SET_NEG (y);
          MPFR_RET (0);
        }
    }
  else if (MPFR_UNLIKELY (n <= 2))
    {
      if (n < 2)
        /* x^1 = x */
        return mpfr_set (y, x, rnd);
      else
        /* x^2 = sqr(x) */
        return mpfr_sqr (y, x, rnd);
    }

  /* Augment exponent range */
  MPFR_SAVE_EXPO_MARK (expo);

  /* setup initial precision */
  prec = MPFR_PREC (y) + 3 + GMP_NUMB_BITS
    + MPFR_INT_CEIL_LOG2 (MPFR_PREC (y));
  mpfr_init2 (res, prec);

  rnd1 = MPFR_IS_POS (x) ? MPFR_RNDU : MPFR_RNDD; /* away */

  MPFR_ZIV_INIT (loop, prec);
  for (;;)
    {
      int i;

      for (m = n, i = 0; m; i++, m >>= 1)
        ;
      /* now 2^(i-1) <= n < 2^i */
      MPFR_ASSERTD (prec > (mpfr_prec_t) i);
      err = prec - 1 - (mpfr_prec_t) i;
      /* First step: compute square from x */
      MPFR_BLOCK (flags,
                  inexact = mpfr_mul (res, x, x, MPFR_RNDU);
                  MPFR_ASSERTD (i >= 2);
                  if (n & (1UL << (i-2)))
                    inexact |= mpfr_mul (res, res, x, rnd1);
                  for (i -= 3; i >= 0 && !MPFR_BLOCK_EXCEP; i--)
                    {
                      inexact |= mpfr_mul (res, res, res, MPFR_RNDU);
                      if (n & (1UL << i))
                        inexact |= mpfr_mul (res, res, x, rnd1);
                    });
      /* let r(n) be the number of roundings: we have r(2)=1, r(3)=2,
         and r(2n)=2r(n)+1, r(2n+1)=2r(n)+2, thus r(n)=n-1.
         Using Higham's method, to each rounding corresponds a factor
         (1-theta) with 0 <= theta <= 2^(1-p), thus at the end the
         absolute error is bounded by (n-1)*2^(1-p)*res <= 2*(n-1)*ulp(res)
         since 2^(-p)*x <= ulp(x). Since n < 2^i, this gives a maximal
         error of 2^(1+i)*ulp(res).
      */
      if (MPFR_LIKELY (inexact == 0
                       || MPFR_OVERFLOW (flags) || MPFR_UNDERFLOW (flags)
                       || MPFR_CAN_ROUND (res, err, MPFR_PREC (y), rnd)))
        break;
      /* Actualisation of the precision */
      MPFR_ZIV_NEXT (loop, prec);
      mpfr_set_prec (res, prec);
    }
  MPFR_ZIV_FREE (loop);

  if (MPFR_UNLIKELY (MPFR_OVERFLOW (flags) || MPFR_UNDERFLOW (flags)))
    {
      mpz_t z;

      /* Internal overflow or underflow. However the approximation error has
       * not been taken into account. So, let's solve this problem by using
       * mpfr_pow_z, which can handle it. This case could be improved in the
       * future, without having to use mpfr_pow_z.
       */
      MPFR_LOG_MSG (("Internal overflow or underflow,"
                     " let's use mpfr_pow_z.\n", 0));
      mpfr_clear (res);
      MPFR_SAVE_EXPO_FREE (expo);
      mpz_init (z);
      mpz_set_ui (z, n);
      inexact = mpfr_pow_z (y, x, z, rnd);
      mpz_clear (z);
      return inexact;
    }

  inexact = mpfr_set (y, res, rnd);
  mpfr_clear (res);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, inexact, rnd);
}
