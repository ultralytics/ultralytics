/* mpfr_mul_ui -- multiply a floating-point number by a machine integer
   mpfr_mul_si -- multiply a floating-point number by a machine integer

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

int
mpfr_mul_ui (mpfr_ptr y, mpfr_srcptr x, unsigned long int u, mpfr_rnd_t rnd_mode)
{
  mp_limb_t *yp;
  mp_size_t xn;
  int cnt, inexact;
  MPFR_TMP_DECL (marker);

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (x))
        {
          if (u != 0)
            {
              MPFR_SET_INF (y);
              MPFR_SET_SAME_SIGN (y, x);
              MPFR_RET (0); /* infinity is exact */
            }
          else /* 0 * infinity */
            {
              MPFR_SET_NAN (y);
              MPFR_RET_NAN;
            }
        }
      else /* x is zero */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (x));
          MPFR_SET_ZERO (y);
          MPFR_SET_SAME_SIGN (y, x);
          MPFR_RET (0); /* zero is exact */
        }
    }
  else if (MPFR_UNLIKELY (u <= 1))
    {
      if (u < 1)
        {
          MPFR_SET_ZERO (y);
          MPFR_SET_SAME_SIGN (y, x);
          MPFR_RET (0); /* zero is exact */
        }
      else
        return mpfr_set (y, x, rnd_mode);
    }
  else if (MPFR_UNLIKELY (IS_POW2 (u)))
    return mpfr_mul_2si (y, x, MPFR_INT_CEIL_LOG2 (u), rnd_mode);

  yp = MPFR_MANT (y);
  xn = MPFR_LIMB_SIZE (x);

  MPFR_ASSERTD (xn < MP_SIZE_T_MAX);
  MPFR_TMP_MARK(marker);
  yp = MPFR_TMP_LIMBS_ALLOC (xn + 1);

  MPFR_ASSERTN (u == (mp_limb_t) u);
  yp[xn] = mpn_mul_1 (yp, MPFR_MANT (x), xn, u);

  /* x * u is stored in yp[xn], ..., yp[0] */

  /* since the case u=1 was treated above, we have u >= 2, thus
     yp[xn] >= 1 since x was msb-normalized */
  MPFR_ASSERTD (yp[xn] != 0);
  if (MPFR_LIKELY (MPFR_LIMB_MSB (yp[xn]) == 0))
    {
      count_leading_zeros (cnt, yp[xn]);
      mpn_lshift (yp, yp, xn + 1, cnt);
    }
  else
    {
      cnt = 0;
    }

  /* now yp[xn], ..., yp[0] is msb-normalized too, and has at most
     PREC(x) + (GMP_NUMB_BITS - cnt) non-zero bits */
  MPFR_RNDRAW (inexact, y, yp, (mpfr_prec_t) (xn + 1) * GMP_NUMB_BITS,
               rnd_mode, MPFR_SIGN (x), cnt -- );

  MPFR_TMP_FREE (marker);

  cnt = GMP_NUMB_BITS - cnt;
  if (MPFR_UNLIKELY (__gmpfr_emax < MPFR_EMAX_MIN + cnt
                     || MPFR_GET_EXP (x) > __gmpfr_emax - cnt))
    return mpfr_overflow (y, rnd_mode, MPFR_SIGN(x));

  MPFR_SET_EXP (y, MPFR_GET_EXP (x) + cnt);
  MPFR_SET_SAME_SIGN (y, x);

  MPFR_RET (inexact);
}

int mpfr_mul_si (mpfr_ptr y, mpfr_srcptr x, long int u, mpfr_rnd_t rnd_mode)
{
  int res;

  if (u >= 0)
    res = mpfr_mul_ui (y, x, u, rnd_mode);
  else
    {
      res = - mpfr_mul_ui (y, x, - (unsigned long) u,
                           MPFR_INVERT_RND (rnd_mode));
      MPFR_CHANGE_SIGN (y);
    }
  return res;
}
