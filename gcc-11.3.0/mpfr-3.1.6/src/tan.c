/* mpfr_tan -- tangent of a floating-point number

Copyright 2001-2017 Free Software Foundation, Inc.
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

/* computes tan(x) = sign(x)*sqrt(1/cos(x)^2-1) */
int
mpfr_tan (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_prec_t precy, m;
  int inexact;
  mpfr_t s, c;
  MPFR_ZIV_DECL (loop);
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_GROUP_DECL (group);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd_mode),
     ("y[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (y), mpfr_log_prec, y, inexact));

  if (MPFR_UNLIKELY(MPFR_IS_SINGULAR(x)))
    {
      if (MPFR_IS_NAN(x) || MPFR_IS_INF(x))
        {
          MPFR_SET_NAN(y);
          MPFR_RET_NAN;
        }
      else /* x is zero */
        {
          MPFR_ASSERTD(MPFR_IS_ZERO(x));
          MPFR_SET_ZERO(y);
          MPFR_SET_SAME_SIGN(y, x);
          MPFR_RET(0);
        }
    }

  /* tan(x) = x + x^3/3 + ... so the error is < 2^(3*EXP(x)-1) */
  MPFR_FAST_COMPUTE_IF_SMALL_INPUT (y, x, -2 * MPFR_GET_EXP (x), 1, 1,
                                    rnd_mode, {});

  MPFR_SAVE_EXPO_MARK (expo);

  /* Compute initial precision */
  precy = MPFR_PREC (y);
  m = precy + MPFR_INT_CEIL_LOG2 (precy) + 13;
  MPFR_ASSERTD (m >= 2); /* needed for the error analysis in algorithms.tex */

  MPFR_GROUP_INIT_2 (group, m, s, c);
  MPFR_ZIV_INIT (loop, m);
  for (;;)
    {
      /* The only way to get an overflow is to get ~ Pi/2
         But the result will be ~ 2^Prec(y). */
      mpfr_sin_cos (s, c, x, MPFR_RNDN); /* err <= 1/2 ulp on s and c */
      mpfr_div (c, s, c, MPFR_RNDN);     /* err <= 4 ulps */
      MPFR_ASSERTD (!MPFR_IS_SINGULAR (c));
      if (MPFR_LIKELY (MPFR_CAN_ROUND (c, m - 2, precy, rnd_mode)))
        break;
      MPFR_ZIV_NEXT (loop, m);
      MPFR_GROUP_REPREC_2 (group, m, s, c);
    }
  MPFR_ZIV_FREE (loop);
  inexact = mpfr_set (y, c, rnd_mode);
  MPFR_GROUP_CLEAR (group);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, inexact, rnd_mode);
}
