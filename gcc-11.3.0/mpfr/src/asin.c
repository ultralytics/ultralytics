/* mpfr_asin -- arc-sinus of a floating-point number

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

#include "mpfr-impl.h"

int
mpfr_asin (mpfr_ptr asin, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_t xp;
  int compared, inexact;
  mpfr_prec_t prec;
  mpfr_exp_t xp_exp;
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_ZIV_DECL (loop);

  MPFR_LOG_FUNC (
    ("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd_mode),
    ("asin[%Pu]=%.*Rg inexact=%d", mpfr_get_prec (asin), mpfr_log_prec, asin,
     inexact));

  /* Special cases */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x) || MPFR_IS_INF (x))
        {
          MPFR_SET_NAN (asin);
          MPFR_RET_NAN;
        }
      else /* x = 0 */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (x));
          MPFR_SET_ZERO (asin);
          MPFR_SET_SAME_SIGN (asin, x);
          MPFR_RET (0); /* exact result */
        }
    }

  /* asin(x) = x + x^3/6 + ... so the error is < 2^(3*EXP(x)-2) */
  MPFR_FAST_COMPUTE_IF_SMALL_INPUT (asin, x, -2 * MPFR_GET_EXP (x), 2, 1,
                                    rnd_mode, {});

  /* Set x_p=|x| (x is a normal number) */
  mpfr_init2 (xp, MPFR_PREC (x));
  inexact = mpfr_abs (xp, x, MPFR_RNDN);
  MPFR_ASSERTD (inexact == 0);

  compared = mpfr_cmp_ui (xp, 1);

  MPFR_SAVE_EXPO_MARK (expo);

  if (MPFR_UNLIKELY (compared >= 0))
    {
      mpfr_clear (xp);
      if (compared > 0)                  /* asin(x) = NaN for |x| > 1 */
        {
          MPFR_SAVE_EXPO_FREE (expo);
          MPFR_SET_NAN (asin);
          MPFR_RET_NAN;
        }
      else                              /* x = 1 or x = -1 */
        {
          if (MPFR_IS_POS (x)) /* asin(+1) = Pi/2 */
            inexact = mpfr_const_pi (asin, rnd_mode);
          else /* asin(-1) = -Pi/2 */
            {
              inexact = -mpfr_const_pi (asin, MPFR_INVERT_RND(rnd_mode));
              MPFR_CHANGE_SIGN (asin);
            }
          mpfr_div_2ui (asin, asin, 1, rnd_mode);
        }
    }
  else
    {
      /* Compute exponent of 1 - ABS(x) */
      mpfr_ui_sub (xp, 1, xp, MPFR_RNDD);
      MPFR_ASSERTD (MPFR_GET_EXP (xp) <= 0);
      MPFR_ASSERTD (MPFR_GET_EXP (x) <= 0);
      xp_exp = 2 - MPFR_GET_EXP (xp);

      /* Set up initial prec */
      prec = MPFR_PREC (asin) + 10 + xp_exp;

      /* use asin(x) = atan(x/sqrt(1-x^2)) */
      MPFR_ZIV_INIT (loop, prec);
      for (;;)
        {
          mpfr_set_prec (xp, prec);
          mpfr_sqr (xp, x, MPFR_RNDN);
          mpfr_ui_sub (xp, 1, xp, MPFR_RNDN);
          mpfr_sqrt (xp, xp, MPFR_RNDN);
          mpfr_div (xp, x, xp, MPFR_RNDN);
          mpfr_atan (xp, xp, MPFR_RNDN);
          if (MPFR_LIKELY (MPFR_CAN_ROUND (xp, prec - xp_exp,
                                           MPFR_PREC (asin), rnd_mode)))
            break;
          MPFR_ZIV_NEXT (loop, prec);
        }
      MPFR_ZIV_FREE (loop);
      inexact = mpfr_set (asin, xp, rnd_mode);

      mpfr_clear (xp);
    }

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (asin, inexact, rnd_mode);
}
