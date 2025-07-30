/* mpfr_acos -- arc-cosinus of a floating-point number

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

int
mpfr_acos (mpfr_ptr acos, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_t xp, arcc, tmp;
  mpfr_exp_t supplement;
  mpfr_prec_t prec;
  int sign, compared, inexact;
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_ZIV_DECL (loop);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec(x), mpfr_log_prec, x, rnd_mode),
     ("acos[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec(acos), mpfr_log_prec, acos, inexact));

  /* Singular cases */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x) || MPFR_IS_INF (x))
        {
          MPFR_SET_NAN (acos);
          MPFR_RET_NAN;
        }
      else /* necessarily x=0 */
        {
          MPFR_ASSERTD(MPFR_IS_ZERO(x));
          /* acos(0)=Pi/2 */
          MPFR_SAVE_EXPO_MARK (expo);
          inexact = mpfr_const_pi (acos, rnd_mode);
          mpfr_div_2ui (acos, acos, 1, rnd_mode); /* exact */
          MPFR_SAVE_EXPO_FREE (expo);
          return mpfr_check_range (acos, inexact, rnd_mode);
        }
    }

  /* Set x_p=|x| */
  sign = MPFR_SIGN (x);
  mpfr_init2 (xp, MPFR_PREC (x));
  mpfr_abs (xp, x, MPFR_RNDN); /* Exact */

  compared = mpfr_cmp_ui (xp, 1);

  if (MPFR_UNLIKELY (compared >= 0))
    {
      mpfr_clear (xp);
      if (compared > 0) /* acos(x) = NaN for x > 1 */
        {
          MPFR_SET_NAN(acos);
          MPFR_RET_NAN;
        }
      else
        {
          if (MPFR_IS_POS_SIGN (sign)) /* acos(+1) = 0 */
            return mpfr_set_ui (acos, 0, rnd_mode);
          else /* acos(-1) = Pi */
            return mpfr_const_pi (acos, rnd_mode);
        }
    }

  MPFR_SAVE_EXPO_MARK (expo);

  /* Compute the supplement */
  mpfr_ui_sub (xp, 1, xp, MPFR_RNDD);
  if (MPFR_IS_POS_SIGN (sign))
    supplement = 2 - 2 * MPFR_GET_EXP (xp);
  else
    supplement = 2 - MPFR_GET_EXP (xp);
  mpfr_clear (xp);

  prec = MPFR_PREC (acos);
  prec += MPFR_INT_CEIL_LOG2(prec) + 10 + supplement;

  /* VL: The following change concerning prec comes from r3145
     "Optimize mpfr_acos by choosing a better initial precision."
     but it doesn't seem to be correct and leads to problems (assertion
     failure or very important inefficiency) with tiny arguments.
     Therefore, I've disabled it. */
  /* If x ~ 2^-N, acos(x) ~ PI/2 - x - x^3/6
     If Prec < 2*N, we can't round since x^3/6 won't be counted. */
#if 0
  if (MPFR_PREC (acos) >= MPFR_PREC (x) && MPFR_GET_EXP (x) < 0)
    {
      mpfr_uexp_t pmin = (mpfr_uexp_t) (-2 * MPFR_GET_EXP (x)) + 5;
      MPFR_ASSERTN (pmin <= MPFR_PREC_MAX);
      if (prec < pmin)
        prec = pmin;
    }
#endif

  mpfr_init2 (tmp, prec);
  mpfr_init2 (arcc, prec);

  MPFR_ZIV_INIT (loop, prec);
  for (;;)
    {
      /* acos(x) = Pi/2 - asin(x) = Pi/2 - atan(x/sqrt(1-x^2)) */
      mpfr_sqr (tmp, x, MPFR_RNDN);
      mpfr_ui_sub (tmp, 1, tmp, MPFR_RNDN);
      mpfr_sqrt (tmp, tmp, MPFR_RNDN);
      mpfr_div (tmp, x, tmp, MPFR_RNDN);
      mpfr_atan (arcc, tmp, MPFR_RNDN);
      mpfr_const_pi (tmp, MPFR_RNDN);
      mpfr_div_2ui (tmp, tmp, 1, MPFR_RNDN);
      mpfr_sub (arcc, tmp, arcc, MPFR_RNDN);

      if (MPFR_LIKELY (MPFR_CAN_ROUND (arcc, prec - supplement,
                                       MPFR_PREC (acos), rnd_mode)))
        break;
      MPFR_ZIV_NEXT (loop, prec);
      mpfr_set_prec (tmp, prec);
      mpfr_set_prec (arcc, prec);
    }
  MPFR_ZIV_FREE (loop);

  inexact = mpfr_set (acos, arcc, rnd_mode);
  mpfr_clear (tmp);
  mpfr_clear (arcc);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (acos, inexact, rnd_mode);
}
