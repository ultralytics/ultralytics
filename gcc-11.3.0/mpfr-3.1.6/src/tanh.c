/* mpfr_tanh -- hyperbolic tangent

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
mpfr_tanh (mpfr_ptr y, mpfr_srcptr xt , mpfr_rnd_t rnd_mode)
{
  /****** Declaration ******/
  mpfr_t x;
  int inexact;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (xt), mpfr_log_prec, xt, rnd_mode),
     ("y[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (y), mpfr_log_prec, y, inexact));

  /* Special value checking */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (xt)))
    {
      if (MPFR_IS_NAN (xt))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (xt))
        {
          /* tanh(inf) = 1 && tanh(-inf) = -1 */
          return mpfr_set_si (y, MPFR_INT_SIGN (xt), rnd_mode);
        }
      else /* tanh (0) = 0 and xt is zero */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO(xt));
          MPFR_SET_ZERO (y);
          MPFR_SET_SAME_SIGN (y, xt);
          MPFR_RET (0);
        }
    }

  /* tanh(x) = x - x^3/3 + ... so the error is < 2^(3*EXP(x)-1) */
  MPFR_FAST_COMPUTE_IF_SMALL_INPUT (y, xt, -2 * MPFR_GET_EXP (xt), 1, 0,
                                    rnd_mode, {});

  MPFR_TMP_INIT_ABS (x, xt);

  MPFR_SAVE_EXPO_MARK (expo);

  /* General case */
  {
    /* Declaration of the intermediary variable */
    mpfr_t t, te;
    mpfr_exp_t d;

    /* Declaration of the size variable */
    mpfr_prec_t Ny = MPFR_PREC(y);   /* target precision */
    mpfr_prec_t Nt;                  /* working precision */
    long int err;                  /* error */
    int sign = MPFR_SIGN (xt);
    MPFR_ZIV_DECL (loop);
    MPFR_GROUP_DECL (group);

    /* First check for BIG overflow of exp(2*x):
       For x > 0, exp(2*x) > 2^(2*x)
       If 2 ^(2*x) > 2^emax or x>emax/2, there is an overflow */
    if (MPFR_UNLIKELY (mpfr_cmp_si (x, __gmpfr_emax/2) >= 0)) {
      /* initialise of intermediary variables
         since 'set_one' label assumes the variables have been
         initialize */
      MPFR_GROUP_INIT_2 (group, MPFR_PREC_MIN, t, te);
      goto set_one;
    }

    /* Compute the precision of intermediary variable */
    /* The optimal number of bits: see algorithms.tex */
    Nt = Ny + MPFR_INT_CEIL_LOG2 (Ny) + 4;
    /* if x is small, there will be a cancellation in exp(2x)-1 */
    if (MPFR_GET_EXP (x) < 0)
      Nt += -MPFR_GET_EXP (x);

    /* initialise of intermediary variable */
    MPFR_GROUP_INIT_2 (group, Nt, t, te);

    MPFR_ZIV_INIT (loop, Nt);
    for (;;) {
      /* tanh = (exp(2x)-1)/(exp(2x)+1) */
      mpfr_mul_2ui (te, x, 1, MPFR_RNDN);  /* 2x */
      /* since x > 0, we can only have an overflow */
      mpfr_exp (te, te, MPFR_RNDN);        /* exp(2x) */
      if (MPFR_UNLIKELY (MPFR_IS_INF (te))) {
      set_one:
        inexact = MPFR_FROM_SIGN_TO_INT (sign);
        mpfr_set4 (y, __gmpfr_one, MPFR_RNDN, sign);
        if (MPFR_IS_LIKE_RNDZ (rnd_mode, MPFR_IS_NEG_SIGN (sign)))
          {
            inexact = -inexact;
            mpfr_nexttozero (y);
          }
        break;
      }
      d = MPFR_GET_EXP (te);              /* For Error calculation */
      mpfr_add_ui (t, te, 1, MPFR_RNDD);   /* exp(2x) + 1*/
      mpfr_sub_ui (te, te, 1, MPFR_RNDU);  /* exp(2x) - 1*/
      d = d - MPFR_GET_EXP (te);
      mpfr_div (t, te, t, MPFR_RNDN);      /* (exp(2x)-1)/(exp(2x)+1)*/

      /* Calculation of the error */
      d = MAX(3, d + 1);
      err = Nt - (d + 1);

      if (MPFR_LIKELY ((d <= Nt / 2) && MPFR_CAN_ROUND (t, err, Ny, rnd_mode)))
        {
          inexact = mpfr_set4 (y, t, rnd_mode, sign);
          break;
        }

      /* if t=1, we still can round since |sinh(x)| < 1 */
      if (MPFR_GET_EXP (t) == 1)
        goto set_one;

      /* Actualisation of the precision */
      MPFR_ZIV_NEXT (loop, Nt);
      MPFR_GROUP_REPREC_2 (group, Nt, t, te);
    }
    MPFR_ZIV_FREE (loop);
    MPFR_GROUP_CLEAR (group);
  }
  MPFR_SAVE_EXPO_FREE (expo);
  inexact = mpfr_check_range (y, inexact, rnd_mode);

  return inexact;
}

