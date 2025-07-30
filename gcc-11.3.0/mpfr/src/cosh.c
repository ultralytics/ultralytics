/* mpfr_cosh -- hyperbolic cosine

Copyright 2001-2002, 2004-2017 Free Software Foundation, Inc.
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

/* The computation of cosh is done by    *
 *  cosh= 1/2[e^(x)+e^(-x)]              */

int
mpfr_cosh (mpfr_ptr y, mpfr_srcptr xt , mpfr_rnd_t rnd_mode)
{
  mpfr_t x;
  int inexact;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC (
    ("x[%Pu]=%*.Rg rnd=%d", mpfr_get_prec (xt), mpfr_log_prec, xt, rnd_mode),
    ("y[%Pu]=%*.Rg inexact=%d", mpfr_get_prec (y), mpfr_log_prec, y,
     inexact));

  if (MPFR_UNLIKELY(MPFR_IS_SINGULAR(xt)))
    {
      if (MPFR_IS_NAN(xt))
        {
          MPFR_SET_NAN(y);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF(xt))
        {
          MPFR_SET_INF(y);
          MPFR_SET_POS(y);
          MPFR_RET(0);
        }
      else
        {
          MPFR_ASSERTD(MPFR_IS_ZERO(xt));
          return mpfr_set_ui (y, 1, rnd_mode); /* cosh(0) = 1 */
        }
    }

  MPFR_SAVE_EXPO_MARK (expo);

  /* cosh(x) = 1+x^2/2 + ... <= 1+x^2 for x <= 2.9828...,
     thus the error < 2^(2*EXP(x)). If x >= 1, then EXP(x) >= 1,
     thus the following will always fail. */
  MPFR_FAST_COMPUTE_IF_SMALL_INPUT (y, __gmpfr_one, -2 * MPFR_GET_EXP (xt), 0,
                                    1, rnd_mode, inexact = _inexact; goto end);

  MPFR_TMP_INIT_ABS(x, xt);
  /* General case */
  {
    /* Declaration of the intermediary variable */
    mpfr_t t, te;
    /* Declaration of the size variable */
    mpfr_prec_t Ny = MPFR_PREC(y);   /* Precision of output variable */
    mpfr_prec_t Nt;                  /* Precision of the intermediary variable */
    long int err;                  /* Precision of error */
    MPFR_ZIV_DECL (loop);
    MPFR_GROUP_DECL (group);

    /* compute the precision of intermediary variable */
    /* The optimal number of bits : see algorithms.tex */
    Nt = Ny + 3 + MPFR_INT_CEIL_LOG2 (Ny);

    /* initialise of intermediary variables */
    MPFR_GROUP_INIT_2 (group, Nt, t, te);

    /* First computation of cosh */
    MPFR_ZIV_INIT (loop, Nt);
    for (;;)
      {
        MPFR_BLOCK_DECL (flags);

        /* Compute cosh */
        MPFR_BLOCK (flags, mpfr_exp (te, x, MPFR_RNDD));  /* exp(x) */
        /* exp can overflow (but not underflow since x>0) */
        if (MPFR_OVERFLOW (flags))
          /* cosh(x) > exp(x), cosh(x) underflows too */
          {
            inexact = mpfr_overflow (y, rnd_mode, MPFR_SIGN_POS);
            MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, MPFR_FLAGS_OVERFLOW);
            break;
          }
        mpfr_ui_div (t, 1, te, MPFR_RNDU);   /* 1/exp(x) */
        mpfr_add (t, te, t, MPFR_RNDU);      /* exp(x) + 1/exp(x)*/
        mpfr_div_2ui (t, t, 1, MPFR_RNDN);   /* 1/2(exp(x) + 1/exp(x))*/

        /* Estimation of the error */
        err = Nt - 3;
        /* Check if we can round */
        if (MPFR_LIKELY (MPFR_CAN_ROUND (t, err, Ny, rnd_mode)))
          {
            inexact = mpfr_set (y, t, rnd_mode);
            break;
          }

        /* Actualisation of the precision */
        MPFR_ZIV_NEXT (loop, Nt);
        MPFR_GROUP_REPREC_2 (group, Nt, t, te);
      }
    MPFR_ZIV_FREE (loop);
    MPFR_GROUP_CLEAR (group);
  }

 end:
  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, inexact, rnd_mode);
}
