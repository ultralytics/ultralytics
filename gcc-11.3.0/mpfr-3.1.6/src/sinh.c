/* mpfr_sinh -- hyperbolic sine

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

 /* The computation of sinh is done by
    sinh(x) = 1/2 [e^(x)-e^(-x)]          */

int
mpfr_sinh (mpfr_ptr y, mpfr_srcptr xt, mpfr_rnd_t rnd_mode)
{
  mpfr_t x;
  int inexact;

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (xt), mpfr_log_prec, xt, rnd_mode),
     ("y[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (y), mpfr_log_prec, y, inexact));

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (xt)))
    {
      if (MPFR_IS_NAN (xt))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (xt))
        {
          MPFR_SET_INF (y);
          MPFR_SET_SAME_SIGN (y, xt);
          MPFR_RET (0);
        }
      else /* xt is zero */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (xt));
          MPFR_SET_ZERO (y);   /* sinh(0) = 0 */
          MPFR_SET_SAME_SIGN (y, xt);
          MPFR_RET (0);
        }
    }

  /* sinh(x) = x + x^3/6 + ... so the error is < 2^(3*EXP(x)-2) */
  MPFR_FAST_COMPUTE_IF_SMALL_INPUT (y, xt, -2 * MPFR_GET_EXP(xt), 2, 1,
                                    rnd_mode, {});

  MPFR_TMP_INIT_ABS (x, xt);

  {
    mpfr_t t, ti;
    mpfr_exp_t d;
    mpfr_prec_t Nt;    /* Precision of the intermediary variable */
    long int err;    /* Precision of error */
    MPFR_ZIV_DECL (loop);
    MPFR_SAVE_EXPO_DECL (expo);
    MPFR_GROUP_DECL (group);

    MPFR_SAVE_EXPO_MARK (expo);

    /* compute the precision of intermediary variable */
    Nt = MAX (MPFR_PREC (x), MPFR_PREC (y));
    /* the optimal number of bits : see algorithms.ps */
    Nt = Nt + MPFR_INT_CEIL_LOG2 (Nt) + 4;
    /* If x is near 0, exp(x) - 1/exp(x) = 2*x+x^3/3+O(x^5) */
    if (MPFR_GET_EXP (x) < 0)
      Nt -= 2*MPFR_GET_EXP (x);

    /* initialise of intermediary variables */
    MPFR_GROUP_INIT_2 (group, Nt, t, ti);

    /* First computation of sinh */
    MPFR_ZIV_INIT (loop, Nt);
    for (;;)
      {
        MPFR_BLOCK_DECL (flags);

        /* compute sinh */
        MPFR_BLOCK (flags, mpfr_exp (t, x, MPFR_RNDD));
        if (MPFR_OVERFLOW (flags))
          /* exp(x) does overflow */
          {
            /* sinh(x) = 2 * sinh(x/2) * cosh(x/2) */
            mpfr_div_2ui (ti, x, 1, MPFR_RNDD); /* exact */

            /* t <- cosh(x/2): error(t) <= 1 ulp(t) */
            MPFR_BLOCK (flags, mpfr_cosh (t, ti, MPFR_RNDD));
            if (MPFR_OVERFLOW (flags))
              /* when x>1 we have |sinh(x)| >= cosh(x/2), so sinh(x)
                 overflows too */
              {
                inexact = mpfr_overflow (y, rnd_mode, MPFR_SIGN (xt));
                MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, MPFR_FLAGS_OVERFLOW);
                break;
              }

            /* ti <- sinh(x/2): , error(ti) <= 1 ulp(ti)
               cannot overflow because 0 < sinh(x) < cosh(x) when x > 0 */
            mpfr_sinh (ti, ti, MPFR_RNDD);

            /* multiplication below, error(t) <= 5 ulp(t) */
            MPFR_BLOCK (flags, mpfr_mul (t, t, ti, MPFR_RNDD));
            if (MPFR_OVERFLOW (flags))
              {
                inexact = mpfr_overflow (y, rnd_mode, MPFR_SIGN (xt));
                MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, MPFR_FLAGS_OVERFLOW);
                break;
              }

            /* doubling below, exact */
            MPFR_BLOCK (flags, mpfr_mul_2ui (t, t, 1, MPFR_RNDN));
            if (MPFR_OVERFLOW (flags))
              {
                inexact = mpfr_overflow (y, rnd_mode, MPFR_SIGN (xt));
                MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, MPFR_FLAGS_OVERFLOW);
                break;
              }

            /* we have lost at most 3 bits of precision */
            err = Nt - 3;
            if (MPFR_LIKELY (MPFR_CAN_ROUND (t, err, MPFR_PREC (y),
                                             rnd_mode)))
              {
                inexact = mpfr_set4 (y, t, rnd_mode, MPFR_SIGN (xt));
                break;
              }
            err = Nt; /* double the precision */
          }
        else
          {
            d = MPFR_GET_EXP (t);
            mpfr_ui_div (ti, 1, t, MPFR_RNDU); /* 1/exp(x) */
            mpfr_sub (t, t, ti, MPFR_RNDN);    /* exp(x) - 1/exp(x) */
            mpfr_div_2ui (t, t, 1, MPFR_RNDN);  /* 1/2(exp(x) - 1/exp(x)) */

            /* it may be that t is zero (in fact, it can only occur when te=1,
               and thus ti=1 too) */
            if (MPFR_IS_ZERO (t))
              err = Nt; /* double the precision */
            else
              {
                /* calculation of the error */
                d = d - MPFR_GET_EXP (t) + 2;
                /* error estimate: err = Nt-(__gmpfr_ceil_log2(1+pow(2,d)));*/
                err = Nt - (MAX (d, 0) + 1);
                if (MPFR_LIKELY (MPFR_CAN_ROUND (t, err, MPFR_PREC (y),
                                                 rnd_mode)))
                  {
                    inexact = mpfr_set4 (y, t, rnd_mode, MPFR_SIGN (xt));
                    break;
                  }
              }
          }

        /* actualisation of the precision */
        Nt += err;
        MPFR_ZIV_NEXT (loop, Nt);
        MPFR_GROUP_REPREC_2 (group, Nt, t, ti);
      }
    MPFR_ZIV_FREE (loop);
    MPFR_GROUP_CLEAR (group);
    MPFR_SAVE_EXPO_FREE (expo);
  }

  return mpfr_check_range (y, inexact, rnd_mode);
}
