/* mpfr_acosh -- inverse hyperbolic cosine

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

/* The computation of acosh is done by   *
 *  acosh= ln(x + sqrt(x^2-1))           */

int
mpfr_acosh (mpfr_ptr y, mpfr_srcptr x , mpfr_rnd_t rnd_mode)
{
  MPFR_SAVE_EXPO_DECL (expo);
  int inexact;
  int comp;

  MPFR_LOG_FUNC (
    ("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd_mode),
    ("y[%Pu]=%.*Rg inexact=%d", mpfr_get_prec (y), mpfr_log_prec, y,
     inexact));

  /* Deal with special cases */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      /* Nan, or zero or -Inf */
      if (MPFR_IS_INF (x) && MPFR_IS_POS (x))
        {
          MPFR_SET_INF (y);
          MPFR_SET_POS (y);
          MPFR_RET (0);
        }
      else /* Nan, or zero or -Inf */
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
    }
  comp = mpfr_cmp_ui (x, 1);
  if (MPFR_UNLIKELY (comp < 0))
    {
      MPFR_SET_NAN (y);
      MPFR_RET_NAN;
    }
  else if (MPFR_UNLIKELY (comp == 0))
    {
      MPFR_SET_ZERO (y); /* acosh(1) = 0 */
      MPFR_SET_POS (y);
      MPFR_RET (0);
    }
  MPFR_SAVE_EXPO_MARK (expo);

  /* General case */
  {
    /* Declaration of the intermediary variables */
    mpfr_t t;
    /* Declaration of the size variables */
    mpfr_prec_t Ny = MPFR_PREC(y);   /* Precision of output variable */
    mpfr_prec_t Nt;                  /* Precision of the intermediary variable */
    mpfr_exp_t  err, exp_te, d;      /* Precision of error */
    MPFR_ZIV_DECL (loop);

    /* compute the precision of intermediary variable */
    /* the optimal number of bits : see algorithms.tex */
    Nt = Ny + 4 + MPFR_INT_CEIL_LOG2 (Ny);

    /* initialization of intermediary variables */
    mpfr_init2 (t, Nt);

    /* First computation of acosh */
    MPFR_ZIV_INIT (loop, Nt);
    for (;;)
      {
        MPFR_BLOCK_DECL (flags);

        /* compute acosh */
        MPFR_BLOCK (flags, mpfr_mul (t, x, x, MPFR_RNDD));  /* x^2 */
        if (MPFR_OVERFLOW (flags))
          {
            mpfr_t ln2;
            mpfr_prec_t pln2;

            /* As x is very large and the precision is not too large, we
               assume that we obtain the same result by evaluating ln(2x).
               We need to compute ln(x) + ln(2) as 2x can overflow. TODO:
               write a proof and add an MPFR_ASSERTN. */
            mpfr_log (t, x, MPFR_RNDN);  /* err(log) < 1/2 ulp(t) */
            pln2 = Nt - MPFR_PREC_MIN < MPFR_GET_EXP (t) ?
              MPFR_PREC_MIN : Nt - MPFR_GET_EXP (t);
            mpfr_init2 (ln2, pln2);
            mpfr_const_log2 (ln2, MPFR_RNDN);  /* err(ln2) < 1/2 ulp(t) */
            mpfr_add (t, t, ln2, MPFR_RNDN);  /* err <= 3/2 ulp(t) */
            mpfr_clear (ln2);
            err = 1;
          }
        else
          {
            exp_te = MPFR_GET_EXP (t);
            mpfr_sub_ui (t, t, 1, MPFR_RNDD);   /* x^2-1 */
            if (MPFR_UNLIKELY (MPFR_IS_ZERO (t)))
              {
                /* This means that x is very close to 1: x = 1 + t with
                   t < 2^(-Nt). We have: acosh(x) = sqrt(2t) (1 - eps(t))
                   with 0 < eps(t) < t / 12. */
                mpfr_sub_ui (t, x, 1, MPFR_RNDD);   /* t = x - 1 */
                mpfr_mul_2ui (t, t, 1, MPFR_RNDN);  /* 2t */
                mpfr_sqrt (t, t, MPFR_RNDN);        /* sqrt(2t) */
                err = 1;
              }
            else
              {
                d = exp_te - MPFR_GET_EXP (t);
                mpfr_sqrt (t, t, MPFR_RNDN);        /* sqrt(x^2-1) */
                mpfr_add (t, t, x, MPFR_RNDN);      /* sqrt(x^2-1)+x */
                mpfr_log (t, t, MPFR_RNDN);         /* ln(sqrt(x^2-1)+x) */

                /* error estimate -- see algorithms.tex */
                err = 3 + MAX (1, d) - MPFR_GET_EXP (t);
                /* error is bounded by 1/2 + 2^err <= 2^(max(0,1+err)) */
                err = MAX (0, 1 + err);
              }
          }

        if (MPFR_LIKELY (MPFR_CAN_ROUND (t, Nt - err, Ny, rnd_mode)))
          break;

        /* reactualisation of the precision */
        MPFR_ZIV_NEXT (loop, Nt);
        mpfr_set_prec (t, Nt);
      }
    MPFR_ZIV_FREE (loop);

    inexact = mpfr_set (y, t, rnd_mode);

    mpfr_clear (t);
  }

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, inexact, rnd_mode);
}
