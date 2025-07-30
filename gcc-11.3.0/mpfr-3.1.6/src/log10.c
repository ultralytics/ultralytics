/* mpfr_log10 -- logarithm in base 10.

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

 /* The computation of r=log10(a)

    r=log10(a)=log(a)/log(10)
 */

int
mpfr_log10 (mpfr_ptr r, mpfr_srcptr a, mpfr_rnd_t rnd_mode)
{
  int inexact;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("a[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (a), mpfr_log_prec, a, rnd_mode),
     ("r[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (r), mpfr_log_prec, r, inexact));

  /* If a is NaN, the result is NaN */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (a)))
    {
      if (MPFR_IS_NAN (a))
        {
          MPFR_SET_NAN (r);
          MPFR_RET_NAN;
        }
      /* check for infinity before zero */
      else if (MPFR_IS_INF (a))
        {
          if (MPFR_IS_NEG (a))
            /* log10(-Inf) = NaN */
            {
              MPFR_SET_NAN (r);
              MPFR_RET_NAN;
            }
          else /* log10(+Inf) = +Inf */
            {
              MPFR_SET_INF (r);
              MPFR_SET_POS (r);
              MPFR_RET (0); /* exact */
            }
        }
      else /* a = 0 */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (a));
          MPFR_SET_INF (r);
          MPFR_SET_NEG (r);
          mpfr_set_divby0 ();
          MPFR_RET (0); /* log10(0) is an exact -infinity */
        }
    }

  /* If a is negative, the result is NaN */
  if (MPFR_UNLIKELY (MPFR_IS_NEG (a)))
    {
      MPFR_SET_NAN (r);
      MPFR_RET_NAN;
    }

  /* If a is 1, the result is 0 */
  if (mpfr_cmp_ui (a, 1) == 0)
    {
      MPFR_SET_ZERO (r);
      MPFR_SET_POS (r);
      MPFR_RET (0); /* result is exact */
    }

  MPFR_SAVE_EXPO_MARK (expo);

  /* General case */
  {
    /* Declaration of the intermediary variable */
    mpfr_t t, tt;
    MPFR_ZIV_DECL (loop);
    /* Declaration of the size variable */
    mpfr_prec_t Ny = MPFR_PREC(r);   /* Precision of output variable */
    mpfr_prec_t Nt;        /* Precision of the intermediary variable */
    mpfr_exp_t  err;                           /* Precision of error */

    /* compute the precision of intermediary variable */
    /* the optimal number of bits : see algorithms.tex */
    Nt = Ny + 4 + MPFR_INT_CEIL_LOG2 (Ny);

    /* initialise of intermediary variables */
    mpfr_init2 (t, Nt);
    mpfr_init2 (tt, Nt);

    /* First computation of log10 */
    MPFR_ZIV_INIT (loop, Nt);
    for (;;)
      {
        /* compute log10 */
        mpfr_set_ui (t, 10, MPFR_RNDN);   /* 10 */
        mpfr_log (t, t, MPFR_RNDD);       /* log(10) */
        mpfr_log (tt, a, MPFR_RNDN);      /* log(a) */
        mpfr_div (t, tt, t, MPFR_RNDN);   /* log(a)/log(10) */

        /* estimation of the error */
        err = Nt - 4;
        if (MPFR_LIKELY (MPFR_CAN_ROUND (t, err, Ny, rnd_mode)))
          break;

        /* log10(10^n) is exact:
           FIXME: Can we have 10^n exactly representable as a mpfr_t
           but n can't fit an unsigned long? */
        if (MPFR_IS_POS (t)
            && mpfr_integer_p (t) && mpfr_fits_ulong_p (t, MPFR_RNDN)
            && !mpfr_ui_pow_ui (tt, 10, mpfr_get_ui (t, MPFR_RNDN), MPFR_RNDN)
            && mpfr_cmp (a, tt) == 0)
          break;

        /* actualisation of the precision */
        MPFR_ZIV_NEXT (loop, Nt);
        mpfr_set_prec (t, Nt);
        mpfr_set_prec (tt, Nt);
      }
    MPFR_ZIV_FREE (loop);

    inexact = mpfr_set (r, t, rnd_mode);

    mpfr_clear (t);
    mpfr_clear (tt);
  }

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (r, inexact, rnd_mode);
}
