/* mpfr_hypot -- Euclidean distance

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

/* The computation of hypot of x and y is done by  *
 *    hypot(x,y)= sqrt(x^2+y^2) = z                */

int
mpfr_hypot (mpfr_ptr z, mpfr_srcptr x, mpfr_srcptr y, mpfr_rnd_t rnd_mode)
{
  int inexact, exact;
  mpfr_t t, te, ti; /* auxiliary variables */
  mpfr_prec_t N, Nz; /* size variables */
  mpfr_prec_t Nt;   /* precision of the intermediary variable */
  mpfr_prec_t threshold;
  mpfr_exp_t Ex, sh;
  mpfr_uexp_t diff_exp;

  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_ZIV_DECL (loop);
  MPFR_BLOCK_DECL (flags);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg y[%Pu]=%.*Rg rnd=%d",
      mpfr_get_prec (x), mpfr_log_prec, x,
      mpfr_get_prec (y), mpfr_log_prec, y, rnd_mode),
     ("z[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (z), mpfr_log_prec, z, inexact));

  /* particular cases */
  if (MPFR_ARE_SINGULAR (x, y))
    {
      if (MPFR_IS_INF (x) || MPFR_IS_INF (y))
        {
          /* Return +inf, even when the other number is NaN. */
          MPFR_SET_INF (z);
          MPFR_SET_POS (z);
          MPFR_RET (0);
        }
      else if (MPFR_IS_NAN (x) || MPFR_IS_NAN (y))
        {
          MPFR_SET_NAN (z);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_ZERO (x))
        return mpfr_abs (z, y, rnd_mode);
      else /* y is necessarily 0 */
        return mpfr_abs (z, x, rnd_mode);
    }

  if (mpfr_cmpabs (x, y) < 0)
    {
      mpfr_srcptr u;
      u = x;
      x = y;
      y = u;
    }

  /* now |x| >= |y| */

  Ex = MPFR_GET_EXP (x);
  diff_exp = (mpfr_uexp_t) Ex - MPFR_GET_EXP (y);

  N = MPFR_PREC (x);   /* Precision of input variable */
  Nz = MPFR_PREC (z);   /* Precision of output variable */
  threshold = (MAX (N, Nz) + (rnd_mode == MPFR_RNDN ? 1 : 0)) << 1;
  if (rnd_mode == MPFR_RNDA)
    rnd_mode = MPFR_RNDU; /* since the result is positive, RNDA = RNDU */

  /* Is |x| a suitable approximation to the precision Nz ?
     (see algorithms.tex for explanations) */
  if (diff_exp > threshold)
    /* result is |x| or |x|+ulp(|x|,Nz) */
    {
      if (MPFR_UNLIKELY (rnd_mode == MPFR_RNDU))
        {
          /* If z > abs(x), then it was already rounded up; otherwise
             z = abs(x), and we need to add one ulp due to y. */
          if (mpfr_abs (z, x, rnd_mode) == 0)
            mpfr_nexttoinf (z);
          MPFR_RET (1);
        }
      else /* MPFR_RNDZ, MPFR_RNDD, MPFR_RNDN */
        {
          if (MPFR_LIKELY (Nz >= N))
            {
              mpfr_abs (z, x, rnd_mode);  /* exact */
              MPFR_RET (-1);
            }
          else
            {
              MPFR_SET_EXP (z, Ex);
              MPFR_SET_SIGN (z, 1);
              MPFR_RNDRAW_GEN (inexact, z, MPFR_MANT (x), N, rnd_mode, 1,
                               goto addoneulp,
                               if (MPFR_UNLIKELY (++ MPFR_EXP (z) >
                                                  __gmpfr_emax))
                                 return mpfr_overflow (z, rnd_mode, 1);
                               );

              if (MPFR_UNLIKELY (inexact == 0))
                inexact = -1;
              MPFR_RET (inexact);
            }
        }
    }

  /* General case */

  N = MAX (MPFR_PREC (x), MPFR_PREC (y));

  /* working precision */
  Nt = Nz + MPFR_INT_CEIL_LOG2 (Nz) + 4;

  mpfr_init2 (t, Nt);
  mpfr_init2 (te, Nt);
  mpfr_init2 (ti, Nt);

  MPFR_SAVE_EXPO_MARK (expo);

  /* Scale x and y to avoid overflow/underflow in x^2 and overflow in y^2
     (as |x| >= |y|). The scaling of y can underflow only when the target
     precision is huge, otherwise the case would already have been handled
     by the diff_exp > threshold code. */
  sh = mpfr_get_emax () / 2 - Ex - 1;

  MPFR_ZIV_INIT (loop, Nt);
  for (;;)
    {
      mpfr_prec_t err;

      exact = mpfr_mul_2si (te, x, sh, MPFR_RNDZ);
      exact |= mpfr_mul_2si (ti, y, sh, MPFR_RNDZ);
      exact |= mpfr_sqr (te, te, MPFR_RNDZ);
      /* Use fma in order to avoid underflow when diff_exp<=MPFR_EMAX_MAX-2 */
      exact |= mpfr_fma (t, ti, ti, te, MPFR_RNDZ);
      exact |= mpfr_sqrt (t, t, MPFR_RNDZ);

      err = Nt < N ? 4 : 2;
      if (MPFR_LIKELY (exact == 0
                       || MPFR_CAN_ROUND (t, Nt-err, Nz, rnd_mode)))
        break;

      MPFR_ZIV_NEXT (loop, Nt);
      mpfr_set_prec (t, Nt);
      mpfr_set_prec (te, Nt);
      mpfr_set_prec (ti, Nt);
    }
  MPFR_ZIV_FREE (loop);

  MPFR_BLOCK (flags, inexact = mpfr_div_2si (z, t, sh, rnd_mode));
  MPFR_ASSERTD (exact == 0 || inexact != 0);

  mpfr_clear (t);
  mpfr_clear (ti);
  mpfr_clear (te);

  /*
    exact  inexact
    0         0         result is exact, ternary flag is 0
    0       non zero    t is exact, ternary flag given by inexact
    1         0         impossible (see above)
    1       non zero    ternary flag given by inexact
  */

  MPFR_SAVE_EXPO_FREE (expo);

  if (MPFR_OVERFLOW (flags))
    mpfr_set_overflow ();
  /* hypot(x,y) >= |x|, thus underflow is not possible. */

  return mpfr_check_range (z, inexact, rnd_mode);
}
