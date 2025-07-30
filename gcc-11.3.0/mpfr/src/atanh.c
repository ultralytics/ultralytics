/* mpfr_atanh -- Inverse Hyperbolic Tangente

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

 /* The computation of atanh is done by
       atanh= 1/2*ln(x+1)-1/2*ln(1-x)   */

int
mpfr_atanh (mpfr_ptr y, mpfr_srcptr xt , mpfr_rnd_t rnd_mode)
{
  int inexact;
  mpfr_t x, t, te;
  mpfr_prec_t Nx, Ny, Nt;
  mpfr_exp_t err;
  MPFR_ZIV_DECL (loop);
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (xt), mpfr_log_prec, xt, rnd_mode),
    ("y[%Pu]=%.*Rg inexact=%d",
     mpfr_get_prec (y), mpfr_log_prec, y, inexact));

  /* Special cases */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (xt)))
    {
      /* atanh(NaN) = NaN, and atanh(+/-Inf) = NaN since tanh gives a result
         between -1 and 1 */
      if (MPFR_IS_NAN (xt) || MPFR_IS_INF (xt))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else /* necessarily xt is 0 */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (xt));
          MPFR_SET_ZERO (y);   /* atanh(0) = 0 */
          MPFR_SET_SAME_SIGN (y,xt);
          MPFR_RET (0);
        }
    }

  /* atanh (x) = NaN as soon as |x| > 1, and arctanh(+/-1) = +/-Inf */
  if (MPFR_UNLIKELY (MPFR_GET_EXP (xt) > 0))
    {
      if (MPFR_GET_EXP (xt) == 1 && mpfr_powerof2_raw (xt))
        {
          MPFR_SET_INF (y);
          MPFR_SET_SAME_SIGN (y, xt);
          mpfr_set_divby0 ();
          MPFR_RET (0);
        }
      MPFR_SET_NAN (y);
      MPFR_RET_NAN;
    }

  /* atanh(x) = x + x^3/3 + ... so the error is < 2^(3*EXP(x)-1) */
  MPFR_FAST_COMPUTE_IF_SMALL_INPUT (y, xt, -2 * MPFR_GET_EXP (xt), 1, 1,
                                    rnd_mode, {});

  MPFR_SAVE_EXPO_MARK (expo);

  /* Compute initial precision */
  Nx = MPFR_PREC (xt);
  MPFR_TMP_INIT_ABS (x, xt);
  Ny = MPFR_PREC (y);
  Nt = MAX (Nx, Ny);
  /* the optimal number of bits : see algorithms.ps */
  Nt = Nt + MPFR_INT_CEIL_LOG2 (Nt) + 4;

  /* initialise of intermediary variable */
  mpfr_init2 (t, Nt);
  mpfr_init2 (te, Nt);

  /* First computation of cosh */
  MPFR_ZIV_INIT (loop, Nt);
  for (;;)
    {
      /* compute atanh */
      mpfr_ui_sub (te, 1, x, MPFR_RNDU);   /* (1-xt)*/
      mpfr_add_ui (t,  x, 1, MPFR_RNDD);   /* (xt+1)*/
      mpfr_div (t, t, te, MPFR_RNDN);      /* (1+xt)/(1-xt)*/
      mpfr_log (t, t, MPFR_RNDN);          /* ln((1+xt)/(1-xt))*/
      mpfr_div_2ui (t, t, 1, MPFR_RNDN);   /* (1/2)*ln((1+xt)/(1-xt))*/

      /* error estimate: see algorithms.tex */
      /* FIXME: this does not correspond to the value in algorithms.tex!!! */
      /* err=Nt-__gmpfr_ceil_log2(1+5*pow(2,1-MPFR_EXP(t)));*/
      err = Nt - (MAX (4 - MPFR_GET_EXP (t), 0) + 1);

      if (MPFR_LIKELY (MPFR_IS_ZERO (t)
                       || MPFR_CAN_ROUND (t, err, Ny, rnd_mode)))
        break;

      /* reactualisation of the precision */
      MPFR_ZIV_NEXT (loop, Nt);
      mpfr_set_prec (t, Nt);
      mpfr_set_prec (te, Nt);
    }
  MPFR_ZIV_FREE (loop);

  inexact = mpfr_set4 (y, t, rnd_mode, MPFR_SIGN (xt));

  mpfr_clear(t);
  mpfr_clear(te);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, inexact, rnd_mode);
}

