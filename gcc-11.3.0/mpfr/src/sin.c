/* mpfr_sin -- sine of a floating-point number

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

static int
mpfr_sin_fast (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  int inex;

  inex = mpfr_sincos_fast (y, NULL, x, rnd_mode);
  inex = inex & 3; /* 0: exact, 1: rounded up, 2: rounded down */
  return (inex == 2) ? -1 : inex;
}

int
mpfr_sin (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_t c, xr;
  mpfr_srcptr xx;
  mpfr_exp_t expx, err;
  mpfr_prec_t precy, m;
  int inexact, sign, reduce;
  MPFR_ZIV_DECL (loop);
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd_mode),
     ("y[%Pu]=%.*Rg inexact=%d", mpfr_get_prec (y), mpfr_log_prec, y,
      inexact));

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x) || MPFR_IS_INF (x))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;

        }
      else /* x is zero */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (x));
          MPFR_SET_ZERO (y);
          MPFR_SET_SAME_SIGN (y, x);
          MPFR_RET (0);
        }
    }

  /* sin(x) = x - x^3/6 + ... so the error is < 2^(3*EXP(x)-2) */
  MPFR_FAST_COMPUTE_IF_SMALL_INPUT (y, x, -2 * MPFR_GET_EXP (x), 2, 0,
                                    rnd_mode, {});

  MPFR_SAVE_EXPO_MARK (expo);

  /* Compute initial precision */
  precy = MPFR_PREC (y);

  if (precy >= MPFR_SINCOS_THRESHOLD)
    {
      inexact = mpfr_sin_fast (y, x, rnd_mode);
      goto end;
    }

  m = precy + MPFR_INT_CEIL_LOG2 (precy) + 13;
  expx = MPFR_GET_EXP (x);

  mpfr_init (c);
  mpfr_init (xr);

  MPFR_ZIV_INIT (loop, m);
  for (;;)
    {
      /* first perform argument reduction modulo 2*Pi (if needed),
         also helps to determine the sign of sin(x) */
      if (expx >= 2) /* If Pi < x < 4, we need to reduce too, to determine
                        the sign of sin(x). For 2 <= |x| < Pi, we could avoid
                        the reduction. */
        {
          reduce = 1;
          /* As expx + m - 1 will silently be converted into mpfr_prec_t
             in the mpfr_set_prec call, the assert below may be useful to
             avoid undefined behavior. */
          MPFR_ASSERTN (expx + m - 1 <= MPFR_PREC_MAX);
          mpfr_set_prec (c, expx + m - 1);
          mpfr_set_prec (xr, m);
          mpfr_const_pi (c, MPFR_RNDN);
          mpfr_mul_2ui (c, c, 1, MPFR_RNDN);
          mpfr_remainder (xr, x, c, MPFR_RNDN);
          /* The analysis is similar to that of cos.c:
             |xr - x - 2kPi| <= 2^(2-m). Thus we can decide the sign
             of sin(x) if xr is at distance at least 2^(2-m) of both
             0 and +/-Pi. */
          mpfr_div_2ui (c, c, 1, MPFR_RNDN);
          /* Since c approximates Pi with an error <= 2^(2-expx-m) <= 2^(-m),
             it suffices to check that c - |xr| >= 2^(2-m). */
          if (MPFR_SIGN (xr) > 0)
            mpfr_sub (c, c, xr, MPFR_RNDZ);
          else
            mpfr_add (c, c, xr, MPFR_RNDZ);
          if (MPFR_IS_ZERO(xr)
              || MPFR_GET_EXP(xr) < (mpfr_exp_t) 3 - (mpfr_exp_t) m
              || MPFR_IS_ZERO(c)
              || MPFR_GET_EXP(c) < (mpfr_exp_t) 3 - (mpfr_exp_t) m)
            goto ziv_next;

          /* |xr - x - 2kPi| <= 2^(2-m), thus |sin(xr) - sin(x)| <= 2^(2-m) */
          xx = xr;
        }
      else /* the input argument is already reduced */
        {
          reduce = 0;
          xx = x;
        }

      sign = MPFR_SIGN(xx);
      /* now that the argument is reduced, precision m is enough */
      mpfr_set_prec (c, m);
      mpfr_cos (c, xx, MPFR_RNDZ);    /* can't be exact */
      mpfr_nexttoinf (c);           /* now c = cos(x) rounded away */
      mpfr_mul (c, c, c, MPFR_RNDU); /* away */
      mpfr_ui_sub (c, 1, c, MPFR_RNDZ);
      mpfr_sqrt (c, c, MPFR_RNDZ);
      if (MPFR_IS_NEG_SIGN(sign))
        MPFR_CHANGE_SIGN(c);

      /* Warning: c may be 0! */
      if (MPFR_UNLIKELY (MPFR_IS_ZERO (c)))
        {
          /* Huge cancellation: increase prec a lot! */
          m = MAX (m, MPFR_PREC (x));
          m = 2 * m;
        }
      else
        {
          /* the absolute error on c is at most 2^(3-m-EXP(c)),
             plus 2^(2-m) if there was an argument reduction.
             Since EXP(c) <= 1, 3-m-EXP(c) >= 2-m, thus the error
             is at most 2^(3-m-EXP(c)) in case of argument reduction. */
          err = 2 * MPFR_GET_EXP (c) + (mpfr_exp_t) m - 3 - (reduce != 0);
          if (MPFR_CAN_ROUND (c, err, precy, rnd_mode))
            break;

          /* check for huge cancellation (Near 0) */
          if (err < (mpfr_exp_t) MPFR_PREC (y))
            m += MPFR_PREC (y) - err;
          /* Check if near 1 */
          if (MPFR_GET_EXP (c) == 1)
            m += m;
        }

    ziv_next:
      /* Else generic increase */
      MPFR_ZIV_NEXT (loop, m);
    }
  MPFR_ZIV_FREE (loop);

  inexact = mpfr_set (y, c, rnd_mode);
  /* inexact cannot be 0, since this would mean that c was representable
     within the target precision, but in that case mpfr_can_round will fail */

  mpfr_clear (c);
  mpfr_clear (xr);

 end:
  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, inexact, rnd_mode);
}
