/* mpfr_exp -- exponential of a floating-point number

Copyright 1999-2017 Free Software Foundation, Inc.
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

/* #define DEBUG */

int
mpfr_exp (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_exp_t expx;
  mpfr_prec_t precy;
  int inexact;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd_mode),
     ("y[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (y), mpfr_log_prec, y, inexact));

  if (MPFR_UNLIKELY( MPFR_IS_SINGULAR(x) ))
    {
      if (MPFR_IS_NAN(x))
        {
          MPFR_SET_NAN(y);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF(x))
        {
          if (MPFR_IS_POS(x))
            MPFR_SET_INF(y);
          else
            MPFR_SET_ZERO(y);
          MPFR_SET_POS(y);
          MPFR_RET(0);
        }
      else
        {
          MPFR_ASSERTD(MPFR_IS_ZERO(x));
          return mpfr_set_ui (y, 1, rnd_mode);
        }
    }

  /* First, let's detect most overflow and underflow cases. */
  {
    mpfr_t e, bound;

    /* We must extended the exponent range and save the flags now. */
    MPFR_SAVE_EXPO_MARK (expo);

    mpfr_init2 (e, sizeof (mpfr_exp_t) * CHAR_BIT);
    mpfr_init2 (bound, 32);

    inexact = mpfr_set_exp_t (e, expo.saved_emax, MPFR_RNDN);
    MPFR_ASSERTD (inexact == 0);
    mpfr_const_log2 (bound, expo.saved_emax < 0 ? MPFR_RNDD : MPFR_RNDU);
    mpfr_mul (bound, bound, e, MPFR_RNDU);
    if (MPFR_UNLIKELY (mpfr_cmp (x, bound) >= 0))
      {
        /* x > log(2^emax), thus exp(x) > 2^emax */
        mpfr_clears (e, bound, (mpfr_ptr) 0);
        MPFR_SAVE_EXPO_FREE (expo);
        return mpfr_overflow (y, rnd_mode, 1);
      }

    inexact = mpfr_set_exp_t (e, expo.saved_emin, MPFR_RNDN);
    MPFR_ASSERTD (inexact == 0);
    inexact = mpfr_sub_ui (e, e, 2, MPFR_RNDN);
    MPFR_ASSERTD (inexact == 0);
    mpfr_const_log2 (bound, expo.saved_emin < 0 ? MPFR_RNDU : MPFR_RNDD);
    mpfr_mul (bound, bound, e, MPFR_RNDD);
    if (MPFR_UNLIKELY (mpfr_cmp (x, bound) <= 0))
      {
        /* x < log(2^(emin - 2)), thus exp(x) < 2^(emin - 2) */
        mpfr_clears (e, bound, (mpfr_ptr) 0);
        MPFR_SAVE_EXPO_FREE (expo);
        return mpfr_underflow (y, rnd_mode == MPFR_RNDN ? MPFR_RNDZ : rnd_mode,
                               1);
      }

    /* Other overflow/underflow cases must be detected
       by the generic routines. */
    mpfr_clears (e, bound, (mpfr_ptr) 0);
    MPFR_SAVE_EXPO_FREE (expo);
  }

  expx  = MPFR_GET_EXP (x);
  precy = MPFR_PREC (y);

  /* if x < 2^(-precy), then exp(x) i.e. gives 1 +/- 1 ulp(1) */
  if (MPFR_UNLIKELY (expx < 0 && (mpfr_uexp_t) (-expx) > precy))
    {
      mpfr_exp_t emin = __gmpfr_emin;
      mpfr_exp_t emax = __gmpfr_emax;
      int signx = MPFR_SIGN (x);

      MPFR_SET_POS (y);
      if (MPFR_IS_NEG_SIGN (signx) && (rnd_mode == MPFR_RNDD ||
                                       rnd_mode == MPFR_RNDZ))
        {
          __gmpfr_emin = 0;
          __gmpfr_emax = 0;
          mpfr_setmax (y, 0);  /* y = 1 - epsilon */
          inexact = -1;
        }
      else
        {
          __gmpfr_emin = 1;
          __gmpfr_emax = 1;
          mpfr_setmin (y, 1);  /* y = 1 */
          if (MPFR_IS_POS_SIGN (signx) && (rnd_mode == MPFR_RNDU ||
                                           rnd_mode == MPFR_RNDA))
            {
              mp_size_t yn;
              int sh;

              yn = MPFR_LIMB_SIZE (y);
              sh = (mpfr_prec_t) yn * GMP_NUMB_BITS - MPFR_PREC(y);
              MPFR_MANT(y)[0] += MPFR_LIMB_ONE << sh;
              inexact = 1;
            }
          else
            inexact = -MPFR_FROM_SIGN_TO_INT(signx);
        }

      __gmpfr_emin = emin;
      __gmpfr_emax = emax;
    }
  else  /* General case */
    {
      if (MPFR_UNLIKELY (precy >= MPFR_EXP_THRESHOLD))
        /* mpfr_exp_3 saves the exponent range and flags itself, otherwise
           the flag changes in mpfr_exp_3 are lost */
        inexact = mpfr_exp_3 (y, x, rnd_mode); /* O(M(n) log(n)^2) */
      else
        {
          MPFR_SAVE_EXPO_MARK (expo);
          inexact = mpfr_exp_2 (y, x, rnd_mode); /* O(n^(1/3) M(n)) */
          MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);
          MPFR_SAVE_EXPO_FREE (expo);
        }
    }

  return mpfr_check_range (y, inexact, rnd_mode);
}
