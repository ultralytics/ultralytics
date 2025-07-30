/* mpfr_expm1 -- Compute exp(x)-1

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

 /* The computation of expm1 is done by
    expm1(x)=exp(x)-1
 */

int
mpfr_expm1 (mpfr_ptr y, mpfr_srcptr x , mpfr_rnd_t rnd_mode)
{
  int inexact;
  mpfr_exp_t ex;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd_mode),
     ("y[%Pu]=%.*Rg inexact=%d", mpfr_get_prec (y), mpfr_log_prec, y,
      inexact));

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      /* check for inf or -inf (expm1(-inf)=-1) */
      else if (MPFR_IS_INF (x))
        {
          if (MPFR_IS_POS (x))
            {
              MPFR_SET_INF (y);
              MPFR_SET_POS (y);
              MPFR_RET (0);
            }
          else
            return mpfr_set_si (y, -1, rnd_mode);
        }
      else
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (x));
          MPFR_SET_ZERO (y);   /* expm1(+/- 0) = +/- 0 */
          MPFR_SET_SAME_SIGN (y, x);
          MPFR_RET (0);
        }
    }

  ex = MPFR_GET_EXP (x);
  if (ex < 0)
    {
      /* For -1 < x < 0, abs(expm1(x)-x) < x^2/2.
         For 0 < x < 1,  abs(expm1(x)-x) < x^2. */
      if (MPFR_IS_POS (x))
        MPFR_FAST_COMPUTE_IF_SMALL_INPUT (y, x, - ex, 0, 1, rnd_mode, {});
      else
        MPFR_FAST_COMPUTE_IF_SMALL_INPUT (y, x, - ex, 1, 0, rnd_mode, {});
    }

  MPFR_SAVE_EXPO_MARK (expo);

  if (MPFR_IS_NEG (x) && ex > 5)  /* x <= -32 */
    {
      mpfr_t minus_one, t;
      mpfr_exp_t err;

      mpfr_init2 (minus_one, 2);
      mpfr_init2 (t, 64);
      mpfr_set_si (minus_one, -1, MPFR_RNDN);
      mpfr_const_log2 (t, MPFR_RNDU); /* round upward since x is negative */
      mpfr_div (t, x, t, MPFR_RNDU); /* > x / ln(2) */
      err = mpfr_cmp_si (t, MPFR_EMIN_MIN >= -LONG_MAX ?
                         MPFR_EMIN_MIN : -LONG_MAX) <= 0 ?
        - (MPFR_EMIN_MIN >= -LONG_MAX ? MPFR_EMIN_MIN : -LONG_MAX) :
        - mpfr_get_si (t, MPFR_RNDU);
      /* exp(x) = 2^(x/ln(2))
               <= 2^max(MPFR_EMIN_MIN,-LONG_MAX,ceil(x/ln(2)+epsilon))
         with epsilon > 0 */
      mpfr_clear (t);
      MPFR_SMALL_INPUT_AFTER_SAVE_EXPO (y, minus_one, err, 0, 0, rnd_mode,
                                        expo, { mpfr_clear (minus_one); });
      mpfr_clear (minus_one);
    }

  /* General case */
  {
    /* Declaration of the intermediary variable */
    mpfr_t t;
    /* Declaration of the size variable */
    mpfr_prec_t Ny = MPFR_PREC(y);   /* target precision */
    mpfr_prec_t Nt;                  /* working precision */
    mpfr_exp_t err, exp_te;          /* error */
    MPFR_ZIV_DECL (loop);

    /* compute the precision of intermediary variable */
    /* the optimal number of bits : see algorithms.tex */
    Nt = Ny + MPFR_INT_CEIL_LOG2 (Ny) + 6;

    /* if |x| is smaller than 2^(-e), we will loose about e bits in the
       subtraction exp(x) - 1 */
    if (ex < 0)
      Nt += - ex;

    /* initialize auxiliary variable */
    mpfr_init2 (t, Nt);

    /* First computation of expm1 */
    MPFR_ZIV_INIT (loop, Nt);
    for (;;)
      {
        MPFR_BLOCK_DECL (flags);

        /* exp(x) may overflow and underflow */
        MPFR_BLOCK (flags, mpfr_exp (t, x, MPFR_RNDN));
        if (MPFR_OVERFLOW (flags))
          {
            inexact = mpfr_overflow (y, rnd_mode, MPFR_SIGN_POS);
            MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, MPFR_FLAGS_OVERFLOW);
            break;
          }
        else if (MPFR_UNDERFLOW (flags))
          {
            inexact = mpfr_set_si (y, -1, rnd_mode);
            MPFR_ASSERTD (inexact == 0);
            inexact = -1;
            if (MPFR_IS_LIKE_RNDZ (rnd_mode, 1))
              {
                inexact = 1;
                mpfr_nexttozero (y);
              }
            break;
          }

        exp_te = MPFR_GET_EXP (t);         /* FIXME: exp(x) may overflow! */
        mpfr_sub_ui (t, t, 1, MPFR_RNDN);   /* exp(x)-1 */

        /* error estimate */
        /*err=Nt-(__gmpfr_ceil_log2(1+pow(2,MPFR_EXP(te)-MPFR_EXP(t))));*/
        err = Nt - (MAX (exp_te - MPFR_GET_EXP (t), 0) + 1);

        if (MPFR_LIKELY (MPFR_CAN_ROUND (t, err, Ny, rnd_mode)))
          {
            inexact = mpfr_set (y, t, rnd_mode);
            break;
          }

        /* increase the precision */
        MPFR_ZIV_NEXT (loop, Nt);
        mpfr_set_prec (t, Nt);
      }
    MPFR_ZIV_FREE (loop);

    mpfr_clear (t);
  }

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, inexact, rnd_mode);
}
