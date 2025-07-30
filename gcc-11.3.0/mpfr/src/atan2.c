/* mpfr_atan2 -- arc-tan 2 of a floating-point number

Copyright 2005-2017 Free Software Foundation, Inc.
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
pi_div_2ui (mpfr_ptr dest, int i, int neg, mpfr_rnd_t rnd_mode)
{
  int inexact;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_SAVE_EXPO_MARK (expo);
  if (neg) /* -PI/2^i */
    {
      inexact = - mpfr_const_pi (dest, MPFR_INVERT_RND (rnd_mode));
      MPFR_CHANGE_SIGN (dest);
    }
  else /* PI/2^i */
    {
      inexact = mpfr_const_pi (dest, rnd_mode);
    }
  mpfr_div_2ui (dest, dest, i, rnd_mode);  /* exact */
  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (dest, inexact, rnd_mode);
}

int
mpfr_atan2 (mpfr_ptr dest, mpfr_srcptr y, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_t tmp, pi;
  int inexact;
  mpfr_prec_t prec;
  mpfr_exp_t e;
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_ZIV_DECL (loop);

  MPFR_LOG_FUNC
    (("y[%Pu]=%.*Rg x[%Pu]=%.*Rg rnd=%d",
      mpfr_get_prec (y), mpfr_log_prec, y,
      mpfr_get_prec (x), mpfr_log_prec, x, rnd_mode),
     ("atan[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (dest), mpfr_log_prec, dest, inexact));

  /* Special cases */
  if (MPFR_ARE_SINGULAR (x, y))
    {
      /* atan2(0, 0) does not raise the "invalid" floating-point
         exception, nor does atan2(y, 0) raise the "divide-by-zero"
         floating-point exception.
         -- atan2(±0, -0) returns ±pi.313)
         -- atan2(±0, +0) returns ±0.
         -- atan2(±0, x) returns ±pi, for x < 0.
         -- atan2(±0, x) returns ±0, for x > 0.
         -- atan2(y, ±0) returns -pi/2 for y < 0.
         -- atan2(y, ±0) returns pi/2 for y > 0.
         -- atan2(±oo, -oo) returns ±3pi/4.
         -- atan2(±oo, +oo) returns ±pi/4.
         -- atan2(±oo, x) returns ±pi/2, for finite x.
         -- atan2(±y, -oo) returns ±pi, for finite y > 0.
         -- atan2(±y, +oo) returns ±0, for finite y > 0.
      */
      if (MPFR_IS_NAN (x) || MPFR_IS_NAN (y))
        {
          MPFR_SET_NAN (dest);
          MPFR_RET_NAN;
        }
      if (MPFR_IS_ZERO (y))
        {
          if (MPFR_IS_NEG (x)) /* +/- PI */
            {
            set_pi:
              if (MPFR_IS_NEG (y))
                {
                  inexact =  mpfr_const_pi (dest, MPFR_INVERT_RND (rnd_mode));
                  MPFR_CHANGE_SIGN (dest);
                  return -inexact;
                }
              else
                return mpfr_const_pi (dest, rnd_mode);
            }
          else /* +/- 0 */
            {
            set_zero:
              MPFR_SET_ZERO (dest);
              MPFR_SET_SAME_SIGN (dest, y);
              return 0;
            }
        }
      if (MPFR_IS_ZERO (x))
        {
          return pi_div_2ui (dest, 1, MPFR_IS_NEG (y), rnd_mode);
        }
      if (MPFR_IS_INF (y))
        {
          if (!MPFR_IS_INF (x)) /* +/- PI/2 */
            return pi_div_2ui (dest, 1, MPFR_IS_NEG (y), rnd_mode);
          else if (MPFR_IS_POS (x)) /* +/- PI/4 */
            return pi_div_2ui (dest, 2, MPFR_IS_NEG (y), rnd_mode);
          else /* +/- 3*PI/4: Ugly since we have to round properly */
            {
              mpfr_t tmp2;
              MPFR_ZIV_DECL (loop2);
              mpfr_prec_t prec2 = MPFR_PREC (dest) + 10;

              MPFR_SAVE_EXPO_MARK (expo);
              mpfr_init2 (tmp2, prec2);
              MPFR_ZIV_INIT (loop2, prec2);
              for (;;)
                {
                  mpfr_const_pi (tmp2, MPFR_RNDN);
                  mpfr_mul_ui (tmp2, tmp2, 3, MPFR_RNDN); /* Error <= 2  */
                  mpfr_div_2ui (tmp2, tmp2, 2, MPFR_RNDN);
                  if (mpfr_round_p (MPFR_MANT (tmp2), MPFR_LIMB_SIZE (tmp2),
                                    MPFR_PREC (tmp2) - 2,
                                    MPFR_PREC (dest) + (rnd_mode == MPFR_RNDN)))
                    break;
                  MPFR_ZIV_NEXT (loop2, prec2);
                  mpfr_set_prec (tmp2, prec2);
                }
              MPFR_ZIV_FREE (loop2);
              if (MPFR_IS_NEG (y))
                MPFR_CHANGE_SIGN (tmp2);
              inexact = mpfr_set (dest, tmp2, rnd_mode);
              mpfr_clear (tmp2);
              MPFR_SAVE_EXPO_FREE (expo);
              return mpfr_check_range (dest, inexact, rnd_mode);
            }
        }
      MPFR_ASSERTD (MPFR_IS_INF (x));
      if (MPFR_IS_NEG (x))
        goto set_pi;
      else
        goto set_zero;
    }

  /* When x is a power of two, we call directly atan(y/x) since y/x is
     exact. */
  if (MPFR_UNLIKELY (MPFR_IS_POWER_OF_2 (x)))
    {
      int r;
      mpfr_t yoverx;
      unsigned int saved_flags = __gmpfr_flags;

      mpfr_init2 (yoverx, MPFR_PREC (y));
      if (MPFR_LIKELY (mpfr_div_2si (yoverx, y, MPFR_GET_EXP (x) - 1,
                                     MPFR_RNDN) == 0))
        {
          /* Here the flags have not changed due to mpfr_div_2si. */
          r = mpfr_atan (dest, yoverx, rnd_mode);
          mpfr_clear (yoverx);
          return r;
        }
      else
        {
          /* Division is inexact because of a small exponent range */
          mpfr_clear (yoverx);
          __gmpfr_flags = saved_flags;
        }
    }

  MPFR_SAVE_EXPO_MARK (expo);

  /* Set up initial prec */
  prec = MPFR_PREC (dest) + 3 + MPFR_INT_CEIL_LOG2 (MPFR_PREC (dest));
  mpfr_init2 (tmp, prec);

  MPFR_ZIV_INIT (loop, prec);
  if (MPFR_IS_POS (x))
    /* use atan2(y,x) = atan(y/x) */
    for (;;)
      {
        int div_inex;
        MPFR_BLOCK_DECL (flags);

        MPFR_BLOCK (flags, div_inex = mpfr_div (tmp, y, x, MPFR_RNDN));
        if (div_inex == 0)
          {
            /* Result is exact. */
            inexact = mpfr_atan (dest, tmp, rnd_mode);
            goto end;
          }

        /* Error <= ulp (tmp) except in case of underflow or overflow. */

        /* If the division underflowed, since |atan(z)/z| < 1, we have
           an underflow. */
        if (MPFR_UNDERFLOW (flags))
          {
            int sign;

            /* In the case MPFR_RNDN with 2^(emin-2) < |y/x| < 2^(emin-1):
               The smallest significand value S > 1 of |y/x| is:
                 * 1 / (1 - 2^(-px))                        if py <= px,
                 * (1 - 2^(-px) + 2^(-py)) / (1 - 2^(-px))  if py >= px.
               Therefore S - 1 > 2^(-pz), where pz = max(px,py). We have:
               atan(|y/x|) > atan(z), where z = 2^(emin-2) * (1 + 2^(-pz)).
                           > z - z^3 / 3.
                           > 2^(emin-2) * (1 + 2^(-pz) - 2^(2 emin - 5))
               Assuming pz <= -2 emin + 5, we can round away from zero
               (this is what mpfr_underflow always does on MPFR_RNDN).
               In the case MPFR_RNDN with |y/x| <= 2^(emin-2), we round
               toward zero, as |atan(z)/z| < 1. */
            MPFR_ASSERTN (MPFR_PREC_MAX <=
                          2 * (mpfr_uexp_t) - MPFR_EMIN_MIN + 5);
            if (rnd_mode == MPFR_RNDN && MPFR_IS_ZERO (tmp))
              rnd_mode = MPFR_RNDZ;
            sign = MPFR_SIGN (tmp);
            mpfr_clear (tmp);
            MPFR_SAVE_EXPO_FREE (expo);
            return mpfr_underflow (dest, rnd_mode, sign);
          }

        mpfr_atan (tmp, tmp, MPFR_RNDN);   /* Error <= 2*ulp (tmp) since
                                             abs(D(arctan)) <= 1 */
        /* TODO: check that the error bound is correct in case of overflow. */
        /* FIXME: Error <= ulp(tmp) ? */
        if (MPFR_LIKELY (MPFR_CAN_ROUND (tmp, prec - 2, MPFR_PREC (dest),
                                         rnd_mode)))
          break;
        MPFR_ZIV_NEXT (loop, prec);
        mpfr_set_prec (tmp, prec);
      }
  else /* x < 0 */
    /*  Use sign(y)*(PI - atan (|y/x|)) */
    {
      mpfr_init2 (pi, prec);
      for (;;)
        {
          mpfr_div (tmp, y, x, MPFR_RNDN);   /* Error <= ulp (tmp) */
          /* If tmp is 0, we have |y/x| <= 2^(-emin-2), thus
             atan|y/x| < 2^(-emin-2). */
          MPFR_SET_POS (tmp);               /* no error */
          mpfr_atan (tmp, tmp, MPFR_RNDN);   /* Error <= 2*ulp (tmp) since
                                               abs(D(arctan)) <= 1 */
          mpfr_const_pi (pi, MPFR_RNDN);     /* Error <= ulp(pi) /2 */
          e = MPFR_NOTZERO(tmp) ? MPFR_GET_EXP (tmp) : __gmpfr_emin - 1;
          mpfr_sub (tmp, pi, tmp, MPFR_RNDN);          /* see above */
          if (MPFR_IS_NEG (y))
            MPFR_CHANGE_SIGN (tmp);
          /* Error(tmp) <= (1/2+2^(EXP(pi)-EXP(tmp)-1)+2^(e-EXP(tmp)+1))*ulp
                        <= 2^(MAX (MAX (EXP(PI)-EXP(tmp)-1, e-EXP(tmp)+1),
                                        -1)+2)*ulp(tmp) */
          e = MAX (MAX (MPFR_GET_EXP (pi)-MPFR_GET_EXP (tmp) - 1,
                        e - MPFR_GET_EXP (tmp) + 1), -1) + 2;
          if (MPFR_LIKELY (MPFR_CAN_ROUND (tmp, prec - e, MPFR_PREC (dest),
                                           rnd_mode)))
            break;
          MPFR_ZIV_NEXT (loop, prec);
          mpfr_set_prec (tmp, prec);
          mpfr_set_prec (pi, prec);
        }
      mpfr_clear (pi);
    }
  inexact = mpfr_set (dest, tmp, rnd_mode);

 end:
  MPFR_ZIV_FREE (loop);
  mpfr_clear (tmp);
  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (dest, inexact, rnd_mode);
}
