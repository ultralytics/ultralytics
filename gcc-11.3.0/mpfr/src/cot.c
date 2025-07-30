/* mpfr_cot - cotangent function.

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

/* the cotangent is defined by cot(x) = 1/tan(x) = cos(x)/sin(x).
   cot (NaN) = NaN.
   cot (+Inf) = csc (-Inf) = NaN.
   cot (+0) = +Inf.
   cot (-0) = -Inf.
*/

#define FUNCTION mpfr_cot
#define INVERSE  mpfr_tan
#define ACTION_NAN(y) do { MPFR_SET_NAN(y); MPFR_RET_NAN; } while (1)
#define ACTION_INF(y) do { MPFR_SET_NAN(y); MPFR_RET_NAN; } while (1)
#define ACTION_ZERO(y,x) do { MPFR_SET_SAME_SIGN(y,x); MPFR_SET_INF(y); \
                              mpfr_set_divby0 (); MPFR_RET(0); } while (1)

/* (This analysis is adapted from that for mpfr_coth.)
   Near x=0, cot(x) = 1/x - x/3 + ..., more precisely we have
   |cot(x) - 1/x| <= 0.36 for |x| <= 1. The error term has
   the opposite sign as 1/x, thus |cot(x)| <= |1/x|. Then:
   (i) either x is a power of two, then 1/x is exactly representable, and
       as long as 1/2*ulp(1/x) > 0.36, we can conclude;
   (ii) otherwise assume x has <= n bits, and y has <= n+1 bits, then
   |y - 1/x| >= 2^(-2n) ufp(y), where ufp means unit in first place.
   Since |cot(x) - 1/x| <= 0.36, if 2^(-2n) ufp(y) >= 0.72, then
   |y - cot(x)| >= 2^(-2n-1) ufp(y), and rounding 1/x gives the correct
   result. If x < 2^E, then y > 2^(-E), thus ufp(y) > 2^(-E-1).
   A sufficient condition is thus EXP(x) + 1 <= -2 MAX(PREC(x),PREC(Y)).
   The division can be inexact in case of underflow or overflow; but
   an underflow is not possible as emin = - emax. The overflow is a
   real overflow possibly except when |x| = 2^emin. */
#define ACTION_TINY(y,x,r) \
  if (MPFR_EXP(x) + 1 <= -2 * (mpfr_exp_t) MAX(MPFR_PREC(x), MPFR_PREC(y))) \
    {                                                                   \
      int two2emin;                                                     \
      int signx = MPFR_SIGN(x);                                         \
      MPFR_ASSERTN (MPFR_EMIN_MIN + MPFR_EMAX_MAX == 0);                \
      if ((two2emin = mpfr_get_exp (x) == __gmpfr_emin + 1 &&           \
           mpfr_powerof2_raw (x)))                                      \
        {                                                               \
          /* Case |x| = 2^emin. 1/x is not representable; so, compute   \
             1/(2x) instead (exact), and correct the result later. */   \
          mpfr_set_si_2exp (y, signx, __gmpfr_emax, MPFR_RNDN);         \
          inexact = 0;                                                  \
        }                                                               \
      else                                                              \
        inexact = mpfr_ui_div (y, 1, x, r);                             \
      if (inexact == 0) /* x is a power of two */                       \
        { /* result always 1/x, except when rounding to zero */         \
          if (rnd_mode == MPFR_RNDA)                                    \
            rnd_mode = (signx > 0) ? MPFR_RNDU : MPFR_RNDD;             \
          if (rnd_mode == MPFR_RNDU || (rnd_mode == MPFR_RNDZ && signx < 0)) \
            {                                                           \
              if (signx < 0)                                            \
                mpfr_nextabove (y); /* -2^k + epsilon */                \
              inexact = 1;                                              \
            }                                                           \
          else if (rnd_mode == MPFR_RNDD || rnd_mode == MPFR_RNDZ)      \
            {                                                           \
              if (signx > 0)                                            \
                mpfr_nextbelow (y); /* 2^k - epsilon */                 \
              inexact = -1;                                             \
            }                                                           \
          else /* round to nearest */                                   \
            inexact = signx;                                            \
          if (two2emin)                                                 \
            mpfr_mul_2ui (y, y, 1, r);  /* overflow in MPFR_RNDN */      \
        }                                                               \
      /* Underflow is not possible with emin = - emax, but we cannot */ \
      /* add an assert as the underflow flag could have already been */ \
      /* set before the call to mpfr_cot.                            */ \
      MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);                \
      goto end;                                                         \
    }

#include "gen_inverse.h"
