/* mpfr_csc - cosecant function.

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

/* the cosecant is defined by csc(x) = 1/sin(x).
   csc (NaN) = NaN.
   csc (+Inf) = csc (-Inf) = NaN.
   csc (+0) = +Inf.
   csc (-0) = -Inf.
*/

#define FUNCTION mpfr_csc
#define INVERSE  mpfr_sin
#define ACTION_NAN(y) do { MPFR_SET_NAN(y); MPFR_RET_NAN; } while (1)
#define ACTION_INF(y) do { MPFR_SET_NAN(y); MPFR_RET_NAN; } while (1)
#define ACTION_ZERO(y,x) do { MPFR_SET_SAME_SIGN(y,x); MPFR_SET_INF(y); \
                              mpfr_set_divby0 (); MPFR_RET(0); } while (1)
/* near x=0, we have csc(x) = 1/x + x/6 + ..., more precisely we have
   |csc(x) - 1/x| <= 0.2 for |x| <= 1. The analysis is similar to that for
   gamma(x) near x=0 (see gamma.c), except here the error term has the same
   sign as 1/x, thus |csc(x)| >= |1/x|. Then:
   (i) either x is a power of two, then 1/x is exactly representable, and
       as long as 1/2*ulp(1/x) > 0.2, we can conclude;
   (ii) otherwise assume x has <= n bits, and y has <= n+1 bits, then
   |y - 1/x| >= 2^(-2n) ufp(y), where ufp means unit in first place.
   Since |csc(x) - 1/x| <= 0.2, if 2^(-2n) ufp(y) >= 0.4, then
   |y - csc(x)| >= 2^(-2n-1) ufp(y), and rounding 1/x gives the correct result.
   If x < 2^E, then y > 2^(-E), thus ufp(y) > 2^(-E-1).
   A sufficient condition is thus EXP(x) <= -2 MAX(PREC(x),PREC(Y)). */
#define ACTION_TINY(y,x,r) \
  if (MPFR_EXP(x) <= -2 * (mpfr_exp_t) MAX(MPFR_PREC(x), MPFR_PREC(y))) \
    {                                                                   \
      int signx = MPFR_SIGN(x);                                         \
      inexact = mpfr_ui_div (y, 1, x, r);                               \
      if (inexact == 0) /* x is a power of two */                       \
        { /* result always 1/x, except when rounding away from zero */  \
          if (rnd_mode == MPFR_RNDA)                                    \
            rnd_mode = (signx > 0) ? MPFR_RNDU : MPFR_RNDD;             \
          if (rnd_mode == MPFR_RNDU)                                    \
            {                                                           \
              if (signx > 0)                                            \
                mpfr_nextabove (y); /* 2^k + epsilon */                 \
              inexact = 1;                                              \
            }                                                           \
          else if (rnd_mode == MPFR_RNDD)                               \
            {                                                           \
              if (signx < 0)                                            \
                mpfr_nextbelow (y); /* -2^k - epsilon */                \
              inexact = -1;                                             \
            }                                                           \
          else /* round to zero, or nearest */                          \
            inexact = -signx;                                           \
        }                                                               \
      MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);                \
      goto end;                                                         \
    }

#include "gen_inverse.h"
