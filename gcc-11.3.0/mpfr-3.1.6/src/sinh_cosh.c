/* mpfr_sinh_cosh -- hyperbolic sine and cosine

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

#define INEXPOS(y) ((y) == 0 ? 0 : (((y) > 0) ? 1 : 2))
#define INEX(y,z) (INEXPOS(y) | (INEXPOS(z) << 2))

 /* The computations are done by
    cosh(x) = 1/2 [e^(x)+e^(-x)]
    sinh(x) = 1/2 [e^(x)-e^(-x)]
    Adapted from mpfr_sinh.c     */

int
mpfr_sinh_cosh (mpfr_ptr sh, mpfr_ptr ch, mpfr_srcptr xt, mpfr_rnd_t rnd_mode)
{
  mpfr_t x;
  int inexact_sh, inexact_ch;

  MPFR_ASSERTN (sh != ch);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d",
      mpfr_get_prec (xt), mpfr_log_prec, xt, rnd_mode),
     ("sh[%Pu]=%.*Rg ch[%Pu]=%.*Rg",
      mpfr_get_prec (sh), mpfr_log_prec, sh,
      mpfr_get_prec (ch), mpfr_log_prec, ch));

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (xt)))
    {
      if (MPFR_IS_NAN (xt))
        {
          MPFR_SET_NAN (ch);
          MPFR_SET_NAN (sh);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (xt))
        {
          MPFR_SET_INF (sh);
          MPFR_SET_SAME_SIGN (sh, xt);
          MPFR_SET_INF (ch);
          MPFR_SET_POS (ch);
          MPFR_RET (0);
        }
      else /* xt is zero */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (xt));
          MPFR_SET_ZERO (sh);                   /* sinh(0) = 0 */
          MPFR_SET_SAME_SIGN (sh, xt);
          inexact_sh = 0;
          inexact_ch = mpfr_set_ui (ch, 1, rnd_mode); /* cosh(0) = 1 */
          return INEX(inexact_sh,inexact_ch);
        }
    }

  /* Warning: if we use MPFR_FAST_COMPUTE_IF_SMALL_INPUT here, make sure
     that the code also works in case of overlap (see sin_cos.c) */

  MPFR_TMP_INIT_ABS (x, xt);

  {
    mpfr_t s, c, ti;
    mpfr_exp_t d;
    mpfr_prec_t N;    /* Precision of the intermediary variables */
    long int err;    /* Precision of error */
    MPFR_ZIV_DECL (loop);
    MPFR_SAVE_EXPO_DECL (expo);
    MPFR_GROUP_DECL (group);

    MPFR_SAVE_EXPO_MARK (expo);

    /* compute the precision of intermediary variable */
    N = MPFR_PREC (ch);
    N = MAX (N, MPFR_PREC (sh));
    /* the optimal number of bits : see algorithms.ps */
    N = N + MPFR_INT_CEIL_LOG2 (N) + 4;

    /* initialise of intermediary variables */
    MPFR_GROUP_INIT_3 (group, N, s, c, ti);

    /* First computation of sinh_cosh */
    MPFR_ZIV_INIT (loop, N);
    for (;;)
      {
        MPFR_BLOCK_DECL (flags);

        /* compute sinh_cosh */
        MPFR_BLOCK (flags, mpfr_exp (s, x, MPFR_RNDD));
        if (MPFR_OVERFLOW (flags))
          /* exp(x) does overflow */
          {
            /* since cosh(x) >= exp(x), cosh(x) overflows too */
            inexact_ch = mpfr_overflow (ch, rnd_mode, MPFR_SIGN_POS);
            /* sinh(x) may be representable */
            inexact_sh = mpfr_sinh (sh, xt, rnd_mode);
            MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, MPFR_FLAGS_OVERFLOW);
            break;
          }
        d = MPFR_GET_EXP (s);
        mpfr_ui_div (ti, 1, s, MPFR_RNDU);  /* 1/exp(x) */
        mpfr_add (c, s, ti, MPFR_RNDU);     /* exp(x) + 1/exp(x) */
        mpfr_sub (s, s, ti, MPFR_RNDN);     /* exp(x) - 1/exp(x) */
        mpfr_div_2ui (c, c, 1, MPFR_RNDN);  /* 1/2(exp(x) + 1/exp(x)) */
        mpfr_div_2ui (s, s, 1, MPFR_RNDN);  /* 1/2(exp(x) - 1/exp(x)) */

        /* it may be that s is zero (in fact, it can only occur when exp(x)=1,
           and thus ti=1 too) */
        if (MPFR_IS_ZERO (s))
          err = N; /* double the precision */
        else
          {
            /* calculation of the error */
            d = d - MPFR_GET_EXP (s) + 2;
            /* error estimate: err = N-(__gmpfr_ceil_log2(1+pow(2,d)));*/
            err = N - (MAX (d, 0) + 1);
            if (MPFR_LIKELY (MPFR_CAN_ROUND (s, err, MPFR_PREC (sh),
                                             rnd_mode) &&               \
                             MPFR_CAN_ROUND (c, err, MPFR_PREC (ch),
                                             rnd_mode)))
              {
                inexact_sh = mpfr_set4 (sh, s, rnd_mode, MPFR_SIGN (xt));
                inexact_ch = mpfr_set (ch, c, rnd_mode);
                break;
              }
          }
        /* actualisation of the precision */
        N += err;
        MPFR_ZIV_NEXT (loop, N);
        MPFR_GROUP_REPREC_3 (group, N, s, c, ti);
      }
    MPFR_ZIV_FREE (loop);
    MPFR_GROUP_CLEAR (group);
    MPFR_SAVE_EXPO_FREE (expo);
  }

  /* now, let's raise the flags if needed */
  inexact_sh = mpfr_check_range (sh, inexact_sh, rnd_mode);
  inexact_ch = mpfr_check_range (ch, inexact_ch, rnd_mode);

  return INEX(inexact_sh,inexact_ch);
}
