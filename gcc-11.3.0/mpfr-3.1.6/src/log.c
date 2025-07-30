/* mpfr_log -- natural logarithm of a floating-point number

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

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

/* The computation of log(x) is done using the formula :
     if we want p bits of the result,

                       pi
          log(x) ~ ------------  -   m log 2
                    2 AG(1,4/s)

     where s = x 2^m > 2^(p/2)

     More precisely, if F(x) = int(1/sqrt(1-(1-x^2)*sin(t)^2), t=0..PI/2),
     then for s>=1.26 we have log(s) < F(4/s) < log(s)*(1+4/s^2)
     from which we deduce pi/2/AG(1,4/s)*(1-4/s^2) < log(s) < pi/2/AG(1,4/s)
     so the relative error 4/s^2 is < 4/2^p i.e. 4 ulps.
*/

int
mpfr_log (mpfr_ptr r, mpfr_srcptr a, mpfr_rnd_t rnd_mode)
{
  int inexact;
  mpfr_prec_t p, q;
  mpfr_t tmp1, tmp2;
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_ZIV_DECL (loop);
  MPFR_GROUP_DECL(group);

  MPFR_LOG_FUNC
    (("a[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (a), mpfr_log_prec, a, rnd_mode),
     ("r[%Pu]=%.*Rg inexact=%d", mpfr_get_prec (r), mpfr_log_prec, r,
      inexact));

  /* Special cases */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (a)))
    {
      /* If a is NaN, the result is NaN */
      if (MPFR_IS_NAN (a))
        {
          MPFR_SET_NAN (r);
          MPFR_RET_NAN;
        }
      /* check for infinity before zero */
      else if (MPFR_IS_INF (a))
        {
          if (MPFR_IS_NEG (a))
            /* log(-Inf) = NaN */
            {
              MPFR_SET_NAN (r);
              MPFR_RET_NAN;
            }
          else /* log(+Inf) = +Inf */
            {
              MPFR_SET_INF (r);
              MPFR_SET_POS (r);
              MPFR_RET (0);
            }
        }
      else /* a is zero */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (a));
          MPFR_SET_INF (r);
          MPFR_SET_NEG (r);
          mpfr_set_divby0 ();
          MPFR_RET (0); /* log(0) is an exact -infinity */
        }
    }
  /* If a is negative, the result is NaN */
  else if (MPFR_UNLIKELY (MPFR_IS_NEG (a)))
    {
      MPFR_SET_NAN (r);
      MPFR_RET_NAN;
    }
  /* If a is 1, the result is 0 */
  else if (MPFR_UNLIKELY (MPFR_GET_EXP (a) == 1 && mpfr_cmp_ui (a, 1) == 0))
    {
      MPFR_SET_ZERO (r);
      MPFR_SET_POS (r);
      MPFR_RET (0); /* only "normal" case where the result is exact */
    }

  q = MPFR_PREC (r);

  /* use initial precision about q+lg(q)+5 */
  p = q + 5 + 2 * MPFR_INT_CEIL_LOG2 (q);
  /* % ~(mpfr_prec_t)GMP_NUMB_BITS  ;
     m=q; while (m) { p++; m >>= 1; }  */
  /* if (MPFR_LIKELY(p % GMP_NUMB_BITS != 0))
      p += GMP_NUMB_BITS - (p%GMP_NUMB_BITS); */

  MPFR_SAVE_EXPO_MARK (expo);
  MPFR_GROUP_INIT_2 (group, p, tmp1, tmp2);

  MPFR_ZIV_INIT (loop, p);
  for (;;)
    {
      long m;
      mpfr_exp_t cancel;

      /* Calculus of m (depends on p) */
      m = (p + 1) / 2 - MPFR_GET_EXP (a) + 1;

      mpfr_mul_2si (tmp2, a, m, MPFR_RNDN);    /* s=a*2^m,        err<=1 ulp  */
      mpfr_div (tmp1, __gmpfr_four, tmp2, MPFR_RNDN);/* 4/s,      err<=2 ulps */
      mpfr_agm (tmp2, __gmpfr_one, tmp1, MPFR_RNDN); /* AG(1,4/s),err<=3 ulps */
      mpfr_mul_2ui (tmp2, tmp2, 1, MPFR_RNDN); /* 2*AG(1,4/s),    err<=3 ulps */
      mpfr_const_pi (tmp1, MPFR_RNDN);         /* compute pi,     err<=1ulp   */
      mpfr_div (tmp2, tmp1, tmp2, MPFR_RNDN);  /* pi/2*AG(1,4/s), err<=5ulps  */
      mpfr_const_log2 (tmp1, MPFR_RNDN);      /* compute log(2),  err<=1ulp   */
      mpfr_mul_si (tmp1, tmp1, m, MPFR_RNDN); /* compute m*log(2),err<=2ulps  */
      mpfr_sub (tmp1, tmp2, tmp1, MPFR_RNDN); /* log(a),    err<=7ulps+cancel */

      if (MPFR_LIKELY (MPFR_IS_PURE_FP (tmp1) && MPFR_IS_PURE_FP (tmp2)))
        {
          cancel = MPFR_GET_EXP (tmp2) - MPFR_GET_EXP (tmp1);
          MPFR_LOG_MSG (("canceled bits=%ld\n", (long) cancel));
          MPFR_LOG_VAR (tmp1);
          if (MPFR_UNLIKELY (cancel < 0))
            cancel = 0;

          /* we have 7 ulps of error from the above roundings,
             4 ulps from the 4/s^2 second order term,
             plus the canceled bits */
          if (MPFR_LIKELY (MPFR_CAN_ROUND (tmp1, p-cancel-4, q, rnd_mode)))
            break;

          /* VL: I think it is better to have an increment that it isn't
             too low; in particular, the increment must be positive even
             if cancel = 0 (can this occur?). */
          p += cancel >= 8 ? cancel : 8;
        }
      else
        {
          /* TODO: find why this case can occur and what is best to do
             with it. */
          p += 32;
        }

      MPFR_ZIV_NEXT (loop, p);
      MPFR_GROUP_REPREC_2 (group, p, tmp1, tmp2);
    }
  MPFR_ZIV_FREE (loop);
  inexact = mpfr_set (r, tmp1, rnd_mode);
  /* We clean */
  MPFR_GROUP_CLEAR (group);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (r, inexact, rnd_mode);
}
