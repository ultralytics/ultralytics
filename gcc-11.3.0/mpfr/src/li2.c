/* mpfr_li2 -- Dilogarithm.

Copyright 2007-2017 Free Software Foundation, Inc.
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

/* Compute the alternating series
   s = S(z) = \sum_{k=0}^infty B_{2k} (z))^{2k+1} / (2k+1)!
   with 0 < z <= log(2) to the precision of s rounded in the direction
   rnd_mode.
   Return the maximum index of the truncature which is useful
   for determinating the relative error.
*/
static int
li2_series (mpfr_t sum, mpfr_srcptr z, mpfr_rnd_t rnd_mode)
{
  int i, Bm, Bmax;
  mpfr_t s, u, v, w;
  mpfr_prec_t sump, p;
  mpfr_exp_t se, err;
  mpz_t *B;
  MPFR_ZIV_DECL (loop);

  /* The series converges for |z| < 2 pi, but in mpfr_li2 the argument is
     reduced so that 0 < z <= log(2). Here is additionnal check that z is
     (nearly) correct */
  MPFR_ASSERTD (MPFR_IS_STRICTPOS (z));
  MPFR_ASSERTD (mpfr_cmp_d (z, 0.6953125) <= 0);

  sump = MPFR_PREC (sum);       /* target precision */
  p = sump + MPFR_INT_CEIL_LOG2 (sump) + 4;     /* the working precision */
  mpfr_init2 (s, p);
  mpfr_init2 (u, p);
  mpfr_init2 (v, p);
  mpfr_init2 (w, p);

  B = mpfr_bernoulli_internal ((mpz_t *) 0, 0);
  Bm = Bmax = 1;

  MPFR_ZIV_INIT (loop, p);
  for (;;)
    {
      mpfr_sqr (u, z, MPFR_RNDU);
      mpfr_set (v, z, MPFR_RNDU);
      mpfr_set (s, z, MPFR_RNDU);
      se = MPFR_GET_EXP (s);
      err = 0;

      for (i = 1;; i++)
        {
          if (i >= Bmax)
            B = mpfr_bernoulli_internal (B, Bmax++); /* B_2i*(2i+1)!, exact */

          mpfr_mul (v, u, v, MPFR_RNDU);
          mpfr_div_ui (v, v, 2 * i, MPFR_RNDU);
          mpfr_div_ui (v, v, 2 * i, MPFR_RNDU);
          mpfr_div_ui (v, v, 2 * i + 1, MPFR_RNDU);
          mpfr_div_ui (v, v, 2 * i + 1, MPFR_RNDU);
          /* here, v_2i = v_{2i-2} / (2i * (2i+1))^2 */

          mpfr_mul_z (w, v, B[i], MPFR_RNDN);
          /* here, w_2i = v_2i * B_2i * (2i+1)! with
             error(w_2i) < 2^(5 * i + 8) ulp(w_2i) (see algorithms.tex) */

          mpfr_add (s, s, w, MPFR_RNDN);

          err = MAX (err + se, 5 * i + 8 + MPFR_GET_EXP (w))
            - MPFR_GET_EXP (s);
          err = 2 + MAX (-1, err);
          se = MPFR_GET_EXP (s);
          if (MPFR_GET_EXP (w) <= se - (mpfr_exp_t) p)
            break;
        }

      /* the previous value of err is the rounding error,
         the truncation error is less than EXP(z) - 6 * i - 5
         (see algorithms.tex) */
      err = MAX (err, MPFR_GET_EXP (z) - 6 * i - 5) + 1;
      if (MPFR_CAN_ROUND (s, (mpfr_exp_t) p - err, sump, rnd_mode))
        break;

      MPFR_ZIV_NEXT (loop, p);
      mpfr_set_prec (s, p);
      mpfr_set_prec (u, p);
      mpfr_set_prec (v, p);
      mpfr_set_prec (w, p);
    }
  MPFR_ZIV_FREE (loop);
  mpfr_set (sum, s, rnd_mode);

  Bm = Bmax;
  while (Bm--)
    mpz_clear (B[Bm]);
  (*__gmp_free_func) (B, Bmax * sizeof (mpz_t));
  mpfr_clears (s, u, v, w, (mpfr_ptr) 0);

  /* Let K be the returned value.
     1. As we compute an alternating series, the truncation error has the same
     sign as the next term w_{K+2} which is positive iff K%4 == 0.
     2. Assume that error(z) <= (1+t) z', where z' is the actual value, then
     error(s) <= 2 * (K+1) * t (see algorithms.tex).
   */
  return 2 * i;
}

/* try asymptotic expansion when x is large and positive:
   Li2(x) = -log(x)^2/2 + Pi^2/3 - 1/x + O(1/x^2).
   More precisely for x >= 2 we have for g(x) = -log(x)^2/2 + Pi^2/3:
   -2 <= x * (Li2(x) - g(x)) <= -1
   thus |Li2(x) - g(x)| <= 2/x.
   Assumes x >= 38, which ensures log(x)^2/2 >= 2*Pi^2/3, and g(x) <= -3.3.
   Return 0 if asymptotic expansion failed (unable to round), otherwise
   returns correct ternary value.
*/
static int
mpfr_li2_asympt_pos (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_t g, h;
  mpfr_prec_t w = MPFR_PREC (y) + 20;
  int inex = 0;

  MPFR_ASSERTN (mpfr_cmp_ui (x, 38) >= 0);

  mpfr_init2 (g, w);
  mpfr_init2 (h, w);
  mpfr_log (g, x, MPFR_RNDN);    /* rel. error <= |(1 + theta) - 1| */
  mpfr_sqr (g, g, MPFR_RNDN);    /* rel. error <= |(1 + theta)^3 - 1| <= 2^(2-w) */
  mpfr_div_2ui (g, g, 1, MPFR_RNDN);     /* rel. error <= 2^(2-w) */
  mpfr_const_pi (h, MPFR_RNDN);  /* error <= 2^(1-w) */
  mpfr_sqr (h, h, MPFR_RNDN);    /* rel. error <= 2^(2-w) */
  mpfr_div_ui (h, h, 3, MPFR_RNDN);      /* rel. error <= |(1 + theta)^4 - 1|
                                           <= 5 * 2^(-w) */
  /* since x is chosen such that log(x)^2/2 >= 2 * (Pi^2/3), we should have
     g >= 2*h, thus |g-h| >= |h|, and the relative error on g is at most
     multiplied by 2 in the difference, and that by h is unchanged. */
  MPFR_ASSERTN (MPFR_EXP (g) > MPFR_EXP (h));
  mpfr_sub (g, h, g, MPFR_RNDN); /* err <= ulp(g)/2 + g*2^(3-w) + g*5*2^(-w)
                                   <= ulp(g) * (1/2 + 8 + 5) < 14 ulp(g).

                                   If in addition 2/x <= 2 ulp(g), i.e.,
                                   1/x <= ulp(g), then the total error is
                                   bounded by 16 ulp(g). */
  if ((MPFR_EXP (x) >= (mpfr_exp_t) w - MPFR_EXP (g)) &&
      MPFR_CAN_ROUND (g, w - 4, MPFR_PREC (y), rnd_mode))
    inex = mpfr_set (y, g, rnd_mode);

  mpfr_clear (g);
  mpfr_clear (h);

  return inex;
}

/* try asymptotic expansion when x is large and negative:
   Li2(x) = -log(-x)^2/2 - Pi^2/6 - 1/x + O(1/x^2).
   More precisely for x <= -2 we have for g(x) = -log(-x)^2/2 - Pi^2/6:
   |Li2(x) - g(x)| <= 1/|x|.
   Assumes x <= -7, which ensures |log(-x)^2/2| >= Pi^2/6, and g(x) <= -3.5.
   Return 0 if asymptotic expansion failed (unable to round), otherwise
   returns correct ternary value.
*/
static int
mpfr_li2_asympt_neg (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_t g, h;
  mpfr_prec_t w = MPFR_PREC (y) + 20;
  int inex = 0;

  MPFR_ASSERTN (mpfr_cmp_si (x, -7) <= 0);

  mpfr_init2 (g, w);
  mpfr_init2 (h, w);
  mpfr_neg (g, x, MPFR_RNDN);
  mpfr_log (g, g, MPFR_RNDN);    /* rel. error <= |(1 + theta) - 1| */
  mpfr_sqr (g, g, MPFR_RNDN);    /* rel. error <= |(1 + theta)^3 - 1| <= 2^(2-w) */
  mpfr_div_2ui (g, g, 1, MPFR_RNDN);     /* rel. error <= 2^(2-w) */
  mpfr_const_pi (h, MPFR_RNDN);  /* error <= 2^(1-w) */
  mpfr_sqr (h, h, MPFR_RNDN);    /* rel. error <= 2^(2-w) */
  mpfr_div_ui (h, h, 6, MPFR_RNDN);      /* rel. error <= |(1 + theta)^4 - 1|
                                           <= 5 * 2^(-w) */
  MPFR_ASSERTN (MPFR_EXP (g) >= MPFR_EXP (h));
  mpfr_add (g, g, h, MPFR_RNDN); /* err <= ulp(g)/2 + g*2^(2-w) + g*5*2^(-w)
                                   <= ulp(g) * (1/2 + 4 + 5) < 10 ulp(g).

                                   If in addition |1/x| <= 4 ulp(g), then the
                                   total error is bounded by 16 ulp(g). */
  if ((MPFR_EXP (x) >= (mpfr_exp_t) (w - 2) - MPFR_EXP (g)) &&
      MPFR_CAN_ROUND (g, w - 4, MPFR_PREC (y), rnd_mode))
    inex = mpfr_neg (y, g, rnd_mode);

  mpfr_clear (g);
  mpfr_clear (h);

  return inex;
}

/* Compute the real part of the dilogarithm defined by
   Li2(x) = -\Int_{t=0}^x log(1-t)/t dt */
int
mpfr_li2 (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  int inexact;
  mpfr_exp_t err;
  mpfr_prec_t yp, m;
  MPFR_ZIV_DECL (loop);
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd_mode),
     ("y[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (y), mpfr_log_prec, y, inexact));

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (x))
        {
          MPFR_SET_NEG (y);
          MPFR_SET_INF (y);
          MPFR_RET (0);
        }
      else                      /* x is zero */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (x));
          MPFR_SET_SAME_SIGN (y, x);
          MPFR_SET_ZERO (y);
          MPFR_RET (0);
        }
    }

  /* Li2(x) = x + x^2/4 + x^3/9 + ..., more precisely for 0 < x <= 1/2
     we have |Li2(x) - x| < x^2/2 <= 2^(2EXP(x)-1) and for -1/2 <= x < 0
     we have |Li2(x) - x| < x^2/4 <= 2^(2EXP(x)-2) */
  if (MPFR_IS_POS (x))
    MPFR_FAST_COMPUTE_IF_SMALL_INPUT (y, x, -MPFR_GET_EXP (x), 1, 1, rnd_mode,
                                      {});
  else
    MPFR_FAST_COMPUTE_IF_SMALL_INPUT (y, x, -MPFR_GET_EXP (x), 2, 0, rnd_mode,
                                      {});

  MPFR_SAVE_EXPO_MARK (expo);
  yp = MPFR_PREC (y);
  m = yp + MPFR_INT_CEIL_LOG2 (yp) + 13;

  if (MPFR_LIKELY ((mpfr_cmp_ui (x, 0) > 0) && (mpfr_cmp_d (x, 0.5) <= 0)))
    /* 0 < x <= 1/2: Li2(x) = S(-log(1-x))-log^2(1-x)/4 */
    {
      mpfr_t s, u;
      mpfr_exp_t expo_l;
      int k;

      mpfr_init2 (u, m);
      mpfr_init2 (s, m);

      MPFR_ZIV_INIT (loop, m);
      for (;;)
        {
          mpfr_ui_sub (u, 1, x, MPFR_RNDN);
          mpfr_log (u, u, MPFR_RNDU);
          if (MPFR_IS_ZERO(u))
            goto next_m;
          mpfr_neg (u, u, MPFR_RNDN);    /* u = -log(1-x) */
          expo_l = MPFR_GET_EXP (u);
          k = li2_series (s, u, MPFR_RNDU);
          err = 1 + MPFR_INT_CEIL_LOG2 (k + 1);

          mpfr_sqr (u, u, MPFR_RNDU);
          mpfr_div_2ui (u, u, 2, MPFR_RNDU);     /* u = log^2(1-x) / 4 */
          mpfr_sub (s, s, u, MPFR_RNDN);

          /* error(s) <= (0.5 + 2^(d-EXP(s))
             + 2^(3 + MAX(1, - expo_l) - EXP(s))) ulp(s) */
          err = MAX (err, MAX (1, - expo_l) - 1) - MPFR_GET_EXP (s);
          err = 2 + MAX (-1, err);
          if (MPFR_CAN_ROUND (s, (mpfr_exp_t) m - err, yp, rnd_mode))
            break;

        next_m:
          MPFR_ZIV_NEXT (loop, m);
          mpfr_set_prec (u, m);
          mpfr_set_prec (s, m);
        }
      MPFR_ZIV_FREE (loop);
      inexact = mpfr_set (y, s, rnd_mode);

      mpfr_clear (u);
      mpfr_clear (s);
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_check_range (y, inexact, rnd_mode);
    }
  else if (!mpfr_cmp_ui (x, 1))
    /* Li2(1)= pi^2 / 6 */
    {
      mpfr_t u;
      mpfr_init2 (u, m);

      MPFR_ZIV_INIT (loop, m);
      for (;;)
        {
          mpfr_const_pi (u, MPFR_RNDU);
          mpfr_sqr (u, u, MPFR_RNDN);
          mpfr_div_ui (u, u, 6, MPFR_RNDN);

          err = m - 4;          /* error(u) <= 19/2 ulp(u) */
          if (MPFR_CAN_ROUND (u, err, yp, rnd_mode))
            break;

          MPFR_ZIV_NEXT (loop, m);
          mpfr_set_prec (u, m);
        }
      MPFR_ZIV_FREE (loop);
      inexact = mpfr_set (y, u, rnd_mode);

      mpfr_clear (u);
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_check_range (y, inexact, rnd_mode);
    }
  else if (mpfr_cmp_ui (x, 2) >= 0)
    /* x >= 2: Li2(x) = -S(-log(1-1/x))-log^2(x)/2+log^2(1-1/x)/4+pi^2/3 */
    {
      int k;
      mpfr_exp_t expo_l;
      mpfr_t s, u, xx;

      if (mpfr_cmp_ui (x, 38) >= 0)
        {
          inexact = mpfr_li2_asympt_pos (y, x, rnd_mode);
          if (inexact != 0)
            goto end_of_case_gt2;
        }

      mpfr_init2 (u, m);
      mpfr_init2 (s, m);
      mpfr_init2 (xx, m);

      MPFR_ZIV_INIT (loop, m);
      for (;;)
        {
          mpfr_ui_div (xx, 1, x, MPFR_RNDN);
          mpfr_neg (xx, xx, MPFR_RNDN);
          mpfr_log1p (u, xx, MPFR_RNDD);
          mpfr_neg (u, u, MPFR_RNDU);    /* u = -log(1-1/x) */
          expo_l = MPFR_GET_EXP (u);
          k = li2_series (s, u, MPFR_RNDN);
          mpfr_neg (s, s, MPFR_RNDN);
          err = MPFR_INT_CEIL_LOG2 (k + 1) + 1; /* error(s) <= 2^err ulp(s) */

          mpfr_sqr (u, u, MPFR_RNDN);
          mpfr_div_2ui (u, u, 2, MPFR_RNDN);     /* u= log^2(1-1/x)/4 */
          mpfr_add (s, s, u, MPFR_RNDN);
          err =
            MAX (err,
                 3 + MAX (1, -expo_l) + MPFR_GET_EXP (u)) - MPFR_GET_EXP (s);
          err = 2 + MAX (-1, err);      /* error(s) <= 2^err ulp(s) */
          err += MPFR_GET_EXP (s);

          mpfr_log (u, x, MPFR_RNDU);
          mpfr_sqr (u, u, MPFR_RNDN);
          mpfr_div_2ui (u, u, 1, MPFR_RNDN);     /* u = log^2(x)/2 */
          mpfr_sub (s, s, u, MPFR_RNDN);
          err = MAX (err, 3 + MPFR_GET_EXP (u)) - MPFR_GET_EXP (s);
          err = 2 + MAX (-1, err);      /* error(s) <= 2^err ulp(s) */
          err += MPFR_GET_EXP (s);

          mpfr_const_pi (u, MPFR_RNDU);
          mpfr_sqr (u, u, MPFR_RNDN);
          mpfr_div_ui (u, u, 3, MPFR_RNDN);      /* u = pi^2/3 */
          mpfr_add (s, s, u, MPFR_RNDN);
          err = MAX (err, 2) - MPFR_GET_EXP (s);
          err = 2 + MAX (-1, err);      /* error(s) <= 2^err ulp(s) */
          if (MPFR_CAN_ROUND (s, (mpfr_exp_t) m - err, yp, rnd_mode))
            break;

          MPFR_ZIV_NEXT (loop, m);
          mpfr_set_prec (u, m);
          mpfr_set_prec (s, m);
          mpfr_set_prec (xx, m);
        }
      MPFR_ZIV_FREE (loop);
      inexact = mpfr_set (y, s, rnd_mode);
      mpfr_clears (s, u, xx, (mpfr_ptr) 0);

    end_of_case_gt2:
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_check_range (y, inexact, rnd_mode);
    }
  else if (mpfr_cmp_ui (x, 1) > 0)
    /* 2 > x > 1: Li2(x) = S(log(x))+log^2(x)/4-log(x)log(x-1)+pi^2/6 */
    {
      int k;
      mpfr_exp_t e1, e2;
      mpfr_t s, u, v, xx;
      mpfr_init2 (s, m);
      mpfr_init2 (u, m);
      mpfr_init2 (v, m);
      mpfr_init2 (xx, m);

      MPFR_ZIV_INIT (loop, m);
      for (;;)
        {
          mpfr_log (v, x, MPFR_RNDU);
          k = li2_series (s, v, MPFR_RNDN);
          e1 = MPFR_GET_EXP (s);

          mpfr_sqr (u, v, MPFR_RNDN);
          mpfr_div_2ui (u, u, 2, MPFR_RNDN);     /* u = log^2(x)/4 */
          mpfr_add (s, s, u, MPFR_RNDN);

          mpfr_sub_ui (xx, x, 1, MPFR_RNDN);
          mpfr_log (u, xx, MPFR_RNDU);
          e2 = MPFR_GET_EXP (u);
          mpfr_mul (u, v, u, MPFR_RNDN); /* u = log(x) * log(x-1) */
          mpfr_sub (s, s, u, MPFR_RNDN);

          mpfr_const_pi (u, MPFR_RNDU);
          mpfr_sqr (u, u, MPFR_RNDN);
          mpfr_div_ui (u, u, 6, MPFR_RNDN);      /* u = pi^2/6 */
          mpfr_add (s, s, u, MPFR_RNDN);
          /* error(s) <= (31 + (k+1) * 2^(1-e1) + 2^(1-e2)) ulp(s)
             see algorithms.tex */
          err = MAX (MPFR_INT_CEIL_LOG2 (k + 1) + 1 - e1, 1 - e2);
          err = 2 + MAX (5, err);
          if (MPFR_CAN_ROUND (s, (mpfr_exp_t) m - err, yp, rnd_mode))
            break;

          MPFR_ZIV_NEXT (loop, m);
          mpfr_set_prec (s, m);
          mpfr_set_prec (u, m);
          mpfr_set_prec (v, m);
          mpfr_set_prec (xx, m);
        }
      MPFR_ZIV_FREE (loop);
      inexact = mpfr_set (y, s, rnd_mode);

      mpfr_clears (s, u, v, xx, (mpfr_ptr) 0);
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_check_range (y, inexact, rnd_mode);
    }
  else if (mpfr_cmp_ui_2exp (x, 1, -1) > 0) /*  1/2 < x < 1 */
    /* 1 > x > 1/2: Li2(x) = -S(-log(x))+log^2(x)/4-log(x)log(1-x)+pi^2/6 */
    {
      int k;
      mpfr_t s, u, v, xx;
      mpfr_init2 (s, m);
      mpfr_init2 (u, m);
      mpfr_init2 (v, m);
      mpfr_init2 (xx, m);


      MPFR_ZIV_INIT (loop, m);
      for (;;)
        {
          mpfr_log (u, x, MPFR_RNDD);
          mpfr_neg (u, u, MPFR_RNDN);
          k = li2_series (s, u, MPFR_RNDN);
          mpfr_neg (s, s, MPFR_RNDN);
          err = 1 + MPFR_INT_CEIL_LOG2 (k + 1) - MPFR_GET_EXP (s);

          mpfr_ui_sub (xx, 1, x, MPFR_RNDN);
          mpfr_log (v, xx, MPFR_RNDU);
          mpfr_mul (v, v, u, MPFR_RNDN); /* v = - log(x) * log(1-x) */
          mpfr_add (s, s, v, MPFR_RNDN);
          err = MAX (err, 1 - MPFR_GET_EXP (v));
          err = 2 + MAX (3, err) - MPFR_GET_EXP (s);

          mpfr_sqr (u, u, MPFR_RNDN);
          mpfr_div_2ui (u, u, 2, MPFR_RNDN);     /* u = log^2(x)/4 */
          mpfr_add (s, s, u, MPFR_RNDN);
          err = MAX (err, 2 + MPFR_GET_EXP (u)) - MPFR_GET_EXP (s);
          err = 2 + MAX (-1, err) + MPFR_GET_EXP (s);

          mpfr_const_pi (u, MPFR_RNDU);
          mpfr_sqr (u, u, MPFR_RNDN);
          mpfr_div_ui (u, u, 6, MPFR_RNDN);      /* u = pi^2/6 */
          mpfr_add (s, s, u, MPFR_RNDN);
          err = MAX (err, 3) - MPFR_GET_EXP (s);
          err = 2 + MAX (-1, err);

          if (MPFR_CAN_ROUND (s, (mpfr_exp_t) m - err, yp, rnd_mode))
            break;

          MPFR_ZIV_NEXT (loop, m);
          mpfr_set_prec (s, m);
          mpfr_set_prec (u, m);
          mpfr_set_prec (v, m);
          mpfr_set_prec (xx, m);
        }
      MPFR_ZIV_FREE (loop);
      inexact = mpfr_set (y, s, rnd_mode);

      mpfr_clears (s, u, v, xx, (mpfr_ptr) 0);
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_check_range (y, inexact, rnd_mode);
    }
  else if (mpfr_cmp_si (x, -1) >= 0)
    /* 0 > x >= -1: Li2(x) = -S(log(1-x))-log^2(1-x)/4 */
    {
      int k;
      mpfr_exp_t expo_l;
      mpfr_t s, u, xx;
      mpfr_init2 (s, m);
      mpfr_init2 (u, m);
      mpfr_init2 (xx, m);

      MPFR_ZIV_INIT (loop, m);
      for (;;)
        {
          mpfr_neg (xx, x, MPFR_RNDN);
          mpfr_log1p (u, xx, MPFR_RNDN);
          k = li2_series (s, u, MPFR_RNDN);
          mpfr_neg (s, s, MPFR_RNDN);
          expo_l = MPFR_GET_EXP (u);
          err = 1 + MPFR_INT_CEIL_LOG2 (k + 1) - MPFR_GET_EXP (s);

          mpfr_sqr (u, u, MPFR_RNDN);
          mpfr_div_2ui (u, u, 2, MPFR_RNDN);     /* u = log^2(1-x)/4 */
          mpfr_sub (s, s, u, MPFR_RNDN);
          err = MAX (err, - expo_l);
          err = 2 + MAX (err, 3);
          if (MPFR_CAN_ROUND (s, (mpfr_exp_t) m - err, yp, rnd_mode))
            break;

          MPFR_ZIV_NEXT (loop, m);
          mpfr_set_prec (s, m);
          mpfr_set_prec (u, m);
          mpfr_set_prec (xx, m);
        }
      MPFR_ZIV_FREE (loop);
      inexact = mpfr_set (y, s, rnd_mode);

      mpfr_clears (s, u, xx, (mpfr_ptr) 0);
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_check_range (y, inexact, rnd_mode);
    }
  else
    /* x < -1: Li2(x)
       = S(log(1-1/x))-log^2(-x)/4-log(1-x)log(-x)/2+log^2(1-x)/4-pi^2/6 */
    {
      int k;
      mpfr_t s, u, v, w, xx;

      if (mpfr_cmp_si (x, -7) <= 0)
        {
          inexact = mpfr_li2_asympt_neg (y, x, rnd_mode);
          if (inexact != 0)
            goto end_of_case_ltm1;
        }

      mpfr_init2 (s, m);
      mpfr_init2 (u, m);
      mpfr_init2 (v, m);
      mpfr_init2 (w, m);
      mpfr_init2 (xx, m);

      MPFR_ZIV_INIT (loop, m);
      for (;;)
        {
          mpfr_ui_div (xx, 1, x, MPFR_RNDN);
          mpfr_neg (xx, xx, MPFR_RNDN);
          mpfr_log1p (u, xx, MPFR_RNDN);
          k = li2_series (s, u, MPFR_RNDN);

          mpfr_ui_sub (xx, 1, x, MPFR_RNDN);
          mpfr_log (u, xx, MPFR_RNDU);
          mpfr_neg (xx, x, MPFR_RNDN);
          mpfr_log (v, xx, MPFR_RNDU);
          mpfr_mul (w, v, u, MPFR_RNDN);
          mpfr_div_2ui (w, w, 1, MPFR_RNDN);  /* w = log(-x) * log(1-x) / 2 */
          mpfr_sub (s, s, w, MPFR_RNDN);
          err = 1 + MAX (3, MPFR_INT_CEIL_LOG2 (k+1) + 1  - MPFR_GET_EXP (s))
            + MPFR_GET_EXP (s);

          mpfr_sqr (w, v, MPFR_RNDN);
          mpfr_div_2ui (w, w, 2, MPFR_RNDN);  /* w = log^2(-x) / 4 */
          mpfr_sub (s, s, w, MPFR_RNDN);
          err = MAX (err, 3 + MPFR_GET_EXP(w)) - MPFR_GET_EXP (s);
          err = 2 + MAX (-1, err) + MPFR_GET_EXP (s);

          mpfr_sqr (w, u, MPFR_RNDN);
          mpfr_div_2ui (w, w, 2, MPFR_RNDN);     /* w = log^2(1-x) / 4 */
          mpfr_add (s, s, w, MPFR_RNDN);
          err = MAX (err, 3 + MPFR_GET_EXP (w)) - MPFR_GET_EXP (s);
          err = 2 + MAX (-1, err) + MPFR_GET_EXP (s);

          mpfr_const_pi (w, MPFR_RNDU);
          mpfr_sqr (w, w, MPFR_RNDN);
          mpfr_div_ui (w, w, 6, MPFR_RNDN);      /* w = pi^2 / 6 */
          mpfr_sub (s, s, w, MPFR_RNDN);
          err = MAX (err, 3) - MPFR_GET_EXP (s);
          err = 2 + MAX (-1, err) + MPFR_GET_EXP (s);

          if (MPFR_CAN_ROUND (s, (mpfr_exp_t) m - err, yp, rnd_mode))
            break;

          MPFR_ZIV_NEXT (loop, m);
          mpfr_set_prec (s, m);
          mpfr_set_prec (u, m);
          mpfr_set_prec (v, m);
          mpfr_set_prec (w, m);
          mpfr_set_prec (xx, m);
        }
      MPFR_ZIV_FREE (loop);
      inexact = mpfr_set (y, s, rnd_mode);
      mpfr_clears (s, u, v, w, xx, (mpfr_ptr) 0);

    end_of_case_ltm1:
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_check_range (y, inexact, rnd_mode);
    }

  MPFR_RET_NEVER_GO_HERE ();
}
