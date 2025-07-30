/* mpfr_erf -- error function of a floating-point number

Copyright 2001, 2003-2017 Free Software Foundation, Inc.
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

#define EXP1 2.71828182845904523536 /* exp(1) */

static int mpfr_erf_0 (mpfr_ptr, mpfr_srcptr, double, mpfr_rnd_t);

int
mpfr_erf (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_t xf;
  int inex, large;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd_mode),
     ("y[%Pu]=%.*Rg inexact=%d", mpfr_get_prec (y), mpfr_log_prec, y, inex));

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (x)) /* erf(+inf) = +1, erf(-inf) = -1 */
        return mpfr_set_si (y, MPFR_INT_SIGN (x), MPFR_RNDN);
      else /* erf(+0) = +0, erf(-0) = -0 */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (x));
          return mpfr_set (y, x, MPFR_RNDN); /* should keep the sign of x */
        }
    }

  /* now x is neither NaN, Inf nor 0 */

  /* first try expansion at x=0 when x is small, or asymptotic expansion
     where x is large */

  MPFR_SAVE_EXPO_MARK (expo);

  /* around x=0, we have erf(x) = 2x/sqrt(Pi) (1 - x^2/3 + ...),
     with 1 - x^2/3 <= sqrt(Pi)*erf(x)/2/x <= 1 for x >= 0. This means that
     if x^2/3 < 2^(-PREC(y)-1) we can decide of the correct rounding,
     unless we have a worst-case for 2x/sqrt(Pi). */
  if (MPFR_EXP(x) < - (mpfr_exp_t) (MPFR_PREC(y) / 2))
    {
      /* we use 2x/sqrt(Pi) (1 - x^2/3) <= erf(x) <= 2x/sqrt(Pi) for x > 0
         and 2x/sqrt(Pi) <= erf(x) <= 2x/sqrt(Pi) (1 - x^2/3) for x < 0.
         In both cases |2x/sqrt(Pi) (1 - x^2/3)| <= |erf(x)| <= |2x/sqrt(Pi)|.
         We will compute l and h such that l <= |2x/sqrt(Pi) (1 - x^2/3)|
         and |2x/sqrt(Pi)| <= h. If l and h round to the same value to
         precision PREC(y) and rounding rnd_mode, then we are done. */
      mpfr_t l, h; /* lower and upper bounds for erf(x) */
      int ok, inex2;

      mpfr_init2 (l, MPFR_PREC(y) + 17);
      mpfr_init2 (h, MPFR_PREC(y) + 17);
      /* first compute l */
      mpfr_mul (l, x, x, MPFR_RNDU);
      mpfr_div_ui (l, l, 3, MPFR_RNDU); /* upper bound on x^2/3 */
      mpfr_ui_sub (l, 1, l, MPFR_RNDZ); /* lower bound on 1 - x^2/3 */
      mpfr_const_pi (h, MPFR_RNDU); /* upper bound of Pi */
      mpfr_sqrt (h, h, MPFR_RNDU); /* upper bound on sqrt(Pi) */
      mpfr_div (l, l, h, MPFR_RNDZ); /* lower bound on 1/sqrt(Pi) (1 - x^2/3) */
      mpfr_mul_2ui (l, l, 1, MPFR_RNDZ); /* 2/sqrt(Pi) (1 - x^2/3) */
      mpfr_mul (l, l, x, MPFR_RNDZ); /* |l| is a lower bound on
                                       |2x/sqrt(Pi) (1 - x^2/3)| */
      /* now compute h */
      mpfr_const_pi (h, MPFR_RNDD); /* lower bound on Pi */
      mpfr_sqrt (h, h, MPFR_RNDD); /* lower bound on sqrt(Pi) */
      mpfr_div_2ui (h, h, 1, MPFR_RNDD); /* lower bound on sqrt(Pi)/2 */
      /* since sqrt(Pi)/2 < 1, the following should not underflow */
      mpfr_div (h, x, h, MPFR_IS_POS(x) ? MPFR_RNDU : MPFR_RNDD);
      /* round l and h to precision PREC(y) */
      inex = mpfr_prec_round (l, MPFR_PREC(y), rnd_mode);
      inex2 = mpfr_prec_round (h, MPFR_PREC(y), rnd_mode);
      /* Caution: we also need inex=inex2 (inex might be 0). */
      ok = SAME_SIGN (inex, inex2) && mpfr_cmp (l, h) == 0;
      if (ok)
        mpfr_set (y, h, rnd_mode);
      mpfr_clear (l);
      mpfr_clear (h);
      if (ok)
        goto end;
      /* this test can still fail for small precision, for example
         for x=-0.100E-2 with a target precision of 3 bits, since
         the error term x^2/3 is not that small. */
    }

  mpfr_init2 (xf, 53);
  mpfr_const_log2 (xf, MPFR_RNDU);
  mpfr_div (xf, x, xf, MPFR_RNDZ); /* round to zero ensures we get a lower
                                     bound of |x/log(2)| */
  mpfr_mul (xf, xf, x, MPFR_RNDZ);
  large = mpfr_cmp_ui (xf, MPFR_PREC (y) + 1) > 0;
  mpfr_clear (xf);

  /* when x goes to infinity, we have erf(x) = 1 - 1/sqrt(Pi)/exp(x^2)/x + ...
     and |erf(x) - 1| <= exp(-x^2) is true for any x >= 0, thus if
     exp(-x^2) < 2^(-PREC(y)-1) the result is 1 or 1-epsilon.
     This rewrites as x^2/log(2) > p+1. */
  if (MPFR_UNLIKELY (large))
    /* |erf x| = 1 or 1- */
    {
      mpfr_rnd_t rnd2 = MPFR_IS_POS (x) ? rnd_mode : MPFR_INVERT_RND(rnd_mode);
      if (rnd2 == MPFR_RNDN || rnd2 == MPFR_RNDU || rnd2 == MPFR_RNDA)
        {
          inex = MPFR_INT_SIGN (x);
          mpfr_set_si (y, inex, rnd2);
        }
      else /* round to zero */
        {
          inex = -MPFR_INT_SIGN (x);
          mpfr_setmax (y, 0); /* warning: setmax keeps the old sign of y */
          MPFR_SET_SAME_SIGN (y, x);
        }
    }
  else  /* use Taylor */
    {
      double xf2;

      /* FIXME: get rid of doubles/mpfr_get_d here */
      xf2 = mpfr_get_d (x, MPFR_RNDN);
      xf2 = xf2 * xf2; /* xf2 ~ x^2 */
      inex = mpfr_erf_0 (y, x, xf2, rnd_mode);
    }

 end:
  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, inex, rnd_mode);
}

/* return x*2^e */
static double
mul_2exp (double x, mpfr_exp_t e)
{
  if (e > 0)
    {
      while (e--)
        x *= 2.0;
    }
  else
    {
      while (e++)
        x /= 2.0;
    }

  return x;
}

/* evaluates erf(x) using the expansion at x=0:

   erf(x) = 2/sqrt(Pi) * sum((-1)^k*x^(2k+1)/k!/(2k+1), k=0..infinity)

   Assumes x is neither NaN nor infinite nor zero.
   Assumes also that e*x^2 <= n (target precision).
 */
static int
mpfr_erf_0 (mpfr_ptr res, mpfr_srcptr x, double xf2, mpfr_rnd_t rnd_mode)
{
  mpfr_prec_t n, m;
  mpfr_exp_t nuk, sigmak;
  double tauk;
  mpfr_t y, s, t, u;
  unsigned int k;
  int log2tauk;
  int inex;
  MPFR_ZIV_DECL (loop);

  n = MPFR_PREC (res); /* target precision */

  /* initial working precision */
  m = n + (mpfr_prec_t) (xf2 / LOG2) + 8 + MPFR_INT_CEIL_LOG2 (n);

  mpfr_init2 (y, m);
  mpfr_init2 (s, m);
  mpfr_init2 (t, m);
  mpfr_init2 (u, m);

  MPFR_ZIV_INIT (loop, m);
  for (;;)
    {
      mpfr_mul (y, x, x, MPFR_RNDU); /* err <= 1 ulp */
      mpfr_set_ui (s, 1, MPFR_RNDN);
      mpfr_set_ui (t, 1, MPFR_RNDN);
      tauk = 0.0;

      for (k = 1; ; k++)
        {
          mpfr_mul (t, y, t, MPFR_RNDU);
          mpfr_div_ui (t, t, k, MPFR_RNDU);
          mpfr_div_ui (u, t, 2 * k + 1, MPFR_RNDU);
          sigmak = MPFR_GET_EXP (s);
          if (k % 2)
            mpfr_sub (s, s, u, MPFR_RNDN);
          else
            mpfr_add (s, s, u, MPFR_RNDN);
          sigmak -= MPFR_GET_EXP(s);
          nuk = MPFR_GET_EXP(u) - MPFR_GET_EXP(s);

          if ((nuk < - (mpfr_exp_t) m) && ((double) k >= xf2))
            break;

          /* tauk <- 1/2 + tauk * 2^sigmak + (1+8k)*2^nuk */
          tauk = 0.5 + mul_2exp (tauk, sigmak)
            + mul_2exp (1.0 + 8.0 * (double) k, nuk);
        }

      mpfr_mul (s, x, s, MPFR_RNDU);
      MPFR_SET_EXP (s, MPFR_GET_EXP (s) + 1);

      mpfr_const_pi (t, MPFR_RNDZ);
      mpfr_sqrt (t, t, MPFR_RNDZ);
      mpfr_div (s, s, t, MPFR_RNDN);
      tauk = 4.0 * tauk + 11.0; /* final ulp-error on s */
      log2tauk = __gmpfr_ceil_log2 (tauk);

      if (MPFR_LIKELY (MPFR_CAN_ROUND (s, m - log2tauk, n, rnd_mode)))
        break;

      /* Actualisation of the precision */
      MPFR_ZIV_NEXT (loop, m);
      mpfr_set_prec (y, m);
      mpfr_set_prec (s, m);
      mpfr_set_prec (t, m);
      mpfr_set_prec (u, m);

    }
  MPFR_ZIV_FREE (loop);

  inex = mpfr_set (res, s, rnd_mode);

  mpfr_clear (y);
  mpfr_clear (t);
  mpfr_clear (u);
  mpfr_clear (s);

  return inex;
}
