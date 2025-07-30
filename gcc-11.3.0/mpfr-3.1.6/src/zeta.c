/* mpfr_zeta -- compute the Riemann Zeta function

Copyright 2003-2017 Free Software Foundation, Inc.
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

/*
   Parameters:
   s - the input floating-point number
   n, p - parameters from the algorithm
   tc - an array of p floating-point numbers tc[1]..tc[p]
   Output:
   b is the result, i.e.
   sum(tc[i]*product((s+2j)*(s+2j-1)/n^2,j=1..i-1), i=1..p)*s*n^(-s-1)
*/
static void
mpfr_zeta_part_b (mpfr_t b, mpfr_srcptr s, int n, int p, mpfr_t *tc)
{
  mpfr_t s1, d, u;
  unsigned long n2;
  int l, t;
  MPFR_GROUP_DECL (group);

  if (p == 0)
    {
      MPFR_SET_ZERO (b);
      MPFR_SET_POS (b);
      return;
    }

  n2 = n * n;
  MPFR_GROUP_INIT_3 (group, MPFR_PREC (b), s1, d, u);

  /* t equals 2p-2, 2p-3, ... ; s1 equals s+t */
  t = 2 * p - 2;
  mpfr_set (d, tc[p], MPFR_RNDN);
  for (l = 1; l < p; l++)
    {
      mpfr_add_ui (s1, s, t, MPFR_RNDN); /* s + (2p-2l) */
      mpfr_mul (d, d, s1, MPFR_RNDN);
      t = t - 1;
      mpfr_add_ui (s1, s, t, MPFR_RNDN); /* s + (2p-2l-1) */
      mpfr_mul (d, d, s1, MPFR_RNDN);
      t = t - 1;
      mpfr_div_ui (d, d, n2, MPFR_RNDN);
      mpfr_add (d, d, tc[p-l], MPFR_RNDN);
      /* since s is positive and the tc[i] have alternate signs,
         the following is unlikely */
      if (MPFR_UNLIKELY (mpfr_cmpabs (d, tc[p-l]) > 0))
        mpfr_set (d, tc[p-l], MPFR_RNDN);
    }
  mpfr_mul (d, d, s, MPFR_RNDN);
  mpfr_add (s1, s, __gmpfr_one, MPFR_RNDN);
  mpfr_neg (s1, s1, MPFR_RNDN);
  mpfr_ui_pow (u, n, s1, MPFR_RNDN);
  mpfr_mul (b, d, u, MPFR_RNDN);

  MPFR_GROUP_CLEAR (group);
}

/* Input: p - an integer
   Output: fills tc[1..p], tc[i] = bernoulli(2i)/(2i)!
   tc[1]=1/12, tc[2]=-1/720, tc[3]=1/30240, ...
*/
static void
mpfr_zeta_c (int p, mpfr_t *tc)
{
  mpfr_t d;
  int k, l;

  if (p > 0)
    {
      mpfr_init2 (d, MPFR_PREC (tc[1]));
      mpfr_div_ui (tc[1], __gmpfr_one, 12, MPFR_RNDN);
      for (k = 2; k <= p; k++)
        {
          mpfr_set_ui (d, k-1, MPFR_RNDN);
          mpfr_div_ui (d, d, 12*k+6, MPFR_RNDN);
          for (l=2; l < k; l++)
            {
              mpfr_div_ui (d, d, 4*(2*k-2*l+3)*(2*k-2*l+2), MPFR_RNDN);
              mpfr_add (d, d, tc[l], MPFR_RNDN);
            }
          mpfr_div_ui (tc[k], d, 24, MPFR_RNDN);
          MPFR_CHANGE_SIGN (tc[k]);
        }
      mpfr_clear (d);
    }
}

/* Input: s - a floating-point number
          n - an integer
   Output: sum - a floating-point number approximating sum(1/i^s, i=1..n-1) */
static void
mpfr_zeta_part_a (mpfr_t sum, mpfr_srcptr s, int n)
{
  mpfr_t u, s1;
  int i;
  MPFR_GROUP_DECL (group);

  MPFR_GROUP_INIT_2 (group, MPFR_PREC (sum), u, s1);

  mpfr_neg (s1, s, MPFR_RNDN);
  mpfr_ui_pow (u, n, s1, MPFR_RNDN);
  mpfr_div_2ui (u, u, 1, MPFR_RNDN);
  mpfr_set (sum, u, MPFR_RNDN);
  for (i=n-1; i>1; i--)
    {
      mpfr_ui_pow (u, i, s1, MPFR_RNDN);
      mpfr_add (sum, sum, u, MPFR_RNDN);
    }
  mpfr_add (sum, sum, __gmpfr_one, MPFR_RNDN);

  MPFR_GROUP_CLEAR (group);
}

/* Input: s - a floating-point number >= 1/2.
          rnd_mode - a rounding mode.
          Assumes s is neither NaN nor Infinite.
   Output: z - Zeta(s) rounded to the precision of z with direction rnd_mode
*/
static int
mpfr_zeta_pos (mpfr_t z, mpfr_srcptr s, mpfr_rnd_t rnd_mode)
{
  mpfr_t b, c, z_pre, f, s1;
  double beta, sd, dnep;
  mpfr_t *tc1;
  mpfr_prec_t precz, precs, d, dint;
  int p, n, l, add;
  int inex;
  MPFR_GROUP_DECL (group);
  MPFR_ZIV_DECL (loop);

  MPFR_ASSERTD (MPFR_IS_POS (s) && MPFR_GET_EXP (s) >= 0);

  precz = MPFR_PREC (z);
  precs = MPFR_PREC (s);

  /* Zeta(x) = 1+1/2^x+1/3^x+1/4^x+1/5^x+O(1/6^x)
     so with 2^(EXP(x)-1) <= x < 2^EXP(x)
     So for x > 2^3, k^x > k^8, so 2/k^x < 2/k^8
     Zeta(x) = 1 + 1/2^x*(1+(2/3)^x+(2/4)^x+...)
             = 1 + 1/2^x*(1+sum((2/k)^x,k=3..infinity))
            <= 1 + 1/2^x*(1+sum((2/k)^8,k=3..infinity))
     And sum((2/k)^8,k=3..infinity) = -257+128*Pi^8/4725 ~= 0.0438035
     So Zeta(x) <= 1 + 1/2^x*2 for x >= 8
     The error is < 2^(-x+1) <= 2^(-2^(EXP(x)-1)+1) */
  if (MPFR_GET_EXP (s) > 3)
    {
      mpfr_exp_t err;
      err = MPFR_GET_EXP (s) - 1;
      if (err > (mpfr_exp_t) (sizeof (mpfr_exp_t)*CHAR_BIT-2))
        err = MPFR_EMAX_MAX;
      else
        err = ((mpfr_exp_t)1) << err;
      err = 1 - (-err+1); /* GET_EXP(one) - (-err+1) = err :) */
      MPFR_FAST_COMPUTE_IF_SMALL_INPUT (z, __gmpfr_one, err, 0, 1,
                                        rnd_mode, {});
    }

  d = precz + MPFR_INT_CEIL_LOG2(precz) + 10;

  /* we want that s1 = s-1 is exact, i.e. we should have PREC(s1) >= EXP(s) */
  dint = (mpfr_uexp_t) MPFR_GET_EXP (s);
  mpfr_init2 (s1, MAX (precs, dint));
  inex = mpfr_sub (s1, s, __gmpfr_one, MPFR_RNDN);
  MPFR_ASSERTD (inex == 0);

  /* case s=1 should have already been handled */
  MPFR_ASSERTD (!MPFR_IS_ZERO (s1));

  MPFR_GROUP_INIT_4 (group, MPFR_PREC_MIN, b, c, z_pre, f);

  MPFR_ZIV_INIT (loop, d);
  for (;;)
    {
      /* Principal loop: we compute, in z_pre,
         an approximation of Zeta(s), that we send to can_round */
      if (MPFR_GET_EXP (s1) <= -(mpfr_exp_t) ((mpfr_prec_t) (d-3)/2))
        /* Branch 1: when s-1 is very small, one
           uses the approximation Zeta(s)=1/(s-1)+gamma,
           where gamma is Euler's constant */
        {
          dint = MAX (d + 3, precs);
          MPFR_TRACE (printf ("branch 1\ninternal precision=%lu\n",
                              (unsigned long) dint));
          MPFR_GROUP_REPREC_4 (group, dint, b, c, z_pre, f);
          mpfr_div (z_pre, __gmpfr_one, s1, MPFR_RNDN);
          mpfr_const_euler (f, MPFR_RNDN);
          mpfr_add (z_pre, z_pre, f, MPFR_RNDN);
        }
      else /* Branch 2 */
        {
          size_t size;

          MPFR_TRACE (printf ("branch 2\n"));
          /* Computation of parameters n, p and working precision */
          dnep = (double) d * LOG2;
          sd = mpfr_get_d (s, MPFR_RNDN);
          /* beta = dnep + 0.61 + sd * log (6.2832 / sd);
             but a larger value is ok */
#define LOG6dot2832 1.83787940484160805532
          beta = dnep + 0.61 + sd * (LOG6dot2832 - LOG2 *
                                     __gmpfr_floor_log2 (sd));
          if (beta <= 0.0)
            {
              p = 0;
              /* n = 1 + (int) (exp ((dnep - LOG2) / sd)); */
              n = 1 + (int) __gmpfr_ceil_exp2 ((d - 1.0) / sd);
            }
          else
            {
              p = 1 + (int) beta / 2;
              n = 1 + (int) ((sd + 2.0 * (double) p - 1.0) / 6.2832);
            }
          MPFR_TRACE (printf ("\nn=%d\np=%d\n",n,p));
          /* add = 4 + floor(1.5 * log(d) / log (2)).
             We should have add >= 10, which is always fulfilled since
             d = precz + 11 >= 12, thus ceil(log2(d)) >= 4 */
          add = 4 + (3 * MPFR_INT_CEIL_LOG2 (d)) / 2;
          MPFR_ASSERTD(add >= 10);
          dint = d + add;
          if (dint < precs)
            dint = precs;

          MPFR_TRACE (printf ("internal precision=%lu\n",
                              (unsigned long) dint));

          size = (p + 1) * sizeof(mpfr_t);
          tc1 = (mpfr_t*) (*__gmp_allocate_func) (size);
          for (l=1; l<=p; l++)
            mpfr_init2 (tc1[l], dint);
          MPFR_GROUP_REPREC_4 (group, dint, b, c, z_pre, f);

          MPFR_TRACE (printf ("precision of z = %lu\n",
                              (unsigned long) precz));

          /* Computation of the coefficients c_k */
          mpfr_zeta_c (p, tc1);
          /* Computation of the 3 parts of the fonction Zeta. */
          mpfr_zeta_part_a (z_pre, s, n);
          mpfr_zeta_part_b (b, s, n, p, tc1);
          /* s1 = s-1 is already computed above */
          mpfr_div (c, __gmpfr_one, s1, MPFR_RNDN);
          mpfr_ui_pow (f, n, s1, MPFR_RNDN);
          mpfr_div (c, c, f, MPFR_RNDN);
          MPFR_TRACE (MPFR_DUMP (c));
          mpfr_add (z_pre, z_pre, c, MPFR_RNDN);
          mpfr_add (z_pre, z_pre, b, MPFR_RNDN);
          for (l=1; l<=p; l++)
            mpfr_clear (tc1[l]);
          (*__gmp_free_func) (tc1, size);
          /* End branch 2 */
        }

      MPFR_TRACE (MPFR_DUMP (z_pre));
      if (MPFR_LIKELY (MPFR_CAN_ROUND (z_pre, d-3, precz, rnd_mode)))
        break;
      MPFR_ZIV_NEXT (loop, d);
    }
  MPFR_ZIV_FREE (loop);

  inex = mpfr_set (z, z_pre, rnd_mode);

  MPFR_GROUP_CLEAR (group);
  mpfr_clear (s1);

  return inex;
}

int
mpfr_zeta (mpfr_t z, mpfr_srcptr s, mpfr_rnd_t rnd_mode)
{
  mpfr_t z_pre, s1, y, p;
  double sd, eps, m1, c;
  long add;
  mpfr_prec_t precz, prec1, precs, precs1;
  int inex;
  MPFR_GROUP_DECL (group);
  MPFR_ZIV_DECL (loop);
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC (
    ("s[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (s), mpfr_log_prec, s, rnd_mode),
    ("z[%Pu]=%.*Rg inexact=%d", mpfr_get_prec (z), mpfr_log_prec, z, inex));

  /* Zero, Nan or Inf ? */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (s)))
    {
      if (MPFR_IS_NAN (s))
        {
          MPFR_SET_NAN (z);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (s))
        {
          if (MPFR_IS_POS (s))
            return mpfr_set_ui (z, 1, MPFR_RNDN); /* Zeta(+Inf) = 1 */
          MPFR_SET_NAN (z); /* Zeta(-Inf) = NaN */
          MPFR_RET_NAN;
        }
      else /* s iz zero */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (s));
          return mpfr_set_si_2exp (z, -1, -1, rnd_mode);
        }
    }

  /* s is neither Nan, nor Inf, nor Zero */

  /* check tiny s: we have zeta(s) = -1/2 - 1/2 log(2 Pi) s + ... around s=0,
     and for |s| <= 0.074, we have |zeta(s) + 1/2| <= |s|.
     Thus if |s| <= 1/4*ulp(1/2), we can deduce the correct rounding
     (the 1/4 covers the case where |zeta(s)| < 1/2 and rounding to nearest).
     A sufficient condition is that EXP(s) + 1 < -PREC(z). */
  if (MPFR_GET_EXP (s) + 1 < - (mpfr_exp_t) MPFR_PREC(z))
    {
      int signs = MPFR_SIGN(s);

      MPFR_SAVE_EXPO_MARK (expo);
      mpfr_set_si_2exp (z, -1, -1, rnd_mode); /* -1/2 */
      if (rnd_mode == MPFR_RNDA)
        rnd_mode = MPFR_RNDD; /* the result is around -1/2, thus negative */
      if ((rnd_mode == MPFR_RNDU || rnd_mode == MPFR_RNDZ) && signs < 0)
        {
          mpfr_nextabove (z); /* z = -1/2 + epsilon */
          inex = 1;
        }
      else if (rnd_mode == MPFR_RNDD && signs > 0)
        {
          mpfr_nextbelow (z); /* z = -1/2 - epsilon */
          inex = -1;
        }
      else
        {
          if (rnd_mode == MPFR_RNDU) /* s > 0: z = -1/2 */
            inex = 1;
          else if (rnd_mode == MPFR_RNDD)
            inex = -1;              /* s < 0: z = -1/2 */
          else /* (MPFR_RNDZ and s > 0) or MPFR_RNDN: z = -1/2 */
            inex = (signs > 0) ? 1 : -1;
        }
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_check_range (z, inex, rnd_mode);
    }

  /* Check for case s= -2n */
  if (MPFR_IS_NEG (s))
    {
      mpfr_t tmp;
      tmp[0] = *s;
      MPFR_EXP (tmp) = MPFR_GET_EXP (s) - 1;
      if (mpfr_integer_p (tmp))
        {
          MPFR_SET_ZERO (z);
          MPFR_SET_POS (z);
          MPFR_RET (0);
        }
    }

  /* Check for case s=1 before changing the exponent range */
  if (mpfr_cmp (s, __gmpfr_one) == 0)
    {
      MPFR_SET_INF (z);
      MPFR_SET_POS (z);
      mpfr_set_divby0 ();
      MPFR_RET (0);
    }

  MPFR_SAVE_EXPO_MARK (expo);

  /* Compute Zeta */
  if (MPFR_IS_POS (s) && MPFR_GET_EXP (s) >= 0) /* Case s >= 1/2 */
    inex = mpfr_zeta_pos (z, s, rnd_mode);
  else /* use reflection formula
          zeta(s) = 2^s*Pi^(s-1)*sin(Pi*s/2)*gamma(1-s)*zeta(1-s) */
    {
      int overflow = 0;

      precz = MPFR_PREC (z);
      precs = MPFR_PREC (s);

      /* Precision precs1 needed to represent 1 - s, and s + 2,
         without any truncation */
      precs1 = precs + 2 + MAX (0, - MPFR_GET_EXP (s));
      sd = mpfr_get_d (s, MPFR_RNDN) - 1.0;
      if (sd < 0.0)
        sd = -sd; /* now sd = abs(s-1.0) */
      /* Precision prec1 is the precision on elementary computations;
         it ensures a final precision prec1 - add for zeta(s) */
      /* eps = pow (2.0, - (double) precz - 14.0); */
      eps = __gmpfr_ceil_exp2 (- (double) precz - 14.0);
      m1 = 1.0 + MAX(1.0 / eps,  2.0 * sd) * (1.0 + eps);
      c = (1.0 + eps) * (1.0 + eps * MAX(8.0, m1));
      /* add = 1 + floor(log(c*c*c*(13 + m1))/log(2)); */
      add = __gmpfr_ceil_log2 (c * c * c * (13.0 + m1));
      prec1 = precz + add;
      prec1 = MAX (prec1, precs1) + 10;

      MPFR_GROUP_INIT_4 (group, prec1, z_pre, s1, y, p);
      MPFR_ZIV_INIT (loop, prec1);
      for (;;)
        {
          mpfr_sub (s1, __gmpfr_one, s, MPFR_RNDN); /* s1 = 1-s */
          mpfr_zeta_pos (z_pre, s1, MPFR_RNDN);   /* zeta(1-s)  */
          mpfr_gamma (y, s1, MPFR_RNDN);          /* gamma(1-s) */
          if (MPFR_IS_INF (y)) /* Zeta(s) < 0 for -4k-2 < s < -4k,
                                  Zeta(s) > 0 for -4k < s < -4k+2 */
            {
              mpfr_div_2ui (s1, s, 2, MPFR_RNDN); /* s/4, exact */
              mpfr_frac (s1, s1, MPFR_RNDN); /* exact, -1 < s1 < 0 */
              overflow = (mpfr_cmp_si_2exp (s1, -1, -1) > 0) ? -1 : 1;
              break;
            }
          mpfr_mul (z_pre, z_pre, y, MPFR_RNDN);  /* gamma(1-s)*zeta(1-s) */

          mpfr_const_pi (p, MPFR_RNDD); /* p is Pi */

          /* multiply z_pre by 2^s*Pi^(s-1) where p=Pi, s1=1-s */
          mpfr_mul_2ui (y, p, 1, MPFR_RNDN);      /* 2*Pi */
          mpfr_neg (s1, s1, MPFR_RNDN);           /* s-1 */
          mpfr_pow (y, y, s1, MPFR_RNDN);         /* (2*Pi)^(s-1) */
          mpfr_mul (z_pre, z_pre, y, MPFR_RNDN);
          mpfr_mul_2ui (z_pre, z_pre, 1, MPFR_RNDN);

          /* multiply z_pre by sin(Pi*s/2) */
          mpfr_mul (y, s, p, MPFR_RNDN);
          mpfr_div_2ui (p, y, 1, MPFR_RNDN);      /* p = s*Pi/2 */
          mpfr_sin (y, p, MPFR_RNDN);             /* y = sin(Pi*s/2) */
          if (MPFR_GET_EXP(y) < 0) /* take account of cancellation in sin(p) */
            {
              mpfr_t t;
              mpfr_init2 (t, prec1 - MPFR_GET_EXP(y));
              mpfr_const_pi (t, MPFR_RNDD);
              mpfr_mul (t, s, t, MPFR_RNDN);
              mpfr_div_2ui (t, t, 1, MPFR_RNDN);
              mpfr_sin (y, t, MPFR_RNDN);
              mpfr_clear (t);
            }
          mpfr_mul (z_pre, z_pre, y, MPFR_RNDN);

          if (MPFR_LIKELY (MPFR_CAN_ROUND (z_pre, prec1 - add, precz,
                                           rnd_mode)))
            break;

          MPFR_ZIV_NEXT (loop, prec1);
          MPFR_GROUP_REPREC_4 (group, prec1, z_pre, s1, y, p);
        }
      MPFR_ZIV_FREE (loop);
      if (overflow != 0)
        {
          inex = mpfr_overflow (z, rnd_mode, overflow);
          MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, MPFR_FLAGS_OVERFLOW);
        }
      else
        inex = mpfr_set (z, z_pre, rnd_mode);
      MPFR_GROUP_CLEAR (group);
    }

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (z, inex, rnd_mode);
}
