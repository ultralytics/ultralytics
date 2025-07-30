/* mpfr_lngamma -- lngamma function

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

/* given a precision p, return alpha, such that the argument reduction
   will use k = alpha*p*log(2).

   Warning: we should always have alpha >= log(2)/(2Pi) ~ 0.11,
   and the smallest value of alpha multiplied by the smallest working
   precision should be >= 4.
*/
static void
mpfr_gamma_alpha (mpfr_t s, mpfr_prec_t p)
{
  if (p <= 100)
    mpfr_set_ui_2exp (s, 614, -10, MPFR_RNDN); /* about 0.6 */
  else if (p <= 500)
    mpfr_set_ui_2exp (s, 819, -10, MPFR_RNDN); /* about 0.8 */
  else if (p <= 1000)
    mpfr_set_ui_2exp (s, 1331, -10, MPFR_RNDN); /* about 1.3 */
  else if (p <= 2000)
    mpfr_set_ui_2exp (s, 1741, -10, MPFR_RNDN); /* about 1.7 */
  else if (p <= 5000)
    mpfr_set_ui_2exp (s, 2253, -10, MPFR_RNDN); /* about 2.2 */
  else if (p <= 10000)
    mpfr_set_ui_2exp (s, 3482, -10, MPFR_RNDN); /* about 3.4 */
  else
    mpfr_set_ui_2exp (s, 9, -1, MPFR_RNDN); /* 4.5 */
}

#ifdef IS_GAMMA

/* This function is called in case of intermediate overflow/underflow.
   The s1 and s2 arguments are temporary MPFR numbers, having the
   working precision. If the result could be determined, then the
   flags are updated via pexpo, y is set to the result, and the
   (non-zero) ternary value is returned. Otherwise 0 is returned
   in order to perform the next Ziv iteration. */
static int
mpfr_explgamma (mpfr_ptr y, mpfr_srcptr x, mpfr_save_expo_t *pexpo,
                mpfr_ptr s1, mpfr_ptr s2, mpfr_rnd_t rnd)
{
  mpfr_t t1, t2;
  int inex1, inex2, sign;
  MPFR_BLOCK_DECL (flags1);
  MPFR_BLOCK_DECL (flags2);
  MPFR_GROUP_DECL (group);

  MPFR_BLOCK (flags1, inex1 = mpfr_lgamma (s1, &sign, x, MPFR_RNDD));
  MPFR_ASSERTN (inex1 != 0);
  /* s1 = RNDD(lngamma(x)), inexact */
  if (MPFR_UNLIKELY (MPFR_OVERFLOW (flags1)))
    {
      if (MPFR_SIGN (s1) > 0)
        {
          MPFR_SAVE_EXPO_UPDATE_FLAGS (*pexpo, MPFR_FLAGS_OVERFLOW);
          return mpfr_overflow (y, rnd, sign);
        }
      else
        {
          MPFR_SAVE_EXPO_UPDATE_FLAGS (*pexpo, MPFR_FLAGS_UNDERFLOW);
          return mpfr_underflow (y, rnd == MPFR_RNDN ? MPFR_RNDZ : rnd, sign);
        }
    }

  mpfr_set (s2, s1, MPFR_RNDN);     /* exact */
  mpfr_nextabove (s2);              /* v = RNDU(lngamma(z0)) */

  if (sign < 0)
    rnd = MPFR_INVERT_RND (rnd);  /* since the result with be negated */
  MPFR_GROUP_INIT_2 (group, MPFR_PREC (y), t1, t2);
  MPFR_BLOCK (flags1, inex1 = mpfr_exp (t1, s1, rnd));
  MPFR_BLOCK (flags2, inex2 = mpfr_exp (t2, s2, rnd));
  /* t1 is the rounding with mode 'rnd' of a lower bound on |Gamma(x)|,
     t2 is the rounding with mode 'rnd' of an upper bound, thus if both
     are equal, so is the wanted result. If t1 and t2 differ or the flags
     differ, at some point of Ziv's loop they should agree. */
  if (mpfr_equal_p (t1, t2) && flags1 == flags2)
    {
      MPFR_ASSERTN ((inex1 > 0 && inex2 > 0) || (inex1 < 0 && inex2 < 0));
      mpfr_set4 (y, t1, MPFR_RNDN, sign);  /* exact */
      if (sign < 0)
        inex1 = - inex1;
      MPFR_SAVE_EXPO_UPDATE_FLAGS (*pexpo, flags1);
    }
  else
    inex1 = 0;  /* couldn't determine the result */
  MPFR_GROUP_CLEAR (group);

  return inex1;
}

#else

static int
unit_bit (mpfr_srcptr x)
{
  mpfr_exp_t expo;
  mpfr_prec_t prec;
  mp_limb_t x0;

  expo = MPFR_GET_EXP (x);
  if (expo <= 0)
    return 0;  /* |x| < 1 */

  prec = MPFR_PREC (x);
  if (expo > prec)
    return 0;  /* y is a multiple of 2^(expo-prec), thus an even integer */

  /* Now, the unit bit is represented. */

  prec = MPFR_PREC2LIMBS (prec) * GMP_NUMB_BITS - expo;
  /* number of represented fractional bits (including the trailing 0's) */

  x0 = *(MPFR_MANT (x) + prec / GMP_NUMB_BITS);
  /* limb containing the unit bit */

  return (x0 >> (prec % GMP_NUMB_BITS)) & 1;
}

#endif

/* lngamma(x) = log(gamma(x)).
   We use formula [6.1.40] from Abramowitz&Stegun:
   lngamma(z) = (z-1/2)*log(z) - z + 1/2*log(2*Pi)
              + sum (Bernoulli[2m]/(2m)/(2m-1)/z^(2m-1),m=1..infinity)
   According to [6.1.42], if the sum is truncated after m=n, the error
   R_n(z) is bounded by |B[2n+2]|*K(z)/(2n+1)/(2n+2)/|z|^(2n+1)
   where K(z) = max (z^2/(u^2+z^2)) for u >= 0.
   For z real, |K(z)| <= 1 thus R_n(z) is bounded by the first neglected term.
 */
#ifdef IS_GAMMA
#define GAMMA_FUNC mpfr_gamma_aux
#else
#define GAMMA_FUNC mpfr_lngamma_aux
#endif

static int
GAMMA_FUNC (mpfr_ptr y, mpfr_srcptr z0, mpfr_rnd_t rnd)
{
  mpfr_prec_t precy, w; /* working precision */
  mpfr_t s, t, u, v, z;
  unsigned long m, k, maxm;
  mpz_t *INITIALIZED(B);  /* variable B declared as initialized */
  int compared;
  int inexact = 0;  /* 0 means: result y not set yet */
  mpfr_exp_t err_s, err_t;
  unsigned long Bm = 0; /* number of allocated B[] */
  unsigned long oldBm;
  double d;
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_ZIV_DECL (loop);

  compared = mpfr_cmp_ui (z0, 1);

  MPFR_SAVE_EXPO_MARK (expo);

#ifndef IS_GAMMA /* lngamma or lgamma */
  if (compared == 0 || (compared > 0 && mpfr_cmp_ui (z0, 2) == 0))
    {
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_set_ui (y, 0, MPFR_RNDN);  /* lngamma(1 or 2) = +0 */
    }

  /* Deal here with tiny inputs. We have for -0.3 <= x <= 0.3:
     - log|x| - gamma*x <= log|gamma(x)| <= - log|x| - gamma*x + x^2 */
  if (MPFR_EXP(z0) <= - (mpfr_exp_t) MPFR_PREC(y))
    {
      mpfr_t l, h, g;
      int ok, inex1, inex2;
      mpfr_prec_t prec = MPFR_PREC(y) + 14;
      MPFR_ZIV_DECL (loop);

      MPFR_ZIV_INIT (loop, prec);
      do
        {
          mpfr_init2 (l, prec);
          if (MPFR_IS_POS(z0))
            {
              mpfr_log (l, z0, MPFR_RNDU); /* upper bound for log(z0) */
              mpfr_init2 (h, MPFR_PREC(l));
            }
          else
            {
              mpfr_init2 (h, MPFR_PREC(z0));
              mpfr_neg (h, z0, MPFR_RNDN); /* exact */
              mpfr_log (l, h, MPFR_RNDU); /* upper bound for log(-z0) */
              mpfr_set_prec (h, MPFR_PREC(l));
            }
          mpfr_neg (l, l, MPFR_RNDD); /* lower bound for -log(|z0|) */
          mpfr_set (h, l, MPFR_RNDD); /* exact */
          mpfr_nextabove (h); /* upper bound for -log(|z0|), avoids two calls
                                 to mpfr_log */
          mpfr_init2 (g, MPFR_PREC(l));
          /* if z0>0, we need an upper approximation of Euler's constant
             for the left bound */
          mpfr_const_euler (g, MPFR_IS_POS(z0) ? MPFR_RNDU : MPFR_RNDD);
          mpfr_mul (g, g, z0, MPFR_RNDD);
          mpfr_sub (l, l, g, MPFR_RNDD);
          mpfr_const_euler (g, MPFR_IS_POS(z0) ? MPFR_RNDD : MPFR_RNDU); /* cached */
          mpfr_mul (g, g, z0, MPFR_RNDU);
          mpfr_sub (h, h, g, MPFR_RNDD);
          mpfr_mul (g, z0, z0, MPFR_RNDU);
          mpfr_add (h, h, g, MPFR_RNDU);
          inex1 = mpfr_prec_round (l, MPFR_PREC(y), rnd);
          inex2 = mpfr_prec_round (h, MPFR_PREC(y), rnd);
          /* Caution: we not only need l = h, but both inexact flags should
             agree. Indeed, one of the inexact flags might be zero. In that
             case if we assume lngamma(z0) cannot be exact, the other flag
             should be correct. We are conservative here and request that both
             inexact flags agree. */
          ok = SAME_SIGN (inex1, inex2) && mpfr_cmp (l, h) == 0;
          if (ok)
            mpfr_set (y, h, rnd); /* exact */
          mpfr_clear (l);
          mpfr_clear (h);
          mpfr_clear (g);
          if (ok)
            {
              MPFR_ZIV_FREE (loop);
              MPFR_SAVE_EXPO_FREE (expo);
              return mpfr_check_range (y, inex1, rnd);
            }
          /* since we have log|gamma(x)| = - log|x| - gamma*x + O(x^2),
             if x ~ 2^(-n), then we have a n-bit approximation, thus
             we can try again with a working precision of n bits,
             especially when n >> PREC(y).
             Otherwise we would use the reflection formula evaluating x-1,
             which would need precision n. */
          MPFR_ZIV_NEXT (loop, prec);
        }
      while (prec <= -MPFR_EXP(z0));
      MPFR_ZIV_FREE (loop);
    }
#endif

  precy = MPFR_PREC(y);

  mpfr_init2 (s, MPFR_PREC_MIN);
  mpfr_init2 (t, MPFR_PREC_MIN);
  mpfr_init2 (u, MPFR_PREC_MIN);
  mpfr_init2 (v, MPFR_PREC_MIN);
  mpfr_init2 (z, MPFR_PREC_MIN);

  if (compared < 0)
    {
      mpfr_exp_t err_u;

      /* use reflection formula:
         gamma(x) = Pi*(x-1)/sin(Pi*(2-x))/gamma(2-x)
         thus lngamma(x) = log(Pi*(x-1)/sin(Pi*(2-x))) - lngamma(2-x) */

      w = precy + MPFR_INT_CEIL_LOG2 (precy);
      w += MPFR_INT_CEIL_LOG2 (w) + 14;
      MPFR_ZIV_INIT (loop, w);
      while (1)
        {
          MPFR_ASSERTD(w >= 3);
          mpfr_set_prec (s, w);
          mpfr_set_prec (t, w);
          mpfr_set_prec (u, w);
          mpfr_set_prec (v, w);
          /* In the following, we write r for a real of absolute value
             at most 2^(-w). Different instances of r may represent different
             values. */
          mpfr_ui_sub (s, 2, z0, MPFR_RNDD); /* s = (2-z0) * (1+2r) >= 1 */
          mpfr_const_pi (t, MPFR_RNDN);      /* t = Pi * (1+r) */
          mpfr_lngamma (u, s, MPFR_RNDN); /* lngamma(2-x) */
          /* Let s = (2-z0) + h. By construction, -(2-z0)*2^(1-w) <= h <= 0.
             We have lngamma(s) = lngamma(2-z0) + h*Psi(z), z in [2-z0+h,2-z0].
             Since 2-z0+h = s >= 1 and |Psi(x)| <= max(1,log(x)) for x >= 1,
             the error on u is bounded by
             ulp(u)/2 + (2-z0)*max(1,log(2-z0))*2^(1-w)
             = (1/2 + (2-z0)*max(1,log(2-z0))*2^(1-E(u))) ulp(u) */
          d = (double) MPFR_GET_EXP(s) * 0.694; /* upper bound for log(2-z0) */
          err_u = MPFR_GET_EXP(s) + __gmpfr_ceil_log2 (d) + 1 - MPFR_GET_EXP(u);
          err_u = (err_u >= 0) ? err_u + 1 : 0;
          /* now the error on u is bounded by 2^err_u ulps */

          mpfr_mul (s, s, t, MPFR_RNDN); /* Pi*(2-x) * (1+r)^4 */
          err_s = MPFR_GET_EXP(s); /* 2-x <= 2^err_s */
          mpfr_sin (s, s, MPFR_RNDN); /* sin(Pi*(2-x)) */
          /* the error on s is bounded by 1/2*ulp(s) + [(1+2^(-w))^4-1]*(2-x)
             <= 1/2*ulp(s) + 5*2^(-w)*(2-x) for w >= 3
             <= (1/2 + 5 * 2^(-E(s)) * (2-x)) ulp(s) */
          err_s += 3 - MPFR_GET_EXP(s);
          err_s = (err_s >= 0) ? err_s + 1 : 0;
          /* the error on s is bounded by 2^err_s ulp(s), thus by
             2^(err_s+1)*2^(-w)*|s| since ulp(s) <= 2^(1-w)*|s|.
             Now n*2^(-w) can always be written |(1+r)^n-1| for some
             |r|<=2^(-w), thus taking n=2^(err_s+1) we see that
             |S - s| <= |(1+r)^(2^(err_s+1))-1| * |s|, where S is the
             true value.
             In fact if ulp(s) <= ulp(S) the same inequality holds for
             |S| instead of |s| in the right hand side, i.e., we can
             write s = (1+r)^(2^(err_s+1)) * S.
             But if ulp(S) < ulp(s), we need to add one ``bit'' to the error,
             to get s = (1+r)^(2^(err_s+2)) * S. This is true since with
             E = n*2^(-w) we have |s - S| <= E * |s|, thus
             |s - S| <= E/(1-E) * |S|.
             Now E/(1-E) is bounded by 2E as long as E<=1/2,
             and 2E can be written (1+r)^(2n)-1 as above.
          */
          err_s += 2; /* exponent of relative error */

          mpfr_sub_ui (v, z0, 1, MPFR_RNDN); /* v = (x-1) * (1+r) */
          mpfr_mul (v, v, t, MPFR_RNDN); /* v = Pi*(x-1) * (1+r)^3 */
          mpfr_div (v, v, s, MPFR_RNDN); /* Pi*(x-1)/sin(Pi*(2-x)) */
          mpfr_abs (v, v, MPFR_RNDN);
          /* (1+r)^(3+2^err_s+1) */
          err_s = (err_s <= 1) ? 3 : err_s + 1;
          /* now (1+r)^M with M <= 2^err_s */
          mpfr_log (v, v, MPFR_RNDN);
          /* log(v*(1+e)) = log(v)+log(1+e) where |e| <= 2^(err_s-w).
             Since |log(1+e)| <= 2*e for |e| <= 1/4, the error on v is
             bounded by ulp(v)/2 + 2^(err_s+1-w). */
          if (err_s + 2 > w)
            {
              w += err_s + 2;
            }
          else
            {
              err_s += 1 - MPFR_GET_EXP(v);
              err_s = (err_s >= 0) ? err_s + 1 : 0;
              /* the error on v is bounded by 2^err_s ulps */
              err_u += MPFR_GET_EXP(u); /* absolute error on u */
              err_s += MPFR_GET_EXP(v); /* absolute error on v */
              mpfr_sub (s, v, u, MPFR_RNDN);
              /* the total error on s is bounded by ulp(s)/2 + 2^(err_u-w)
                 + 2^(err_s-w) <= ulp(s)/2 + 2^(max(err_u,err_s)+1-w) */
              err_s = (err_s >= err_u) ? err_s : err_u;
              err_s += 1 - MPFR_GET_EXP(s); /* error is 2^err_s ulp(s) */
              err_s = (err_s >= 0) ? err_s + 1 : 0;
              if (mpfr_can_round (s, w - err_s, MPFR_RNDN, MPFR_RNDZ, precy
                                  + (rnd == MPFR_RNDN)))
                goto end;
            }
          MPFR_ZIV_NEXT (loop, w);
        }
      MPFR_ZIV_FREE (loop);
    }

  /* now z0 > 1 */

  MPFR_ASSERTD (compared > 0);

  /* since k is O(w), the value of log(z0*...*(z0+k-1)) is about w*log(w),
     so there is a cancellation of ~log(w) in the argument reconstruction */
  w = precy + MPFR_INT_CEIL_LOG2 (precy);
  w += MPFR_INT_CEIL_LOG2 (w) + 13;
  MPFR_ZIV_INIT (loop, w);
  while (1)
    {
      MPFR_ASSERTD (w >= 3);

      /* argument reduction: we compute gamma(z0 + k), where the series
         has error term B_{2n}/(z0+k)^(2n) ~ (n/(Pi*e*(z0+k)))^(2n)
         and we need k steps of argument reconstruction. Assuming k is large
         with respect to z0, and k = n, we get 1/(Pi*e)^(2n) ~ 2^(-w), i.e.,
         k ~ w*log(2)/2/log(Pi*e) ~ 0.1616 * w.
         However, since the series is more expensive to compute, the optimal
         value seems to be k ~ 4.5 * w experimentally. */
      mpfr_set_prec (s, 53);
      mpfr_gamma_alpha (s, w);
      mpfr_set_ui_2exp (s, 9, -1, MPFR_RNDU);
      mpfr_mul_ui (s, s, w, MPFR_RNDU);
      if (mpfr_cmp (z0, s) < 0)
        {
          mpfr_sub (s, s, z0, MPFR_RNDU);
          k = mpfr_get_ui (s, MPFR_RNDU);
          if (k < 3)
            k = 3;
        }
      else
        k = 3;

      mpfr_set_prec (s, w);
      mpfr_set_prec (t, w);
      mpfr_set_prec (u, w);
      mpfr_set_prec (v, w);
      mpfr_set_prec (z, w);

      mpfr_add_ui (z, z0, k, MPFR_RNDN);
      /* z = (z0+k)*(1+t1) with |t1| <= 2^(-w) */

      /* z >= 4 ensures the relative error on log(z) is small,
         and also (z-1/2)*log(z)-z >= 0 */
      MPFR_ASSERTD (mpfr_cmp_ui (z, 4) >= 0);

      mpfr_log (s, z, MPFR_RNDN); /* log(z) */
      /* we have s = log((z0+k)*(1+t1))*(1+t2) with |t1|, |t2| <= 2^(-w).
         Since w >= 2 and z0+k >= 4, we can write log((z0+k)*(1+t1))
         = log(z0+k) * (1+t3) with |t3| <= 2^(-w), thus we have
         s = log(z0+k) * (1+t4)^2 with |t4| <= 2^(-w) */
      mpfr_mul_2ui (t, z, 1, MPFR_RNDN); /* t = 2z * (1+t5) */
      mpfr_sub_ui (t, t, 1, MPFR_RNDN); /* t = 2z-1 * (1+t6)^3 */
      /* since we can write 2z*(1+t5) = (2z-1)*(1+t5') with
         t5' = 2z/(2z-1) * t5, thus |t5'| <= 8/7 * t5 */
      mpfr_mul (s, s, t, MPFR_RNDN); /* (2z-1)*log(z) * (1+t7)^6 */
      mpfr_div_2ui (s, s, 1, MPFR_RNDN); /* (z-1/2)*log(z) * (1+t7)^6 */
      mpfr_sub (s, s, z, MPFR_RNDN); /* (z-1/2)*log(z)-z */
      /* s = [(z-1/2)*log(z)-z]*(1+u)^14, s >= 1/2 */

      mpfr_ui_div (u, 1, z, MPFR_RNDN); /* 1/z * (1+u), u <= 1/4 since z >= 4 */

      /* the first term is B[2]/2/z = 1/12/z: t=1/12/z, C[2]=1 */
      mpfr_div_ui (t, u, 12, MPFR_RNDN); /* 1/(12z) * (1+u)^2, t <= 3/128 */
      mpfr_set (v, t, MPFR_RNDN);        /* (1+u)^2, v < 2^(-5) */
      mpfr_add (s, s, v, MPFR_RNDN);     /* (1+u)^15 */

      mpfr_mul (u, u, u, MPFR_RNDN); /* 1/z^2 * (1+u)^3 */

      if (Bm == 0)
        {
          B = mpfr_bernoulli_internal ((mpz_t *) 0, 0);
          B = mpfr_bernoulli_internal (B, 1);
          Bm = 2;
        }

      /* m <= maxm ensures that 2*m*(2*m+1) <= ULONG_MAX */
      maxm = 1UL << (GMP_NUMB_BITS / 2 - 1);

      /* s:(1+u)^15, t:(1+u)^2, t <= 3/128 */

      for (m = 2; MPFR_GET_EXP(v) + (mpfr_exp_t) w >= MPFR_GET_EXP(s); m++)
        {
          mpfr_mul (t, t, u, MPFR_RNDN); /* (1+u)^(10m-14) */
          if (m <= maxm)
            {
              mpfr_mul_ui (t, t, 2*(m-1)*(2*m-3), MPFR_RNDN);
              mpfr_div_ui (t, t, 2*m*(2*m-1), MPFR_RNDN);
              mpfr_div_ui (t, t, 2*m*(2*m+1), MPFR_RNDN);
            }
          else
            {
              mpfr_mul_ui (t, t, 2*(m-1), MPFR_RNDN);
              mpfr_mul_ui (t, t, 2*m-3, MPFR_RNDN);
              mpfr_div_ui (t, t, 2*m, MPFR_RNDN);
              mpfr_div_ui (t, t, 2*m-1, MPFR_RNDN);
              mpfr_div_ui (t, t, 2*m, MPFR_RNDN);
              mpfr_div_ui (t, t, 2*m+1, MPFR_RNDN);
            }
          /* (1+u)^(10m-8) */
          /* invariant: t=1/(2m)/(2m-1)/z^(2m-1)/(2m+1)! */
          if (Bm <= m)
            {
              B = mpfr_bernoulli_internal (B, m); /* B[2m]*(2m+1)!, exact */
              Bm ++;
            }
          mpfr_mul_z (v, t, B[m], MPFR_RNDN); /* (1+u)^(10m-7) */
          MPFR_ASSERTD(MPFR_GET_EXP(v) <= - (2 * m + 3));
          mpfr_add (s, s, v, MPFR_RNDN);
        }
      /* m <= 1/2*Pi*e*z ensures that |v[m]| < 1/2^(2m+3) */
      MPFR_ASSERTD ((double) m <= 4.26 * mpfr_get_d (z, MPFR_RNDZ));

      /* We have sum([(1+u)^(10m-7)-1]*1/2^(2m+3), m=2..infinity)
         <= 1.46*u for u <= 2^(-3).
         We have 0 < lngamma(z) - [(z - 1/2) ln(z) - z + 1/2 ln(2 Pi)] < 0.021
         for z >= 4, thus since the initial s >= 0.85, the different values of
         s differ by at most one binade, and the total rounding error on s
         in the for-loop is bounded by 2*(m-1)*ulp(final_s).
         The error coming from the v's is bounded by
         1.46*2^(-w) <= 2*ulp(final_s).
         Thus the total error so far is bounded by [(1+u)^15-1]*s+2m*ulp(s)
         <= (2m+47)*ulp(s).
         Taking into account the truncation error (which is bounded by the last
         term v[] according to 6.1.42 in A&S), the bound is (2m+48)*ulp(s).
      */

      /* add 1/2*log(2*Pi) and subtract log(z0*(z0+1)*...*(z0+k-1)) */
      mpfr_const_pi (v, MPFR_RNDN); /* v = Pi*(1+u) */
      mpfr_mul_2ui (v, v, 1, MPFR_RNDN); /* v = 2*Pi * (1+u) */
      if (k)
        {
          unsigned long l;
          mpfr_set (t, z0, MPFR_RNDN); /* t = z0*(1+u) */
          for (l = 1; l < k; l++)
            {
              mpfr_add_ui (u, z0, l, MPFR_RNDN); /* u = (z0+l)*(1+u) */
              mpfr_mul (t, t, u, MPFR_RNDN);     /* (1+u)^(2l+1) */
            }
          /* now t: (1+u)^(2k-1) */
          /* instead of computing log(sqrt(2*Pi)/t), we compute
             1/2*log(2*Pi/t^2), which trades a square root for a square */
          mpfr_mul (t, t, t, MPFR_RNDN); /* (z0*...*(z0+k-1))^2, (1+u)^(4k-1) */
          mpfr_div (v, v, t, MPFR_RNDN);
          /* 2*Pi/(z0*...*(z0+k-1))^2 (1+u)^(4k+1) */
        }
#ifdef IS_GAMMA
      err_s = MPFR_GET_EXP(s);
      mpfr_exp (s, s, MPFR_RNDN);
      /* If s is +Inf, we compute exp(lngamma(z0)). */
      if (mpfr_inf_p (s))
        {
          inexact = mpfr_explgamma (y, z0, &expo, s, t, rnd);
          if (inexact)
            goto end0;
          else
            goto ziv_next;
        }
      /* before the exponential, we have s = s0 + h where
         |h| <= (2m+48)*ulp(s), thus exp(s0) = exp(s) * exp(-h).
         For |h| <= 1/4, we have |exp(h)-1| <= 1.2*|h| thus
         |exp(s) - exp(s0)| <= 1.2 * exp(s) * (2m+48)* 2^(EXP(s)-w). */
      d = 1.2 * (2.0 * (double) m + 48.0);
      /* the error on s is bounded by d*2^err_s * 2^(-w) */
      mpfr_sqrt (t, v, MPFR_RNDN);
      /* let v0 be the exact value of v. We have v = v0*(1+u)^(4k+1),
         thus t = sqrt(v0)*(1+u)^(2k+3/2). */
      mpfr_mul (s, s, t, MPFR_RNDN);
      /* the error on input s is bounded by (1+u)^(d*2^err_s),
         and that on t is (1+u)^(2k+3/2), thus the
         total error is (1+u)^(d*2^err_s+2k+5/2) */
      err_s += __gmpfr_ceil_log2 (d);
      err_t = __gmpfr_ceil_log2 (2.0 * (double) k + 2.5);
      err_s = (err_s >= err_t) ? err_s + 1 : err_t + 1;
#else
      mpfr_log (t, v, MPFR_RNDN);
      /* let v0 be the exact value of v. We have v = v0*(1+u)^(4k+1),
         thus log(v) = log(v0) + (4k+1)*log(1+u). Since |log(1+u)/u| <= 1.07
         for |u| <= 2^(-3), the absolute error on log(v) is bounded by
         1.07*(4k+1)*u, and the rounding error by ulp(t). */
      mpfr_div_2ui (t, t, 1, MPFR_RNDN);
      /* the error on t is now bounded by ulp(t) + 0.54*(4k+1)*2^(-w).
         We have sqrt(2*Pi)/(z0*(z0+1)*...*(z0+k-1)) <= sqrt(2*Pi)/k! <= 0.5
         since k>=3, thus t <= -0.5 and ulp(t) >= 2^(-w).
         Thus the error on t is bounded by (2.16*k+1.54)*ulp(t). */
      err_t = MPFR_GET_EXP(t) + (mpfr_exp_t)
        __gmpfr_ceil_log2 (2.2 * (double) k + 1.6);
      err_s = MPFR_GET_EXP(s) + (mpfr_exp_t)
        __gmpfr_ceil_log2 (2.0 * (double) m + 48.0);
      mpfr_add (s, s, t, MPFR_RNDN); /* this is a subtraction in fact */
      /* the final error in ulp(s) is
         <= 1 + 2^(err_t-EXP(s)) + 2^(err_s-EXP(s))
         <= 2^(1+max(err_t,err_s)-EXP(s)) if err_t <> err_s
         <= 2^(2+max(err_t,err_s)-EXP(s)) if err_t = err_s */
      err_s = (err_t == err_s) ? 1 + err_s : ((err_t > err_s) ? err_t : err_s);
      err_s += 1 - MPFR_GET_EXP(s);
#endif
      if (MPFR_LIKELY (MPFR_CAN_ROUND (s, w - err_s, precy, rnd)))
        break;
#ifdef IS_GAMMA
    ziv_next:
#endif
      MPFR_ZIV_NEXT (loop, w);
    }

#ifdef IS_GAMMA
 end0:
#endif
  oldBm = Bm;
  while (Bm--)
    mpz_clear (B[Bm]);
  (*__gmp_free_func) (B, oldBm * sizeof (mpz_t));

 end:
  if (inexact == 0)
    inexact = mpfr_set (y, s, rnd);
  MPFR_ZIV_FREE (loop);

  mpfr_clear (s);
  mpfr_clear (t);
  mpfr_clear (u);
  mpfr_clear (v);
  mpfr_clear (z);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, inexact, rnd);
}

#ifndef IS_GAMMA

int
mpfr_lngamma (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  int inex;

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd),
     ("y[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (y), mpfr_log_prec, y, inex));

  /* special cases */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x) ||
                     (MPFR_IS_NEG (x) && mpfr_integer_p (x))))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else /* lngamma(+/-Inf) = lngamma(nonpositive integer) = +Inf */
        {
          if (!MPFR_IS_INF (x))
            mpfr_set_divby0 ();
          MPFR_SET_INF (y);
          MPFR_SET_POS (y);
          MPFR_RET (0);  /* exact */
        }
    }

  /* if -2k-1 < x < -2k <= 0, then lngamma(x) = NaN */
  if (MPFR_IS_NEG (x) && unit_bit (x) == 0)
    {
      MPFR_SET_NAN (y);
      MPFR_RET_NAN;
    }

  inex = mpfr_lngamma_aux (y, x, rnd);
  return inex;
}

int
mpfr_lgamma (mpfr_ptr y, int *signp, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  int inex;

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd),
     ("y[%Pu]=%.*Rg signp=%d inexact=%d",
      mpfr_get_prec (y), mpfr_log_prec, y, *signp, inex));

  *signp = 1;  /* most common case */

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else
        {
          if (MPFR_IS_ZERO (x))
            mpfr_set_divby0 ();
          *signp = MPFR_INT_SIGN (x);
          MPFR_SET_INF (y);
          MPFR_SET_POS (y);
          MPFR_RET (0);
        }
    }

  if (MPFR_IS_NEG (x))
    {
      if (mpfr_integer_p (x))
        {
          MPFR_SET_INF (y);
          MPFR_SET_POS (y);
          mpfr_set_divby0 ();
          MPFR_RET (0);
        }

      if (unit_bit (x) == 0)
        *signp = -1;

      /* For tiny negative x, we have gamma(x) = 1/x - euler + O(x),
         thus |gamma(x)| = -1/x + euler + O(x), and
         log |gamma(x)| = -log(-x) - euler*x + O(x^2).
         More precisely we have for -0.4 <= x < 0:
         -log(-x) <= log |gamma(x)| <= -log(-x) - x.
         Since log(x) is not representable, we may have an instance of the
         Table Maker Dilemma. The only way to ensure correct rounding is to
         compute an interval [l,h] such that l <= -log(-x) and
         -log(-x) - x <= h, and check whether l and h round to the same number
         for the target precision and rounding modes. */
      if (MPFR_EXP(x) + 1 <= - (mpfr_exp_t) MPFR_PREC(y))
        /* since PREC(y) >= 1, this ensures EXP(x) <= -2,
           thus |x| <= 0.25 < 0.4 */
        {
          mpfr_t l, h;
          int ok, inex2;
          mpfr_prec_t w = MPFR_PREC (y) + 14;
          mpfr_exp_t expl;

          while (1)
            {
              mpfr_init2 (l, w);
              mpfr_init2 (h, w);
              /* we want a lower bound on -log(-x), thus an upper bound
                 on log(-x), thus an upper bound on -x. */
              mpfr_neg (l, x, MPFR_RNDU); /* upper bound on -x */
              mpfr_log (l, l, MPFR_RNDU); /* upper bound for log(-x) */
              mpfr_neg (l, l, MPFR_RNDD); /* lower bound for -log(-x) */
              mpfr_neg (h, x, MPFR_RNDD); /* lower bound on -x */
              mpfr_log (h, h, MPFR_RNDD); /* lower bound on log(-x) */
              mpfr_neg (h, h, MPFR_RNDU); /* upper bound for -log(-x) */
              mpfr_sub (h, h, x, MPFR_RNDU); /* upper bound for -log(-x) - x */
              inex = mpfr_prec_round (l, MPFR_PREC (y), rnd);
              inex2 = mpfr_prec_round (h, MPFR_PREC (y), rnd);
              /* Caution: we not only need l = h, but both inexact flags
                 should agree. Indeed, one of the inexact flags might be
                 zero. In that case if we assume ln|gamma(x)| cannot be
                 exact, the other flag should be correct. We are conservative
                 here and request that both inexact flags agree. */
              ok = SAME_SIGN (inex, inex2) && mpfr_equal_p (l, h);
              if (ok)
                mpfr_set (y, h, rnd); /* exact */
              else
                expl = MPFR_EXP (l);
              mpfr_clear (l);
              mpfr_clear (h);
              if (ok)
                return inex;
              /* if ulp(log(-x)) <= |x| there is no reason to loop,
                 since the width of [l, h] will be at least |x| */
              if (expl < MPFR_EXP(x) + (mpfr_exp_t) w)
                break;
              w += MPFR_INT_CEIL_LOG2(w) + 3;
            }
        }
    }

  inex = mpfr_lngamma_aux (y, x, rnd);
  return inex;
}

#endif
