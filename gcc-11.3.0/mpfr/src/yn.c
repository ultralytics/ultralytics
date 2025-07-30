/* mpfr_y0, mpfr_y1, mpfr_yn -- Bessel functions of 2nd kind, integer order.
   http://www.opengroup.org/onlinepubs/009695399/functions/y0.html

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

static int mpfr_yn_asympt (mpfr_ptr, long, mpfr_srcptr, mpfr_rnd_t);

int
mpfr_y0 (mpfr_ptr res, mpfr_srcptr z, mpfr_rnd_t r)
{
  return mpfr_yn (res, 0, z, r);
}

int
mpfr_y1 (mpfr_ptr res, mpfr_srcptr z, mpfr_rnd_t r)
{
  return mpfr_yn (res, 1, z, r);
}

/* compute in s an approximation of S1 = sum((n-k)!/k!*y^k,k=0..n)
   return e >= 0 the exponent difference between the maximal value of |s|
   during the for loop and the final value of |s|.
*/
static mpfr_exp_t
mpfr_yn_s1 (mpfr_ptr s, mpfr_srcptr y, unsigned long n)
{
  unsigned long k;
  mpz_t f;
  mpfr_exp_t e, emax;

  mpz_init_set_ui (f, 1);
  /* we compute n!*S1 = sum(a[k]*y^k,k=0..n) where a[k] = n!*(n-k)!/k!,
     a[0] = (n!)^2, a[1] = n!*(n-1)!, ..., a[n-1] = n, a[n] = 1 */
  mpfr_set_ui (s, 1, MPFR_RNDN); /* a[n] */
  emax = MPFR_EXP(s);
  for (k = n; k-- > 0;)
    {
      /* a[k]/a[k+1] = (n-k)!/k!/(n-(k+1))!*(k+1)! = (k+1)*(n-k) */
      mpfr_mul (s, s, y, MPFR_RNDN);
      mpz_mul_ui (f, f, n - k);
      mpz_mul_ui (f, f, k + 1);
      /* invariant: f = a[k] */
      mpfr_add_z (s, s, f, MPFR_RNDN);
      e = MPFR_EXP(s);
      if (e > emax)
        emax = e;
    }
  /* now we have f = (n!)^2 */
  mpz_sqrt (f, f);
  mpfr_div_z (s, s, f, MPFR_RNDN);
  mpz_clear (f);
  return emax - MPFR_EXP(s);
}

/* compute in s an approximation of
   S3 = c*sum((h(k)+h(n+k))*y^k/k!/(n+k)!,k=0..infinity)
   where h(k) = 1 + 1/2 + ... + 1/k
   k=0: h(n)
   k=1: 1+h(n+1)
   k=2: 3/2+h(n+2)
   Returns e such that the error is bounded by 2^e ulp(s).
*/
static mpfr_exp_t
mpfr_yn_s3 (mpfr_ptr s, mpfr_srcptr y, mpfr_srcptr c, unsigned long n)
{
  unsigned long k, zz;
  mpfr_t t, u;
  mpz_t p, q; /* p/q will store h(k)+h(n+k) */
  mpfr_exp_t exps, expU;

  zz = mpfr_get_ui (y, MPFR_RNDU); /* y = z^2/4 */
  MPFR_ASSERTN (zz < ULONG_MAX - 2);
  zz += 2; /* z^2 <= 2^zz */
  mpz_init_set_ui (p, 0);
  mpz_init_set_ui (q, 1);
  /* initialize p/q to h(n) */
  for (k = 1; k <= n; k++)
    {
      /* p/q + 1/k = (k*p+q)/(q*k) */
      mpz_mul_ui (p, p, k);
      mpz_add (p, p, q);
      mpz_mul_ui (q, q, k);
    }
  mpfr_init2 (t, MPFR_PREC(s));
  mpfr_init2 (u, MPFR_PREC(s));
  mpfr_fac_ui (t, n, MPFR_RNDN);
  mpfr_div (t, c, t, MPFR_RNDN);    /* c/n! */
  mpfr_mul_z (u, t, p, MPFR_RNDN);
  mpfr_div_z (s, u, q, MPFR_RNDN);
  exps = MPFR_EXP (s);
  expU = exps;
  for (k = 1; ;k ++)
    {
      /* update t */
      mpfr_mul (t, t, y, MPFR_RNDN);
      mpfr_div_ui (t, t, k, MPFR_RNDN);
      mpfr_div_ui (t, t, n + k, MPFR_RNDN);
      /* update p/q:
         p/q + 1/k + 1/(n+k) = [p*k*(n+k) + q*(n+k) + q*k]/(q*k*(n+k)) */
      mpz_mul_ui (p, p, k);
      mpz_mul_ui (p, p, n + k);
      mpz_addmul_ui (p, q, n + 2 * k);
      mpz_mul_ui (q, q, k);
      mpz_mul_ui (q, q, n + k);
      mpfr_mul_z (u, t, p, MPFR_RNDN);
      mpfr_div_z (u, u, q, MPFR_RNDN);
      exps = MPFR_EXP (u);
      if (exps > expU)
        expU = exps;
      mpfr_add (s, s, u, MPFR_RNDN);
      exps = MPFR_EXP (s);
      if (exps > expU)
        expU = exps;
      if (MPFR_EXP (u) + (mpfr_exp_t) MPFR_PREC (u) < MPFR_EXP (s) &&
          zz / (2 * k) < k + n)
        break;
    }
  mpfr_clear (t);
  mpfr_clear (u);
  mpz_clear (p);
  mpz_clear (q);
  exps = expU - MPFR_EXP (s);
  /* the error is bounded by (6k^2+33/2k+11) 2^exps ulps
     <= 8*(k+2)^2 2^exps ulps */
  return 3 + 2 * MPFR_INT_CEIL_LOG2(k + 2) + exps;
}

int
mpfr_yn (mpfr_ptr res, long n, mpfr_srcptr z, mpfr_rnd_t r)
{
  int inex;
  unsigned long absn;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("n=%ld x[%Pu]=%.*Rg rnd=%d", n, mpfr_get_prec (z), mpfr_log_prec, z, r),
     ("y[%Pu]=%.*Rg inexact=%d", mpfr_get_prec (res), mpfr_log_prec, res, inex));

  absn = SAFE_ABS (unsigned long, n);

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (z)))
    {
      if (MPFR_IS_NAN (z))
        {
          MPFR_SET_NAN (res); /* y(n,NaN) = NaN */
          MPFR_RET_NAN;
        }
      /* y(n,z) tends to zero when z goes to +Inf, oscillating around
         0. We choose to return +0 in that case. */
      else if (MPFR_IS_INF (z))
        {
          if (MPFR_SIGN(z) > 0)
            return mpfr_set_ui (res, 0, r);
          else /* y(n,-Inf) = NaN */
            {
              MPFR_SET_NAN (res);
              MPFR_RET_NAN;
            }
        }
      else /* y(n,z) tends to -Inf for n >= 0 or n even, to +Inf otherwise,
              when z goes to zero */
        {
          MPFR_SET_INF(res);
          if (n >= 0 || ((unsigned long) n & 1) == 0)
            MPFR_SET_NEG(res);
          else
            MPFR_SET_POS(res);
          mpfr_set_divby0 ();
          MPFR_RET(0);
        }
    }

  /* for z < 0, y(n,z) is imaginary except when j(n,|z|) = 0, which we
     assume does not happen for a rational z. */
  if (MPFR_SIGN(z) < 0)
    {
      MPFR_SET_NAN (res);
      MPFR_RET_NAN;
    }

  /* now z is not singular, and z > 0 */

  MPFR_SAVE_EXPO_MARK (expo);

  /* Deal with tiny arguments. We have:
     y0(z) = 2 log(z)/Pi + 2 (euler - log(2))/Pi + O(log(z)*z^2), more
     precisely for 0 <= z <= 1/2, with g(z) = 2/Pi + 2(euler-log(2))/Pi/log(z),
                g(z) - 0.41*z^2 < y0(z)/log(z) < g(z)
     thus since log(z) is negative:
             g(z)*log(z) < y0(z) < (g(z) - z^2/2)*log(z)
     and since |g(z)| >= 0.63 for 0 <= z <= 1/2, the relative error on
     y0(z)/log(z) is bounded by 0.41*z^2/0.63 <= 0.66*z^2.
     Note: we use both the main term in log(z) and the constant term, because
     otherwise the relative error would be only in 1/log(|log(z)|).
  */
  if (n == 0 && MPFR_EXP(z) < - (mpfr_exp_t) (MPFR_PREC(res) / 2))
    {
      mpfr_t l, h, t, logz;
      mpfr_prec_t prec;
      int ok, inex2;

      prec = MPFR_PREC(res) + 10;
      mpfr_init2 (l, prec);
      mpfr_init2 (h, prec);
      mpfr_init2 (t, prec);
      mpfr_init2 (logz, prec);
      /* first enclose log(z) + euler - log(2) = log(z/2) + euler */
      mpfr_log (logz, z, MPFR_RNDD);    /* lower bound of log(z) */
      mpfr_set (h, logz, MPFR_RNDU);    /* exact */
      mpfr_nextabove (h);              /* upper bound of log(z) */
      mpfr_const_euler (t, MPFR_RNDD);  /* lower bound of euler */
      mpfr_add (l, logz, t, MPFR_RNDD); /* lower bound of log(z) + euler */
      mpfr_nextabove (t);              /* upper bound of euler */
      mpfr_add (h, h, t, MPFR_RNDU);    /* upper bound of log(z) + euler */
      mpfr_const_log2 (t, MPFR_RNDU);   /* upper bound of log(2) */
      mpfr_sub (l, l, t, MPFR_RNDD);    /* lower bound of log(z/2) + euler */
      mpfr_nextbelow (t);              /* lower bound of log(2) */
      mpfr_sub (h, h, t, MPFR_RNDU);    /* upper bound of log(z/2) + euler */
      mpfr_const_pi (t, MPFR_RNDU);     /* upper bound of Pi */
      mpfr_div (l, l, t, MPFR_RNDD);    /* lower bound of (log(z/2)+euler)/Pi */
      mpfr_nextbelow (t);              /* lower bound of Pi */
      mpfr_div (h, h, t, MPFR_RNDD);    /* upper bound of (log(z/2)+euler)/Pi */
      mpfr_mul_2ui (l, l, 1, MPFR_RNDD); /* lower bound on g(z)*log(z) */
      mpfr_mul_2ui (h, h, 1, MPFR_RNDU); /* upper bound on g(z)*log(z) */
      /* we now have l <= g(z)*log(z) <= h, and we need to add -z^2/2*log(z)
         to h */
      mpfr_mul (t, z, z, MPFR_RNDU);     /* upper bound on z^2 */
      /* since logz is negative, a lower bound corresponds to an upper bound
         for its absolute value */
      mpfr_neg (t, t, MPFR_RNDD);
      mpfr_div_2ui (t, t, 1, MPFR_RNDD);
      mpfr_mul (t, t, logz, MPFR_RNDU); /* upper bound on z^2/2*log(z) */
      mpfr_add (h, h, t, MPFR_RNDU);
      inex = mpfr_prec_round (l, MPFR_PREC(res), r);
      inex2 = mpfr_prec_round (h, MPFR_PREC(res), r);
      /* we need h=l and inex=inex2 */
      ok = (inex == inex2) && mpfr_equal_p (l, h);
      if (ok)
        mpfr_set (res, h, r); /* exact */
      mpfr_clear (l);
      mpfr_clear (h);
      mpfr_clear (t);
      mpfr_clear (logz);
      if (ok)
        goto end;
    }

  /* small argument check for y1(z) = -2/Pi/z + O(log(z)):
     for 0 <= z <= 1, |y1(z) + 2/Pi/z| <= 0.25 */
  if (n == 1 && MPFR_EXP(z) + 1 < - (mpfr_exp_t) MPFR_PREC(res))
    {
      mpfr_t y;
      mpfr_prec_t prec;
      mpfr_exp_t err1;
      int ok;
      MPFR_BLOCK_DECL (flags);

      /* since 2/Pi > 0.5, and |y1(z)| >= |2/Pi/z|, if z <= 2^(-emax-1),
         then |y1(z)| > 2^emax */
      prec = MPFR_PREC(res) + 10;
      mpfr_init2 (y, prec);
      mpfr_const_pi (y, MPFR_RNDU); /* Pi*(1+u)^2, where here and below u
                                      represents a quantity <= 1/2^prec */
      mpfr_mul (y, y, z, MPFR_RNDU); /* Pi*z * (1+u)^4, upper bound */
      MPFR_BLOCK (flags, mpfr_ui_div (y, 2, y, MPFR_RNDZ));
      /* 2/Pi/z * (1+u)^6, lower bound, with possible overflow */
      if (MPFR_OVERFLOW (flags))
        {
          mpfr_clear (y);
          MPFR_SAVE_EXPO_FREE (expo);
          return mpfr_overflow (res, r, -1);
        }
      mpfr_neg (y, y, MPFR_RNDN);
      /* (1+u)^6 can be written 1+7u [for another value of u], thus the
         error on 2/Pi/z is less than 7ulp(y). The truncation error is less
         than 1/4, thus if ulp(y)>=1/4, the total error is less than 8ulp(y),
         otherwise it is less than 1/4+7/8 <= 2. */
      if (MPFR_EXP(y) + 2 >= MPFR_PREC(y)) /* ulp(y) >= 1/4 */
        err1 = 3;
      else /* ulp(y) <= 1/8 */
        err1 = (mpfr_exp_t) MPFR_PREC(y) - MPFR_EXP(y) + 1;
      ok = MPFR_CAN_ROUND (y, prec - err1, MPFR_PREC(res), r);
      if (ok)
        inex = mpfr_set (res, y, r);
      mpfr_clear (y);
      if (ok)
        goto end;
    }

  /* we can use the asymptotic expansion as soon as z > p log(2)/2,
     but to get some margin we use it for z > p/2 */
  if (mpfr_cmp_ui (z, MPFR_PREC(res) / 2 + 3) > 0)
    {
      inex = mpfr_yn_asympt (res, n, z, r);
      if (inex != 0)
        goto end;
    }

  /* General case */
  {
    mpfr_prec_t prec;
    mpfr_exp_t err1, err2, err3;
    mpfr_t y, s1, s2, s3;
    MPFR_ZIV_DECL (loop);

    mpfr_init (y);
    mpfr_init (s1);
    mpfr_init (s2);
    mpfr_init (s3);

    prec = MPFR_PREC(res) + 2 * MPFR_INT_CEIL_LOG2 (MPFR_PREC (res)) + 13;
    MPFR_ZIV_INIT (loop, prec);
    for (;;)
      {
        mpfr_set_prec (y, prec);
        mpfr_set_prec (s1, prec);
        mpfr_set_prec (s2, prec);
        mpfr_set_prec (s3, prec);

        mpfr_mul (y, z, z, MPFR_RNDN);
        mpfr_div_2ui (y, y, 2, MPFR_RNDN); /* z^2/4 */

        /* store (z/2)^n temporarily in s2 */
        mpfr_pow_ui (s2, z, absn, MPFR_RNDN);
        mpfr_div_2si (s2, s2, absn, MPFR_RNDN);

        /* compute S1 * (z/2)^(-n) */
        if (n == 0)
          {
            mpfr_set_ui (s1, 0, MPFR_RNDN);
            err1 = 0;
          }
        else
          err1 = mpfr_yn_s1 (s1, y, absn - 1);
        mpfr_div (s1, s1, s2, MPFR_RNDN); /* (z/2)^(-n) * S1 */
        /* See algorithms.tex: the relative error on s1 is bounded by
           (3n+3)*2^(e+1-prec). */
        err1 = MPFR_INT_CEIL_LOG2 (3 * absn + 3) + err1 + 1;
        /* rel_err(s1) <= 2^(err1-prec), thus err(s1) <= 2^err1 ulps */

        /* compute (z/2)^n * S3 */
        mpfr_neg (y, y, MPFR_RNDN); /* -z^2/4 */
        err3 = mpfr_yn_s3 (s3, y, s2, absn); /* (z/2)^n * S3 */
        /* the error on s3 is bounded by 2^err3 ulps */

        /* add s1+s3 */
        err1 += MPFR_EXP(s1);
        mpfr_add (s1, s1, s3, MPFR_RNDN);
        /* the error is bounded by 1/2 + 2^err1*2^(- EXP(s1))
           + 2^err3*2^(EXP(s3) - EXP(s1)) */
        err3 += MPFR_EXP(s3);
        err1 = (err3 > err1) ? err3 + 1 : err1 + 1;
        err1 -= MPFR_EXP(s1);
        err1 = (err1 >= 0) ? err1 + 1 : 1;
        /* now the error on s1 is bounded by 2^err1*ulp(s1) */

        /* compute S2 */
        mpfr_div_2ui (s2, z, 1, MPFR_RNDN); /* z/2 */
        mpfr_log (s2, s2, MPFR_RNDN); /* log(z/2) */
        mpfr_const_euler (s3, MPFR_RNDN);
        err2 = MPFR_EXP(s2) > MPFR_EXP(s3) ? MPFR_EXP(s2) : MPFR_EXP(s3);
        mpfr_add (s2, s2, s3, MPFR_RNDN); /* log(z/2) + gamma */
        err2 -= MPFR_EXP(s2);
        mpfr_mul_2ui (s2, s2, 1, MPFR_RNDN); /* 2*(log(z/2) + gamma) */
        mpfr_jn (s3, absn, z, MPFR_RNDN); /* Jn(z) */
        mpfr_mul (s2, s2, s3, MPFR_RNDN); /* 2*(log(z/2) + gamma)*Jn(z) */
        err2 += 4; /* the error on s2 is bounded by 2^err2 ulps, see
                      algorithms.tex */

        /* add all three sums */
        err1 += MPFR_EXP(s1); /* the error on s1 is bounded by 2^err1 */
        err2 += MPFR_EXP(s2); /* the error on s2 is bounded by 2^err2 */
        mpfr_sub (s2, s2, s1, MPFR_RNDN); /* s2 - (s1+s3) */
        err2 = (err1 > err2) ? err1 + 1 : err2 + 1;
        err2 -= MPFR_EXP(s2);
        err2 = (err2 >= 0) ? err2 + 1 : 1;
        /* now the error on s2 is bounded by 2^err2*ulp(s2) */
        mpfr_const_pi (y, MPFR_RNDN); /* error bounded by 1 ulp */
        mpfr_div (s2, s2, y, MPFR_RNDN); /* error bounded by
                                           2^(err2+1)*ulp(s2) */
        err2 ++;

        if (MPFR_LIKELY (MPFR_CAN_ROUND (s2, prec - err2, MPFR_PREC(res), r)))
          break;
        MPFR_ZIV_NEXT (loop, prec);
      }
    MPFR_ZIV_FREE (loop);

    /* Assume two's complement for the test n & 1 */
    inex = mpfr_set4 (res, s2, r, n >= 0 || (n & 1) == 0 ?
                      MPFR_SIGN (s2) : - MPFR_SIGN (s2));

    mpfr_clear (y);
    mpfr_clear (s1);
    mpfr_clear (s2);
    mpfr_clear (s3);
  }

 end:
  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (res, inex, r);
}

#define MPFR_YN
#include "jyn_asympt.c"
