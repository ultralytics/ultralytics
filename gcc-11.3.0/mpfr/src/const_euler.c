/* mpfr_const_euler -- Euler's constant

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

/* Declare the cache */
MPFR_DECL_INIT_CACHE (__gmpfr_cache_const_euler, mpfr_const_euler_internal)

/* Set User Interface */
#undef mpfr_const_euler
int
mpfr_const_euler (mpfr_ptr x, mpfr_rnd_t rnd_mode) {
  return mpfr_cache (x, __gmpfr_cache_const_euler, rnd_mode);
}


static void mpfr_const_euler_S2 (mpfr_ptr, unsigned long);
static void mpfr_const_euler_R (mpfr_ptr, unsigned long);

int
mpfr_const_euler_internal (mpfr_t x, mpfr_rnd_t rnd)
{
  mpfr_prec_t prec = MPFR_PREC(x), m, log2m;
  mpfr_t y, z;
  unsigned long n;
  int inexact;
  MPFR_ZIV_DECL (loop);

  log2m = MPFR_INT_CEIL_LOG2 (prec);
  m = prec + 2 * log2m + 23;

  mpfr_init2 (y, m);
  mpfr_init2 (z, m);

  MPFR_ZIV_INIT (loop, m);
  for (;;)
    {
      mpfr_exp_t exp_S, err;
      /* since prec >= 1, we have m >= 24 here, which ensures n >= 9 below */
      n = 1 + (unsigned long) ((double) m * LOG2 / 2.0);
      MPFR_ASSERTD (n >= 9);
      mpfr_const_euler_S2 (y, n); /* error <= 3 ulps */
      exp_S = MPFR_EXP(y);
      mpfr_set_ui (z, n, MPFR_RNDN);
      mpfr_log (z, z, MPFR_RNDD); /* error <= 1 ulp */
      mpfr_sub (y, y, z, MPFR_RNDN); /* S'(n) - log(n) */
      /* the error is less than 1/2 + 3*2^(exp_S-EXP(y)) + 2^(EXP(z)-EXP(y))
         <= 1/2 + 2^(exp_S+2-EXP(y)) + 2^(EXP(z)-EXP(y))
         <= 1/2 + 2^(1+MAX(exp_S+2,EXP(z))-EXP(y)) */
      err = 1 + MAX(exp_S + 2, MPFR_EXP(z)) - MPFR_EXP(y);
      err = (err >= -1) ? err + 1 : 0; /* error <= 2^err ulp(y) */
      exp_S = MPFR_EXP(y);
      mpfr_const_euler_R (z, n); /* err <= ulp(1/2) = 2^(-m) */
      mpfr_sub (y, y, z, MPFR_RNDN);
      /* err <= 1/2 ulp(y) + 2^(-m) + 2^(err + exp_S - EXP(y)) ulp(y).
         Since the result is between 0.5 and 1, ulp(y) = 2^(-m).
         So we get 3/2*ulp(y) + 2^(err + exp_S - EXP(y)) ulp(y).
         3/2 + 2^e <= 2^(e+1) for e>=1, and <= 2^2 otherwise */
      err = err + exp_S - MPFR_EXP(y);
      err = (err >= 1) ? err + 1 : 2;
      if (MPFR_LIKELY (MPFR_CAN_ROUND (y, m - err, prec, rnd)))
        break;
      MPFR_ZIV_NEXT (loop, m);
      mpfr_set_prec (y, m);
      mpfr_set_prec (z, m);
    }
  MPFR_ZIV_FREE (loop);

  inexact = mpfr_set (x, y, rnd);

  mpfr_clear (y);
  mpfr_clear (z);

  return inexact; /* always inexact */
}

static void
mpfr_const_euler_S2_aux (mpz_t P, mpz_t Q, mpz_t T, unsigned long n,
                         unsigned long a, unsigned long b, int need_P)
{
  if (a + 1 == b)
    {
      mpz_set_ui (P, n);
      if (a > 1)
        mpz_mul_si (P, P, 1 - (long) a);
      mpz_set (T, P);
      mpz_set_ui (Q, a);
      mpz_mul_ui (Q, Q, a);
    }
  else
    {
      unsigned long c = (a + b) / 2;
      mpz_t P2, Q2, T2;
      mpfr_const_euler_S2_aux (P, Q, T, n, a, c, 1);
      mpz_init (P2);
      mpz_init (Q2);
      mpz_init (T2);
      mpfr_const_euler_S2_aux (P2, Q2, T2, n, c, b, 1);
      mpz_mul (T, T, Q2);
      mpz_mul (T2, T2, P);
      mpz_add (T, T, T2);
      if (need_P)
        mpz_mul (P, P, P2);
      mpz_mul (Q, Q, Q2);
      mpz_clear (P2);
      mpz_clear (Q2);
      mpz_clear (T2);
      /* divide by 2 if possible */
      {
        unsigned long v2;
        v2 = mpz_scan1 (P, 0);
        c = mpz_scan1 (Q, 0);
        if (c < v2)
          v2 = c;
        c = mpz_scan1 (T, 0);
        if (c < v2)
          v2 = c;
        if (v2)
          {
            mpz_tdiv_q_2exp (P, P, v2);
            mpz_tdiv_q_2exp (Q, Q, v2);
            mpz_tdiv_q_2exp (T, T, v2);
          }
      }
    }
}

/* computes S(n) = sum(n^k*(-1)^(k-1)/k!/k, k=1..ceil(4.319136566 * n))
   using binary splitting.
   We have S(n) = sum(f(k), k=1..N) with N=ceil(4.319136566 * n)
   and f(k) = n^k*(-1)*(k-1)/k!/k,
   thus f(k)/f(k-1) = -n*(k-1)/k^2
*/
static void
mpfr_const_euler_S2 (mpfr_t x, unsigned long n)
{
  mpz_t P, Q, T;
  unsigned long N = (unsigned long) (ALPHA * (double) n + 1.0);
  mpz_init (P);
  mpz_init (Q);
  mpz_init (T);
  mpfr_const_euler_S2_aux (P, Q, T, n, 1, N + 1, 0);
  mpfr_set_z (x, T, MPFR_RNDN);
  mpfr_div_z (x, x, Q, MPFR_RNDN);
  mpz_clear (P);
  mpz_clear (Q);
  mpz_clear (T);
}

/* computes R(n) = exp(-n)/n * sum(k!/(-n)^k, k=0..n-2)
   with error at most 4*ulp(x). Assumes n>=2.
   Since x <= exp(-n)/n <= 1/8, then 4*ulp(x) <= ulp(1).
*/
static void
mpfr_const_euler_R (mpfr_t x, unsigned long n)
{
  unsigned long k, m;
  mpz_t a, s;
  mpfr_t y;

  MPFR_ASSERTN (n >= 2); /* ensures sum(k!/(-n)^k, k=0..n-2) >= 2/3 */

  /* as we multiply the sum by exp(-n), we need only PREC(x) - n/LOG2 bits */
  m = MPFR_PREC(x) - (unsigned long) ((double) n / LOG2);

  mpz_init_set_ui (a, 1);
  mpz_mul_2exp (a, a, m);
  mpz_init_set (s, a);

  for (k = 1; k <= n; k++)
    {
      mpz_mul_ui (a, a, k);
      mpz_fdiv_q_ui (a, a, n);
      /* the error e(k) on a is e(k) <= 1 + k/n*e(k-1) with e(0)=0,
         i.e. e(k) <= k */
      if (k % 2)
        mpz_sub (s, s, a);
      else
        mpz_add (s, s, a);
    }
  /* the error on s is at most 1+2+...+n = n*(n+1)/2 */
  mpz_fdiv_q_ui (s, s, n); /* err <= 1 + (n+1)/2 */
  MPFR_ASSERTN (MPFR_PREC(x) >= mpz_sizeinbase(s, 2));
  mpfr_set_z (x, s, MPFR_RNDD); /* exact */
  mpfr_div_2ui (x, x, m, MPFR_RNDD);
  /* now x = 1/n * sum(k!/(-n)^k, k=0..n-2) <= 1/n */
  /* err(x) <= (n+1)/2^m <= (n+1)*exp(n)/2^PREC(x) */

  mpfr_init2 (y, m);
  mpfr_set_si (y, -(long)n, MPFR_RNDD); /* assumed exact */
  mpfr_exp (y, y, MPFR_RNDD); /* err <= ulp(y) <= exp(-n)*2^(1-m) */
  mpfr_mul (x, x, y, MPFR_RNDD);
  /* err <= ulp(x) + (n + 1 + 2/n) / 2^prec(x)
     <= ulp(x) + (n + 1 + 2/n) ulp(x)/x since x*2^(-prec(x)) < ulp(x)
     <= ulp(x) + (n + 1 + 2/n) 3/(2n) ulp(x) since x >= 2/3*n for n >= 2
     <= 4 * ulp(x) for n >= 2 */
  mpfr_clear (y);

  mpz_clear (a);
  mpz_clear (s);
}
