/* mpfr_const_log2 -- compute natural logarithm of 2

Copyright 1999, 2001-2017 Free Software Foundation, Inc.
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
#ifndef MPFR_USE_LOGGING
MPFR_DECL_INIT_CACHE (__gmpfr_cache_const_log2, mpfr_const_log2_internal)
#else
MPFR_DECL_INIT_CACHE (__gmpfr_normal_log2, mpfr_const_log2_internal)
MPFR_DECL_INIT_CACHE (__gmpfr_logging_log2, mpfr_const_log2_internal)
MPFR_THREAD_VAR (mpfr_cache_ptr, __gmpfr_cache_const_log2, __gmpfr_normal_log2)
#endif

/* Set User interface */
#undef mpfr_const_log2
int
mpfr_const_log2 (mpfr_ptr x, mpfr_rnd_t rnd_mode) {
  return mpfr_cache (x, __gmpfr_cache_const_log2, rnd_mode);
}

/* Auxiliary function: Compute the terms from n1 to n2 (excluded)
   3/4*sum((-1)^n*n!^2/2^n/(2*n+1)!, n = n1..n2-1).
   Numerator is T[0], denominator is Q[0],
   Compute P[0] only when need_P is non-zero.
   Need 1+ceil(log(n2-n1)/log(2)) cells in T[],P[],Q[].
*/
static void
S (mpz_t *T, mpz_t *P, mpz_t *Q, unsigned long n1, unsigned long n2, int need_P)
{
  if (n2 == n1 + 1)
    {
      if (n1 == 0)
        mpz_set_ui (P[0], 3);
      else
        {
          mpz_set_ui (P[0], n1);
          mpz_neg (P[0], P[0]);
        }
      if (n1 <= (ULONG_MAX / 4 - 1) / 2)
        mpz_set_ui (Q[0], 4 * (2 * n1 + 1));
      else /* to avoid overflow in 4 * (2 * n1 + 1) */
        {
          mpz_set_ui (Q[0], n1);
          mpz_mul_2exp (Q[0], Q[0], 1);
          mpz_add_ui (Q[0], Q[0], 1);
          mpz_mul_2exp (Q[0], Q[0], 2);
        }
      mpz_set (T[0], P[0]);
    }
  else
    {
      unsigned long m = (n1 / 2) + (n2 / 2) + (n1 & 1UL & n2);
      unsigned long v, w;

      S (T, P, Q, n1, m, 1);
      S (T + 1, P + 1, Q + 1, m, n2, need_P);
      mpz_mul (T[0], T[0], Q[1]);
      mpz_mul (T[1], T[1], P[0]);
      mpz_add (T[0], T[0], T[1]);
      if (need_P)
        mpz_mul (P[0], P[0], P[1]);
      mpz_mul (Q[0], Q[0], Q[1]);

      /* remove common trailing zeroes if any */
      v = mpz_scan1 (T[0], 0);
      if (v > 0)
        {
          w = mpz_scan1 (Q[0], 0);
          if (w < v)
            v = w;
          if (need_P)
            {
              w = mpz_scan1 (P[0], 0);
              if (w < v)
                v = w;
            }
          /* now v = min(val(T), val(Q), val(P)) */
          if (v > 0)
            {
              mpz_fdiv_q_2exp (T[0], T[0], v);
              mpz_fdiv_q_2exp (Q[0], Q[0], v);
              if (need_P)
                mpz_fdiv_q_2exp (P[0], P[0], v);
            }
        }
    }
}

/* Don't need to save / restore exponent range: the cache does it */
int
mpfr_const_log2_internal (mpfr_ptr x, mpfr_rnd_t rnd_mode)
{
  unsigned long n = MPFR_PREC (x);
  mpfr_prec_t w; /* working precision */
  unsigned long N;
  mpz_t *T, *P, *Q;
  mpfr_t t, q;
  int inexact;
  int ok = 1; /* ensures that the 1st try will give correct rounding */
  unsigned long lgN, i;
  MPFR_ZIV_DECL (loop);

  MPFR_LOG_FUNC (
    ("rnd_mode=%d", rnd_mode),
    ("x[%Pu]=%.*Rg inex=%d", mpfr_get_prec(x), mpfr_log_prec, x, inexact));

  mpfr_init2 (t, MPFR_PREC_MIN);
  mpfr_init2 (q, MPFR_PREC_MIN);

  if (n < 1253)
    w = n + 10; /* ensures correct rounding for the four rounding modes,
                   together with N = w / 3 + 1 (see below). */
  else if (n < 2571)
    w = n + 11; /* idem */
  else if (n < 3983)
    w = n + 12;
  else if (n < 4854)
    w = n + 13;
  else if (n < 26248)
    w = n + 14;
  else
    {
      w = n + 15;
      ok = 0;
    }

  MPFR_ZIV_INIT (loop, w);
  for (;;)
    {
      N = w / 3 + 1; /* Warning: do not change that (even increasing N!)
                        without checking correct rounding in the above
                        ranges for n. */

      /* the following are needed for error analysis (see algorithms.tex) */
      MPFR_ASSERTD(w >= 3 && N >= 2);

      lgN = MPFR_INT_CEIL_LOG2 (N) + 1;
      T  = (mpz_t *) (*__gmp_allocate_func) (3 * lgN * sizeof (mpz_t));
      P  = T + lgN;
      Q  = T + 2*lgN;
      for (i = 0; i < lgN; i++)
        {
          mpz_init (T[i]);
          mpz_init (P[i]);
          mpz_init (Q[i]);
        }

      S (T, P, Q, 0, N, 0);

      mpfr_set_prec (t, w);
      mpfr_set_prec (q, w);

      mpfr_set_z (t, T[0], MPFR_RNDN);
      mpfr_set_z (q, Q[0], MPFR_RNDN);
      mpfr_div (t, t, q, MPFR_RNDN);

      for (i = 0; i < lgN; i++)
        {
          mpz_clear (T[i]);
          mpz_clear (P[i]);
          mpz_clear (Q[i]);
        }
      (*__gmp_free_func) (T, 3 * lgN * sizeof (mpz_t));

      if (MPFR_LIKELY (ok != 0
                       || mpfr_can_round (t, w - 2, MPFR_RNDN, rnd_mode, n)))
        break;

      MPFR_ZIV_NEXT (loop, w);
    }
  MPFR_ZIV_FREE (loop);

  inexact = mpfr_set (x, t, rnd_mode);

  mpfr_clear (t);
  mpfr_clear (q);

  return inexact;
}
