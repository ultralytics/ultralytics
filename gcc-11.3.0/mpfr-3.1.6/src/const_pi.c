/* mpfr_const_pi -- compute Pi

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

#include "mpfr-impl.h"

/* Declare the cache */
#ifndef MPFR_USE_LOGGING
MPFR_DECL_INIT_CACHE (__gmpfr_cache_const_pi, mpfr_const_pi_internal)
#else
MPFR_DECL_INIT_CACHE (__gmpfr_normal_pi, mpfr_const_pi_internal)
MPFR_DECL_INIT_CACHE (__gmpfr_logging_pi, mpfr_const_pi_internal)
MPFR_THREAD_VAR (mpfr_cache_ptr, __gmpfr_cache_const_pi, __gmpfr_normal_pi)
#endif

/* Set User Interface */
#undef mpfr_const_pi
int
mpfr_const_pi (mpfr_ptr x, mpfr_rnd_t rnd_mode) {
  return mpfr_cache (x, __gmpfr_cache_const_pi, rnd_mode);
}

/* Don't need to save/restore exponent range: the cache does it */
int
mpfr_const_pi_internal (mpfr_ptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_t a, A, B, D, S;
  mpfr_prec_t px, p, cancel, k, kmax;
  MPFR_ZIV_DECL (loop);
  int inex;

  MPFR_LOG_FUNC
    (("rnd_mode=%d", rnd_mode),
     ("x[%Pu]=%.*Rg inexact=%d", mpfr_get_prec(x), mpfr_log_prec, x, inex));

  px = MPFR_PREC (x);

  /* we need 9*2^kmax - 4 >= px+2*kmax+8 */
  for (kmax = 2; ((px + 2 * kmax + 12) / 9) >> kmax; kmax ++);

  p = px + 3 * kmax + 14; /* guarantees no recomputation for px <= 10000 */

  mpfr_init2 (a, p);
  mpfr_init2 (A, p);
  mpfr_init2 (B, p);
  mpfr_init2 (D, p);
  mpfr_init2 (S, p);

  MPFR_ZIV_INIT (loop, p);
  for (;;) {
    mpfr_set_ui (a, 1, MPFR_RNDN);          /* a = 1 */
    mpfr_set_ui (A, 1, MPFR_RNDN);          /* A = a^2 = 1 */
    mpfr_set_ui_2exp (B, 1, -1, MPFR_RNDN); /* B = b^2 = 1/2 */
    mpfr_set_ui_2exp (D, 1, -2, MPFR_RNDN); /* D = 1/4 */

#define b B
#define ap a
#define Ap A
#define Bp B
    for (k = 0; ; k++)
      {
        /* invariant: 1/2 <= B <= A <= a < 1 */
        mpfr_add (S, A, B, MPFR_RNDN); /* 1 <= S <= 2 */
        mpfr_div_2ui (S, S, 2, MPFR_RNDN); /* exact, 1/4 <= S <= 1/2 */
        mpfr_sqrt (b, B, MPFR_RNDN); /* 1/2 <= b <= 1 */
        mpfr_add (ap, a, b, MPFR_RNDN); /* 1 <= ap <= 2 */
        mpfr_div_2ui (ap, ap, 1, MPFR_RNDN); /* exact, 1/2 <= ap <= 1 */
        mpfr_mul (Ap, ap, ap, MPFR_RNDN); /* 1/4 <= Ap <= 1 */
        mpfr_sub (Bp, Ap, S, MPFR_RNDN); /* -1/4 <= Bp <= 3/4 */
        mpfr_mul_2ui (Bp, Bp, 1, MPFR_RNDN); /* -1/2 <= Bp <= 3/2 */
        mpfr_sub (S, Ap, Bp, MPFR_RNDN);
        MPFR_ASSERTN (mpfr_cmp_ui (S, 1) < 0);
        cancel = mpfr_cmp_ui (S, 0) ? (mpfr_uexp_t) -mpfr_get_exp(S) : p;
        /* MPFR_ASSERTN (cancel >= px || cancel >= 9 * (1 << k) - 4); */
        mpfr_mul_2ui (S, S, k, MPFR_RNDN);
        mpfr_sub (D, D, S, MPFR_RNDN);
        /* stop when |A_k - B_k| <= 2^(k-p) i.e. cancel >= p-k */
        if (cancel + k >= p)
          break;
      }
#undef b
#undef ap
#undef Ap
#undef Bp

      mpfr_div (A, B, D, MPFR_RNDN);

      /* MPFR_ASSERTN(p >= 2 * k + 8); */
      if (MPFR_LIKELY (MPFR_CAN_ROUND (A, p - 2 * k - 8, px, rnd_mode)))
        break;

      p += kmax;
      MPFR_ZIV_NEXT (loop, p);
      mpfr_set_prec (a, p);
      mpfr_set_prec (A, p);
      mpfr_set_prec (B, p);
      mpfr_set_prec (D, p);
      mpfr_set_prec (S, p);
  }
  MPFR_ZIV_FREE (loop);
  inex = mpfr_set (x, A, rnd_mode);

  mpfr_clear (a);
  mpfr_clear (A);
  mpfr_clear (B);
  mpfr_clear (D);
  mpfr_clear (S);

  return inex;
}
