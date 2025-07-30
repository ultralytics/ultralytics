/* mpfr_const_catalan -- compute Catalan's constant.

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

/* Declare the cache */
MPFR_DECL_INIT_CACHE (__gmpfr_cache_const_catalan, mpfr_const_catalan_internal)

/* Set User Interface */
#undef mpfr_const_catalan
int
mpfr_const_catalan (mpfr_ptr x, mpfr_rnd_t rnd_mode) {
  return mpfr_cache (x, __gmpfr_cache_const_catalan, rnd_mode);
}

/* return T, Q such that T/Q = sum(k!^2/(2k)!/(2k+1)^2, k=n1..n2-1) */
static void
S (mpz_t T, mpz_t P, mpz_t Q, unsigned long n1, unsigned long n2)
{
  if (n2 == n1 + 1)
    {
      if (n1 == 0)
        {
          mpz_set_ui (P, 1);
          mpz_set_ui (Q, 1);
        }
      else
        {
          mpz_set_ui (P, 2 * n1 - 1);
          mpz_mul_ui (P, P, n1);
          mpz_ui_pow_ui (Q, 2 * n1 + 1, 2);
          mpz_mul_2exp (Q, Q, 1);
        }
      mpz_set (T, P);
    }
  else
    {
      unsigned long m = (n1 + n2) / 2;
      mpz_t T2, P2, Q2;
      S (T, P, Q, n1, m);
      mpz_init (T2);
      mpz_init (P2);
      mpz_init (Q2);
      S (T2, P2, Q2, m, n2);
      mpz_mul (T, T, Q2);
      mpz_mul (T2, T2, P);
      mpz_add (T, T, T2);
      mpz_mul (P, P, P2);
      mpz_mul (Q, Q, Q2);
      mpz_clear (T2);
      mpz_clear (P2);
      mpz_clear (Q2);
    }
}

/* Don't need to save/restore exponent range: the cache does it.
   Catalan's constant is G = sum((-1)^k/(2*k+1)^2, k=0..infinity).
   We compute it using formula (31) of Victor Adamchik's page
   "33 representations for Catalan's constant"
   http://www-2.cs.cmu.edu/~adamchik/articles/catalan/catalan.htm

   G = Pi/8*log(2+sqrt(3)) + 3/8*sum(k!^2/(2k)!/(2k+1)^2,k=0..infinity)
*/
int
mpfr_const_catalan_internal (mpfr_ptr g, mpfr_rnd_t rnd_mode)
{
  mpfr_t x, y, z;
  mpz_t T, P, Q;
  mpfr_prec_t pg, p;
  int inex;
  MPFR_ZIV_DECL (loop);
  MPFR_GROUP_DECL (group);

  MPFR_LOG_FUNC (("rnd_mode=%d", rnd_mode),
    ("g[%Pu]=%.*Rg inex=%d", mpfr_get_prec (g), mpfr_log_prec, g, inex));

  /* Here are the WC (max prec = 100.000.000)
     Once we have found a chain of 11, we only look for bigger chain.
     Found 3 '1' at 0
     Found 5 '1' at 9
     Found 6 '0' at 34
     Found 9 '1' at 176
     Found 11 '1' at 705
     Found 12 '0' at 913
     Found 14 '1' at 12762
     Found 15 '1' at 152561
     Found 16 '0' at 171725
     Found 18 '0' at 525355
     Found 20 '0' at 529245
     Found 21 '1' at 6390133
     Found 22 '0' at 7806417
     Found 25 '1' at 11936239
     Found 27 '1' at 51752950
  */
  pg = MPFR_PREC (g);
  p = pg + MPFR_INT_CEIL_LOG2 (pg) + 7;

  MPFR_GROUP_INIT_3 (group, p, x, y, z);
  mpz_init (T);
  mpz_init (P);
  mpz_init (Q);

  MPFR_ZIV_INIT (loop, p);
  for (;;) {
    mpfr_sqrt_ui (x, 3, MPFR_RNDU);
    mpfr_add_ui (x, x, 2, MPFR_RNDU);
    mpfr_log (x, x, MPFR_RNDU);
    mpfr_const_pi (y, MPFR_RNDU);
    mpfr_mul (x, x, y, MPFR_RNDN);
    S (T, P, Q, 0, (p - 1) / 2);
    mpz_mul_ui (T, T, 3);
    mpfr_set_z (y, T, MPFR_RNDU);
    mpfr_set_z (z, Q, MPFR_RNDD);
    mpfr_div (y, y, z, MPFR_RNDN);
    mpfr_add (x, x, y, MPFR_RNDN);
    mpfr_div_2ui (x, x, 3, MPFR_RNDN);

    if (MPFR_LIKELY (MPFR_CAN_ROUND (x, p - 5, pg, rnd_mode)))
      break;

    MPFR_ZIV_NEXT (loop, p);
    MPFR_GROUP_REPREC_3 (group, p, x, y, z);
  }
  MPFR_ZIV_FREE (loop);
  inex = mpfr_set (g, x, rnd_mode);

  MPFR_GROUP_CLEAR (group);
  mpz_clear (T);
  mpz_clear (P);
  mpz_clear (Q);

  return inex;
}
