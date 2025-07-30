/* mpz_mfac_uiui(RESULT, N, M) -- Set RESULT to N!^(M) = N(N-M)(N-2M)...

Contributed to the GNU project by Marco Bodrato.

Copyright 2012, 2013 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

#include "gmp.h"
#include "gmp-impl.h"

/*************************************************************/
/* Section macros: common macros, for swing/fac/bin (&sieve) */
/*************************************************************/

#define FACTOR_LIST_STORE(P, PR, MAX_PR, VEC, I)		\
  do {								\
    if ((PR) > (MAX_PR)) {					\
      (VEC)[(I)++] = (PR);					\
      (PR) = (P);						\
    } else							\
      (PR) *= (P);						\
  } while (0)

/*********************************************************/
/* Section oder factorials:                              */
/*********************************************************/

/* mpz_mfac_uiui (x, n, m) computes x = n!^(m) = n*(n-m)*(n-2m)*...   */

void
mpz_mfac_uiui (mpz_ptr x, unsigned long n, unsigned long m)
{
  ASSERT (n <= GMP_NUMB_MAX);
  ASSERT (m != 0);

  if ((n < 3) | (n - 3 < m - 1)) { /* (n < 3 || n - 1 <= m || m == 0) */
    PTR (x)[0] = n + (n == 0);
    SIZ (x) = 1;
  } else { /* m < n - 1 < GMP_NUMB_MAX */
    mp_limb_t g, sn;
    mpz_t     t;

    sn = n;
    g = mpn_gcd_1 (&sn, 1, m);
    if (g > 1) { n/=g; m/=g; }

    if (m <= 2) { /* fac or 2fac */
      if (m == 1) {
	if (g > 2) {
	  mpz_init (t);
	  mpz_fac_ui (t, n);
	  sn = n;
	} else {
	  if (g == 2)
	    mpz_2fac_ui (x, n << 1);
	  else
	    mpz_fac_ui (x, n);
	  return;
	}
      } else { /* m == 2 */
	if (g > 1) {
	  mpz_init (t);
	  mpz_2fac_ui (t, n);
	  sn = n / 2 + 1;
	} else {
	  mpz_2fac_ui (x, n);
	  return;
	}
      }
    } else { /* m >= 3, gcd(n,m) = 1 */
      mp_limb_t *factors;
      mp_limb_t prod, max_prod, j;
      TMP_DECL;

      sn = n / m + 1;

      j = 0;
      prod = n;
      n -= m;
      max_prod = GMP_NUMB_MAX / n;

      if (g > 1)
	factors = MPZ_NEWALLOC (x, sn / log_n_max (n) + 2);
      else {
	TMP_MARK;
	factors = TMP_ALLOC_LIMBS (sn / log_n_max (n) + 2);
      }

      for (; n > m; n -= m)
	FACTOR_LIST_STORE (n, prod, max_prod, factors, j);

      factors[j++] = n;
      factors[j++] = prod;

      if (g > 1) {
	mpz_init (t);
	mpz_prodlimbs (t, factors, j);
      } else {
	mpz_prodlimbs (x, factors, j);
	TMP_FREE;
	return;
      }
    }

    {
      mpz_t p;

      mpz_init (p);
      mpz_ui_pow_ui (p, g, sn); /* g^sn */
      mpz_mul (x, p, t);
      mpz_clear (p);
      mpz_clear (t);
    }
  }
}
