/* mpz_2fac_ui(RESULT, N) -- Set RESULT to N!!.

Contributed to the GNU project by Marco Bodrato.

Copyright 2012 Free Software Foundation, Inc.

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

#define FACTOR_LIST_STORE(P, PR, MAX_PR, VEC, I)		\
  do {								\
    if ((PR) > (MAX_PR)) {					\
      (VEC)[(I)++] = (PR);					\
      (PR) = (P);						\
    } else							\
      (PR) *= (P);						\
  } while (0)

#define FAC_2DSC_THRESHOLD ((FAC_DSC_THRESHOLD << 1) | (FAC_DSC_THRESHOLD & 1))
#define FACTORS_PER_LIMB   (GMP_NUMB_BITS / (LOG2C(FAC_2DSC_THRESHOLD-1)+1))

/* Computes n!!, the 2-multi-factorial of n. (aka double-factorial or semi-factorial)
   WARNING: it assumes that n fits in a limb!
 */
void
mpz_2fac_ui (mpz_ptr x, unsigned long n)
{
  ASSERT (n <= GMP_NUMB_MAX);

  if ((n & 1) == 0) { /* n is even, n = 2k, (2k)!! = k! 2^k */
    mp_limb_t count;

    if ((n <= TABLE_LIMIT_2N_MINUS_POPC_2N) & (n != 0))
      count = __gmp_fac2cnt_table[n / 2 - 1];
    else
      {
	popc_limb (count, n);	/* popc(n) == popc(k) */
	count = n - count;		/* n - popc(n) == k + k - popc(k) */
      }
    mpz_oddfac_1 (x, n >> 1, 0);
    mpz_mul_2exp (x, x, count);
  } else { /* n is odd */
    if (n <= ODD_DOUBLEFACTORIAL_TABLE_LIMIT) {
	PTR (x)[0] = __gmp_odd2fac_table[n >> 1];
	SIZ (x) = 1;
    } else if (BELOW_THRESHOLD (n, FAC_2DSC_THRESHOLD)) { /* odd basecase, */
      mp_limb_t *factors, prod, max_prod, j;
      TMP_SDECL;

      /* FIXME: we might alloc a fixed amount 1+FAC_2DSC_THRESHOLD/FACTORS_PER_LIMB */
      TMP_SMARK;
      factors = TMP_SALLOC_LIMBS (1 + n / (2 * FACTORS_PER_LIMB));

      factors[0] = ODD_DOUBLEFACTORIAL_TABLE_MAX;
      j = 1;
      prod = n;

      max_prod = GMP_NUMB_MAX / FAC_2DSC_THRESHOLD;
      while ((n -= 2) > ODD_DOUBLEFACTORIAL_TABLE_LIMIT)
	FACTOR_LIST_STORE (n, prod, max_prod, factors, j);

      factors[j++] = prod;
      mpz_prodlimbs (x, factors, j);

      TMP_SFREE;
    } else { /* for the asymptotically fast odd case, let oddfac do the job. */
      mpz_oddfac_1 (x, n, 1);
    }
  }
}

#undef FACTORS_PER_LIMB
#undef FACTOR_LIST_STORE
#undef FAC_2DSC_THRESHOLD
