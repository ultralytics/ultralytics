/* mpz_fac_ui(RESULT, N) -- Set RESULT to N!.

Contributed to the GNU project by Marco Bodrato.

Copyright 1991, 1993-1995, 2000-2003, 2011, 2012 Free Software Foundation, Inc.

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

#if TUNE_PROGRAM_BUILD
#define FACTORS_PER_LIMB (GMP_NUMB_BITS / (LOG2C(FAC_DSC_THRESHOLD_LIMIT-1)+1))
#else
#define FACTORS_PER_LIMB (GMP_NUMB_BITS / (LOG2C(FAC_ODD_THRESHOLD)+1))
#endif

/* Computes n!, the factorial of n.
   WARNING: it assumes that n fits in a limb!
 */
void
mpz_fac_ui (mpz_ptr x, unsigned long n)
{
  static const mp_limb_t table[] = { ONE_LIMB_FACTORIAL_TABLE };

  ASSERT (n <= GMP_NUMB_MAX);

  if (n < numberof (table))
    {
      PTR (x)[0] = table[n];
      SIZ (x) = 1;
    }
  else if (BELOW_THRESHOLD (n, FAC_ODD_THRESHOLD))
    {
      mp_limb_t prod, max_prod;
      mp_size_t j;
      mp_ptr    factors;
      TMP_SDECL;

      TMP_SMARK;
      factors = TMP_SALLOC_LIMBS (2 + (n - numberof (table)) / FACTORS_PER_LIMB);

      factors[0] = table[numberof (table)-1];
      j = 1;
      prod = n;
#if TUNE_PROGRAM_BUILD
      max_prod = GMP_NUMB_MAX / FAC_DSC_THRESHOLD_LIMIT;
#else
      max_prod = GMP_NUMB_MAX / (FAC_ODD_THRESHOLD | 1);
#endif
      while (--n >= numberof (table))
	FACTOR_LIST_STORE (n, prod, max_prod, factors, j);

      factors[j++] = prod;
      mpz_prodlimbs (x, factors, j);

      TMP_SFREE;
    }
  else
    {
      mp_limb_t count;
      mpz_oddfac_1 (x, n, 0);
      if (n <= TABLE_LIMIT_2N_MINUS_POPC_2N)
	count = __gmp_fac2cnt_table[n / 2 - 1];
      else
	{
	  popc_limb (count, n);
	  count = n - count;
	}
      mpz_mul_2exp (x, x, count);
    }
}

#undef FACTORS_PER_LIMB
#undef FACTOR_LIST_STORE
