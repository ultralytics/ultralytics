/* mpz_si_kronecker -- long+mpz Kronecker/Jacobi symbol.

Copyright 1999-2002 Free Software Foundation, Inc.

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
#include "longlong.h"


int
mpz_si_kronecker (long a, mpz_srcptr b)
{
  mp_srcptr  b_ptr;
  mp_limb_t  b_low;
  mp_size_t  b_size;
  mp_size_t  b_abs_size;
  mp_limb_t  a_limb, b_rem;
  unsigned   twos;
  int        result_bit1;

#if GMP_NUMB_BITS < BITS_PER_ULONG
  if (a > GMP_NUMB_MAX || a < -GMP_NUMB_MAX)
    {
      mp_limb_t  alimbs[2];
      mpz_t      az;
      ALLOC(az) = numberof (alimbs);
      PTR(az) = alimbs;
      mpz_set_si (az, a);
      return mpz_kronecker (az, b);
    }
#endif

  b_size = SIZ (b);
  if (b_size == 0)
    return JACOBI_S0 (a);  /* (a/0) */

  /* account for the effect of the sign of b, then ignore it */
  result_bit1 = JACOBI_BSGN_SS_BIT1 (a, b_size);

  b_ptr = PTR(b);
  b_low = b_ptr[0];
  b_abs_size = ABS (b_size);

  if ((b_low & 1) != 0)
    {
      /* b odd */

      result_bit1 ^= JACOBI_ASGN_SU_BIT1 (a, b_low);
      a_limb = (unsigned long) ABS(a);

      if ((a_limb & 1) == 0)
	{
	  /* (0/b)=1 for b=+/-1, 0 otherwise */
	  if (a_limb == 0)
	    return (b_abs_size == 1 && b_low == 1);

	  /* a even, b odd */
	  count_trailing_zeros (twos, a_limb);
	  a_limb >>= twos;
	  /* (a*2^n/b) = (a/b) * twos(n,a) */
	  result_bit1 ^= JACOBI_TWOS_U_BIT1 (twos, b_low);
	}
    }
  else
    {
      /* (even/even)=0, and (0/b)=0 for b!=+/-1 */
      if ((a & 1) == 0)
	return 0;

      /* a odd, b even

	 Establish shifted b_low with valid bit1 for ASGN and RECIP below.
	 Zero limbs stripped are accounted for, but zero bits on b_low are
	 not because they remain in {b_ptr,b_abs_size} for the
	 JACOBI_MOD_OR_MODEXACT_1_ODD. */

      JACOBI_STRIP_LOW_ZEROS (result_bit1, a, b_ptr, b_abs_size, b_low);
      if ((b_low & 1) == 0)
	{
	  if (UNLIKELY (b_low == GMP_NUMB_HIGHBIT))
	    {
	      /* need b_ptr[1] to get bit1 in b_low */
	      if (b_abs_size == 1)
		{
		  /* (a/0x80000000) = (a/2)^(BPML-1) */
		  if ((GMP_NUMB_BITS % 2) == 0)
		    result_bit1 ^= JACOBI_TWO_U_BIT1 (a);
		  return JACOBI_BIT1_TO_PN (result_bit1);
		}

	      /* b_abs_size > 1 */
	      b_low = b_ptr[1] << 1;
	    }
	  else
	    {
	      count_trailing_zeros (twos, b_low);
	      b_low >>= twos;
	    }
	}

      result_bit1 ^= JACOBI_ASGN_SU_BIT1 (a, b_low);
      a_limb = (unsigned long) ABS(a);
    }

  if (a_limb == 1)
    return JACOBI_BIT1_TO_PN (result_bit1);  /* (1/b)=1 */

  /* (a/b*2^n) = (b*2^n mod a / a) * recip(a,b) */
  JACOBI_MOD_OR_MODEXACT_1_ODD (result_bit1, b_rem, b_ptr, b_abs_size, a_limb);
  result_bit1 ^= JACOBI_RECIP_UU_BIT1 (a_limb, b_low);
  return mpn_jacobi_base (b_rem, a_limb, result_bit1);
}
