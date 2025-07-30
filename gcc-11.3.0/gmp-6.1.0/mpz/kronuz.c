/* mpz_ui_kronecker -- ulong+mpz Kronecker/Jacobi symbol.

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
mpz_ui_kronecker (unsigned long a, mpz_srcptr b)
{
  mp_srcptr  b_ptr;
  mp_limb_t  b_low;
  int        b_abs_size;
  mp_limb_t  b_rem;
  int        twos;
  int        result_bit1;

  /* (a/-1)=1 when a>=0, so the sign of b is ignored */
  b_abs_size = ABSIZ (b);

  if (b_abs_size == 0)
    return JACOBI_U0 (a);  /* (a/0) */

  if (a > GMP_NUMB_MAX)
    {
      mp_limb_t  alimbs[2];
      mpz_t      az;
      ALLOC(az) = numberof (alimbs);
      PTR(az) = alimbs;
      mpz_set_ui (az, a);
      return mpz_kronecker (az, b);
    }

  b_ptr = PTR(b);
  b_low = b_ptr[0];
  result_bit1 = 0;

  if (! (b_low & 1))
    {
      /* (0/b)=0 for b!=+/-1; and (even/even)=0 */
      if (! (a & 1))
	return 0;

      /* a odd, b even

	 Establish shifted b_low with valid bit1 for the RECIP below.  Zero
	 limbs stripped are accounted for, but zero bits on b_low are not
	 because they remain in {b_ptr,b_abs_size} for
	 JACOBI_MOD_OR_MODEXACT_1_ODD. */

      JACOBI_STRIP_LOW_ZEROS (result_bit1, a, b_ptr, b_abs_size, b_low);
      if (! (b_low & 1))
	{
	  if (UNLIKELY (b_low == GMP_NUMB_HIGHBIT))
	    {
	      /* need b_ptr[1] to get bit1 in b_low */
	      if (b_abs_size == 1)
		{
		  /* (a/0x80...00) == (a/2)^(NUMB-1) */
		  if ((GMP_NUMB_BITS % 2) == 0)
		    {
		      /* JACOBI_STRIP_LOW_ZEROS does nothing to result_bit1
			 when GMP_NUMB_BITS is even, so it's still 0. */
		      ASSERT (result_bit1 == 0);
		      result_bit1 = JACOBI_TWO_U_BIT1 (a);
		    }
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
    }
  else
    {
      if (a == 0)        /* (0/b)=1 for b=+/-1, 0 otherwise */
	return (b_abs_size == 1 && b_low == 1);

      if (! (a & 1))
	{
	  /* a even, b odd */
	  count_trailing_zeros (twos, a);
	  a >>= twos;
	  /* (a*2^n/b) = (a/b) * (2/a)^n */
	  result_bit1 = JACOBI_TWOS_U_BIT1 (twos, b_low);
	}
    }

  if (a == 1)
    return JACOBI_BIT1_TO_PN (result_bit1);  /* (1/b)=1 */

  /* (a/b*2^n) = (b*2^n mod a / a) * RECIP(a,b) */
  JACOBI_MOD_OR_MODEXACT_1_ODD (result_bit1, b_rem, b_ptr, b_abs_size, a);
  result_bit1 ^= JACOBI_RECIP_UU_BIT1 (a, b_low);
  return mpn_jacobi_base (b_rem, (mp_limb_t) a, result_bit1);
}
