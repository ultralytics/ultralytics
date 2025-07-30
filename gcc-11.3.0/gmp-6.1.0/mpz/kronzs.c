/* mpz_kronecker_si -- mpz+long Kronecker/Jacobi symbol.

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


/* After the absolute value of b is established it's treated as an unsigned
   long, because 0x80..00 doesn't fit in a signed long. */

int
mpz_kronecker_si (mpz_srcptr a, long b)
{
  mp_srcptr  a_ptr;
  mp_size_t  a_size;
  mp_limb_t  a_rem, b_limb;
  int        result_bit1;

  a_size = SIZ(a);
  if (a_size == 0)
    return JACOBI_0S (b);

#if GMP_NUMB_BITS < BITS_PER_ULONG
  if (b > GMP_NUMB_MAX || b < -GMP_NUMB_MAX)
    {
      mp_limb_t  blimbs[2];
      mpz_t      bz;
      ALLOC(bz) = numberof (blimbs);
      PTR(bz) = blimbs;
      mpz_set_si (bz, b);
      return mpz_kronecker (a, bz);
    }
#endif

  result_bit1 = JACOBI_BSGN_SS_BIT1 (a_size, b);
  b_limb = ABS_CAST (unsigned long, b);
  a_ptr = PTR(a);

  if ((b_limb & 1) == 0)
    {
      mp_limb_t  a_low = a_ptr[0];
      int        twos;

      if (b_limb == 0)
	return JACOBI_LS0 (a_low, a_size);   /* (a/0) */

      if (! (a_low & 1))
	return 0;  /* (even/even)=0 */

      /* (a/2)=(2/a) for a odd */
      count_trailing_zeros (twos, b_limb);
      b_limb >>= twos;
      result_bit1 ^= JACOBI_TWOS_U_BIT1 (twos, a_low);
    }

  if (b_limb == 1)
    return JACOBI_BIT1_TO_PN (result_bit1);  /* (a/1)=1 for any a */

  result_bit1 ^= JACOBI_ASGN_SU_BIT1 (a_size, b_limb);
  a_size = ABS(a_size);

  /* (a/b) = (a mod b / b) */
  JACOBI_MOD_OR_MODEXACT_1_ODD (result_bit1, a_rem, a_ptr, a_size, b_limb);
  return mpn_jacobi_base (a_rem, b_limb, result_bit1);
}
