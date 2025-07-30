/* mpn_bsqrtinv, compute r such that r^2 * y = 1 (mod 2^{b+1}).

   Contributed to the GNU project by Martin Boij (as part of perfpow.c).

Copyright 2009, 2010, 2012, 2015 Free Software Foundation, Inc.

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

/* Compute r such that r^2 * y = 1 (mod 2^{b+1}).
   Return non-zero if such an integer r exists.

   Iterates
     r' <-- (3r - r^3 y) / 2
   using Hensel lifting.  Since we divide by two, the Hensel lifting is
   somewhat degenerates.  Therefore, we lift from 2^b to 2^{b+1}-1.

   FIXME:
     (1) Simplify to do precision book-keeping in limbs rather than bits.

     (2) Rewrite iteration as
	   r' <-- r - r (r^2 y - 1) / 2
	 and take advantage of zero low part of r^2 y - 1.

     (3) Use wrap-around trick.

     (4) Use a small table to get starting value.
*/
int
mpn_bsqrtinv (mp_ptr rp, mp_srcptr yp, mp_bitcnt_t bnb, mp_ptr tp)
{
  mp_ptr tp2;
  mp_size_t bn, order[GMP_LIMB_BITS + 1];
  int i, d;

  ASSERT (bnb > 0);

  bn = 1 + bnb / GMP_LIMB_BITS;

  tp2 = tp + bn;

  rp[0] = 1;
  if (bnb == 1)
    {
      if ((yp[0] & 3) != 1)
	return 0;
    }
  else
    {
      if ((yp[0] & 7) != 1)
	return 0;

      d = 0;
      for (; bnb != 2; bnb = (bnb + 2) >> 1)
	order[d++] = bnb;

      for (i = d - 1; i >= 0; i--)
	{
	  bnb = order[i];
	  bn = 1 + bnb / GMP_LIMB_BITS;

	  mpn_sqrlo (tp, rp, bn);
	  mpn_mullo_n (tp2, rp, tp, bn); /* tp2 <- rp ^ 3 */

	  mpn_mul_1 (tp, rp, bn, 3);

	  mpn_mullo_n (rp, yp, tp2, bn);

#if HAVE_NATIVE_mpn_rsh1sub_n
	  mpn_rsh1sub_n (rp, tp, rp, bn);
#else
	  mpn_sub_n (tp2, tp, rp, bn);
	  mpn_rshift (rp, tp2, bn, 1);
#endif
	}
    }
  return 1;
}
