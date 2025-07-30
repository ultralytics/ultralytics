/* mpn_divrem_2 -- Divide natural numbers, producing both remainder and
   quotient.  The divisor is two limbs.

   THIS FILE CONTAINS INTERNAL FUNCTIONS WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.


Copyright 1993-1996, 1999-2002 Free Software Foundation, Inc.

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


/* Divide num {np,nn} by den {dp,2} and write the nn-2 least significant
   quotient limbs at qp and the 2 long remainder at np.  If qxn is non-zero,
   generate that many fraction bits and append them after the other quotient
   limbs.  Return the most significant limb of the quotient, this is always 0
   or 1.

   Preconditions:
   1. The most significant bit of the divisor must be set.
   2. qp must either not overlap with the input operands at all, or
      qp >= np + 2 must hold true.  (This means that it's possible to put
      the quotient in the high part of {np,nn}, right above the remainder.
   3. nn >= 2, even if qxn is non-zero.  */

mp_limb_t
mpn_divrem_2 (mp_ptr qp, mp_size_t qxn,
	      mp_ptr np, mp_size_t nn,
	      mp_srcptr dp)
{
  mp_limb_t most_significant_q_limb;
  mp_size_t i;
  mp_limb_t r1, r0, d1, d0;
  gmp_pi1_t di;

  ASSERT (nn >= 2);
  ASSERT (qxn >= 0);
  ASSERT (dp[1] & GMP_NUMB_HIGHBIT);
  ASSERT (! MPN_OVERLAP_P (qp, nn-2+qxn, np, nn) || qp >= np+2);
  ASSERT_MPN (np, nn);
  ASSERT_MPN (dp, 2);

  np += nn - 2;
  d1 = dp[1];
  d0 = dp[0];
  r1 = np[1];
  r0 = np[0];

  most_significant_q_limb = 0;
  if (r1 >= d1 && (r1 > d1 || r0 >= d0))
    {
#if GMP_NAIL_BITS == 0
      sub_ddmmss (r1, r0, r1, r0, d1, d0);
#else
      r0 = r0 - d0;
      r1 = r1 - d1 - (r0 >> GMP_LIMB_BITS - 1);
      r0 &= GMP_NUMB_MASK;
#endif
      most_significant_q_limb = 1;
    }

  invert_pi1 (di, d1, d0);

  qp += qxn;

  for (i = nn - 2 - 1; i >= 0; i--)
    {
      mp_limb_t n0, q;
      n0 = np[-1];
      udiv_qr_3by2 (q, r1, r0, r1, r0, n0, d1, d0, di.inv32);
      np--;
      qp[i] = q;
    }

  if (UNLIKELY (qxn != 0))
    {
      qp -= qxn;
      for (i = qxn - 1; i >= 0; i--)
	{
	  mp_limb_t q;
	  udiv_qr_3by2 (q, r1, r0, r1, r0, CNST_LIMB(0), d1, d0, di.inv32);
	  qp[i] = q;
	}
    }

  np[1] = r1;
  np[0] = r0;

  return most_significant_q_limb;
}
