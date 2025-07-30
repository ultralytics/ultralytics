/* mpn_div_qr_2u_pi1

   Contributed to the GNU project by Niels Möller

   THIS FILE CONTAINS INTERNAL FUNCTIONS WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.


Copyright 2011 Free Software Foundation, Inc.

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


/* 3/2 loop, for unnormalized divisor. Caller must pass shifted d1 and
   d0, while {np,nn} is shifted on the fly. */
mp_limb_t
mpn_div_qr_2u_pi1 (mp_ptr qp, mp_ptr rp, mp_srcptr np, mp_size_t nn,
		   mp_limb_t d1, mp_limb_t d0, int shift, mp_limb_t di)
{
  mp_limb_t qh;
  mp_limb_t r2, r1, r0;
  mp_size_t i;

  ASSERT (nn >= 2);
  ASSERT (d1 & GMP_NUMB_HIGHBIT);
  ASSERT (shift > 0);

  r2 = np[nn-1] >> (GMP_LIMB_BITS - shift);
  r1 = (np[nn-1] << shift) | (np[nn-2] >> (GMP_LIMB_BITS - shift));
  r0 = np[nn-2] << shift;

  udiv_qr_3by2 (qh, r2, r1, r2, r1, r0, d1, d0, di);

  for (i = nn - 2 - 1; i >= 0; i--)
    {
      mp_limb_t q;
      r0 = np[i];
      r1 |= r0 >> (GMP_LIMB_BITS - shift);
      r0 <<= shift;
      udiv_qr_3by2 (q, r2, r1, r2, r1, r0, d1, d0, di);
      qp[i] = q;
    }

  rp[0] = (r1 >> shift) | (r2 << (GMP_LIMB_BITS - shift));
  rp[1] = r2 >> shift;

  return qh;
}
