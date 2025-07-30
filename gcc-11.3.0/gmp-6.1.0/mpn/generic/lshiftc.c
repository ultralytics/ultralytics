/* mpn_lshiftc -- Shift left low level with complement.

Copyright 1991, 1993, 1994, 1996, 2000-2002, 2009 Free Software Foundation,
Inc.

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

/* Shift U (pointed to by up and n limbs long) cnt bits to the left
   and store the n least significant limbs of the result at rp.
   Return the bits shifted out from the most significant limb.

   Argument constraints:
   1. 0 < cnt < GMP_NUMB_BITS.
   2. If the result is to be written over the input, rp must be >= up.
*/

mp_limb_t
mpn_lshiftc (mp_ptr rp, mp_srcptr up, mp_size_t n, unsigned int cnt)
{
  mp_limb_t high_limb, low_limb;
  unsigned int tnc;
  mp_size_t i;
  mp_limb_t retval;

  ASSERT (n >= 1);
  ASSERT (cnt >= 1);
  ASSERT (cnt < GMP_NUMB_BITS);
  ASSERT (MPN_SAME_OR_DECR_P (rp, up, n));

  up += n;
  rp += n;

  tnc = GMP_NUMB_BITS - cnt;
  low_limb = *--up;
  retval = low_limb >> tnc;
  high_limb = (low_limb << cnt);

  for (i = n - 1; i != 0; i--)
    {
      low_limb = *--up;
      *--rp = (~(high_limb | (low_limb >> tnc))) & GMP_NUMB_MASK;
      high_limb = low_limb << cnt;
    }
  *--rp = (~high_limb) & GMP_NUMB_MASK;

  return retval;
}
