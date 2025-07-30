/* mpn_cnd_add_n -- Compute R = U + V if CND != 0 or R = U if CND == 0.
   Both cases should take the same time and perform the exact same memory
   accesses, since this function is intended to be used where side-channel
   attack resilience is relevant.

Copyright 1992-1994, 1996, 2000, 2002, 2008, 2009, 2011, 2013 Free Software
Foundation, Inc.

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

mp_limb_t
mpn_cnd_add_n (mp_limb_t cnd, mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
{
  mp_limb_t ul, vl, sl, rl, cy, cy1, cy2, mask;

  ASSERT (n >= 1);
  ASSERT (MPN_SAME_OR_SEPARATE_P (rp, up, n));
  ASSERT (MPN_SAME_OR_SEPARATE_P (rp, vp, n));

  mask = -(mp_limb_t) (cnd != 0);
  cy = 0;
  do
    {
      ul = *up++;
      vl = *vp++ & mask;
#if GMP_NAIL_BITS == 0
      sl = ul + vl;
      cy1 = sl < ul;
      rl = sl + cy;
      cy2 = rl < sl;
      cy = cy1 | cy2;
      *rp++ = rl;
#else
      rl = ul + vl;
      rl += cy;
      cy = rl >> GMP_NUMB_BITS;
      *rp++ = rl & GMP_NUMB_MASK;
#endif
    }
  while (--n != 0);

  return cy;
}
