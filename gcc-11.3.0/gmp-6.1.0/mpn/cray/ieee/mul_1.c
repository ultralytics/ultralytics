/* Cray PVP/IEEE mpn_mul_1 -- multiply a limb vector with a limb and store the
   result in a second limb vector.

Copyright 2000, 2001 Free Software Foundation, Inc.

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

/* This code runs at 5 cycles/limb on a T90.  That would probably
   be hard to improve upon, even with assembly code.  */

#include <intrinsics.h>
#include "gmp.h"
#include "gmp-impl.h"

mp_limb_t
mpn_mul_1 (mp_ptr rp, mp_srcptr up, mp_size_t n, mp_limb_t vl)
{
  mp_limb_t cy[n];
  mp_limb_t a, b, r, s0, s1, c0, c1;
  mp_size_t i;
  int more_carries;

  if (up == rp)
    {
      /* The algorithm used below cannot handle overlap.  Handle it here by
	 making a temporary copy of the source vector, then call ourselves.  */
      mp_limb_t xp[n];
      MPN_COPY (xp, up, n);
      return mpn_mul_1 (rp, xp, n, vl);
    }

  a = up[0] * vl;
  rp[0] = a;
  cy[0] = 0;

  /* Main multiply loop.  Generate a raw accumulated output product in rp[]
     and a carry vector in cy[].  */
#pragma _CRI ivdep
  for (i = 1; i < n; i++)
    {
      a = up[i] * vl;
      b = _int_mult_upper (up[i - 1], vl);
      s0 = a + b;
      c0 = ((a & b) | ((a | b) & ~s0)) >> 63;
      rp[i] = s0;
      cy[i] = c0;
    }
  /* Carry add loop.  Add the carry vector cy[] to the raw sum rp[] and
     store the new sum back to rp[0].  */
  more_carries = 0;
#pragma _CRI ivdep
  for (i = 2; i < n; i++)
    {
      r = rp[i];
      c0 = cy[i - 1];
      s0 = r + c0;
      rp[i] = s0;
      c0 = (r & ~s0) >> 63;
      more_carries += c0;
    }
  /* If that second loop generated carry, handle that in scalar loop.  */
  if (more_carries)
    {
      mp_limb_t cyrec = 0;
      /* Look for places where rp[k] is zero and cy[k-1] is non-zero.
	 These are where we got a recurrency carry.  */
      for (i = 2; i < n; i++)
	{
	  r = rp[i];
	  c0 = (r == 0 && cy[i - 1] != 0);
	  s0 = r + cyrec;
	  rp[i] = s0;
	  c1 = (r & ~s0) >> 63;
	  cyrec = c0 | c1;
	}
      return _int_mult_upper (up[n - 1], vl) + cyrec + cy[n - 1];
    }

  return _int_mult_upper (up[n - 1], vl) + cy[n - 1];
}
