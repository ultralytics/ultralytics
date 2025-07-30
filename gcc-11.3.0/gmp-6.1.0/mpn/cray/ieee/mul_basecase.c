/* Cray PVP/IEEE mpn_mul_basecase.

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

/* The most critical loop of this code runs at about 5 cycles/limb on a T90.
   That is not perfect, mainly due to vector register shortage.  */

#include <intrinsics.h>
#include "gmp.h"
#include "gmp-impl.h"

void
mpn_mul_basecase (mp_ptr rp,
		  mp_srcptr up, mp_size_t un,
		  mp_srcptr vp, mp_size_t vn)
{
  mp_limb_t cy[un + vn];
  mp_limb_t vl;
  mp_limb_t a, b, r, s0, s1, c0, c1;
  mp_size_t i, j;
  int more_carries;

  for (i = 0; i < un + vn; i++)
    {
      rp[i] = 0;
      cy[i] = 0;
    }

#pragma _CRI novector
  for (j = 0; j < vn; j++)
    {
      vl = vp[j];

      a = up[0] * vl;
      r = rp[j];
      s0 = a + r;
      rp[j] = s0;
      c0 = ((a & r) | ((a | r) & ~s0)) >> 63;
      cy[j] += c0;

#pragma _CRI ivdep
      for (i = 1; i < un; i++)
	{
	  a = up[i] * vl;
	  b = _int_mult_upper (up[i - 1], vl);
	  s0 = a + b;
	  c0 = ((a & b) | ((a | b) & ~s0)) >> 63;
	  r = rp[j + i];
	  s1 = s0 + r;
	  rp[j + i] = s1;
	  c1 = ((s0 & r) | ((s0 | r) & ~s1)) >> 63;
	  cy[j + i] += c0 + c1;
	}
      rp[j + un] = _int_mult_upper (up[un - 1], vl);
    }

  more_carries = 0;
#pragma _CRI ivdep
  for (i = 1; i < un + vn; i++)
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
      for (i = 1; i < un + vn; i++)
	{
	  r = rp[i];
	  c0 = (r < cy[i - 1]);
	  s0 = r + cyrec;
	  rp[i] = s0;
	  c1 = (r & ~s0) >> 63;
	  cyrec = c0 | c1;
	}
    }
}
