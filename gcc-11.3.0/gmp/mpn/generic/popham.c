/* mpn_popcount, mpn_hamdist -- mpn bit population count/hamming distance.

Copyright 1994, 1996, 2000-2002, 2005, 2011, 2012 Free Software Foundation,
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

#if OPERATION_popcount
#define FNAME mpn_popcount
#define POPHAM(u,v) u
#endif

#if OPERATION_hamdist
#define FNAME mpn_hamdist
#define POPHAM(u,v) u ^ v
#endif

mp_bitcnt_t
FNAME (mp_srcptr up,
#if OPERATION_hamdist
       mp_srcptr vp,
#endif
       mp_size_t n) __GMP_NOTHROW
{
  mp_bitcnt_t result = 0;
  mp_limb_t p0, p1, p2, p3, x, p01, p23;
  mp_size_t i;

  ASSERT (n >= 1);		/* Actually, this code handles any n, but some
				   assembly implementations do not.  */

  for (i = n >> 2; i != 0; i--)
    {
      p0 = POPHAM (up[0], vp[0]);
      p0 -= (p0 >> 1) & MP_LIMB_T_MAX/3;				/* 2 0-2 */
      p0 = ((p0 >> 2) & MP_LIMB_T_MAX/5) + (p0 & MP_LIMB_T_MAX/5);	/* 4 0-4 */

      p1 = POPHAM (up[1], vp[1]);
      p1 -= (p1 >> 1) & MP_LIMB_T_MAX/3;				/* 2 0-2 */
      p1 = ((p1 >> 2) & MP_LIMB_T_MAX/5) + (p1 & MP_LIMB_T_MAX/5);	/* 4 0-4 */

      p01 = p0 + p1;							/* 8 0-8 */
      p01 = ((p01 >> 4) & MP_LIMB_T_MAX/17) + (p01 & MP_LIMB_T_MAX/17);	/* 8 0-16 */

      p2 = POPHAM (up[2], vp[2]);
      p2 -= (p2 >> 1) & MP_LIMB_T_MAX/3;				/* 2 0-2 */
      p2 = ((p2 >> 2) & MP_LIMB_T_MAX/5) + (p2 & MP_LIMB_T_MAX/5);	/* 4 0-4 */

      p3 = POPHAM (up[3], vp[3]);
      p3 -= (p3 >> 1) & MP_LIMB_T_MAX/3;				/* 2 0-2 */
      p3 = ((p3 >> 2) & MP_LIMB_T_MAX/5) + (p3 & MP_LIMB_T_MAX/5);	/* 4 0-4 */

      p23 = p2 + p3;							/* 8 0-8 */
      p23 = ((p23 >> 4) & MP_LIMB_T_MAX/17) + (p23 & MP_LIMB_T_MAX/17);	/* 8 0-16 */

      x = p01 + p23;							/* 8 0-32 */
      x = (x >> 8) + x;							/* 8 0-64 */
      x = (x >> 16) + x;						/* 8 0-128 */
#if GMP_LIMB_BITS > 32
      x = ((x >> 32) & 0xff) + (x & 0xff);				/* 8 0-256 */
      result += x;
#else
      result += x & 0xff;
#endif
      up += 4;
#if OPERATION_hamdist
      vp += 4;
#endif
    }

  n &= 3;
  if (n != 0)
    {
      x = 0;
      do
	{
	  p0 = POPHAM (up[0], vp[0]);
	  p0 -= (p0 >> 1) & MP_LIMB_T_MAX/3;				/* 2 0-2 */
	  p0 = ((p0 >> 2) & MP_LIMB_T_MAX/5) + (p0 & MP_LIMB_T_MAX/5);	/* 4 0-4 */
	  p0 = ((p0 >> 4) + p0) & MP_LIMB_T_MAX/17;			/* 8 0-8 */

	  x += p0;
	  up += 1;
#if OPERATION_hamdist
	  vp += 1;
#endif
	}
      while (--n);

      x = (x >> 8) + x;
      x = (x >> 16) + x;
#if GMP_LIMB_BITS > 32
      x = (x >> 32) + x;
#endif
      result += x & 0xff;
    }

  return result;
}
