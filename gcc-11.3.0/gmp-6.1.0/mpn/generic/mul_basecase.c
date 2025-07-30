/* mpn_mul_basecase -- Internal routine to multiply two natural numbers
   of length m and n.

   THIS IS AN INTERNAL FUNCTION WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH THIS FUNCTION THROUGH DOCUMENTED INTERFACES.

Copyright 1991-1994, 1996, 1997, 2000-2002 Free Software Foundation, Inc.

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


/* Multiply {up,usize} by {vp,vsize} and write the result to
   {prodp,usize+vsize}.  Must have usize>=vsize.

   Note that prodp gets usize+vsize limbs stored, even if the actual result
   only needs usize+vsize-1.

   There's no good reason to call here with vsize>=MUL_TOOM22_THRESHOLD.
   Currently this is allowed, but it might not be in the future.

   This is the most critical code for multiplication.  All multiplies rely
   on this, both small and huge.  Small ones arrive here immediately, huge
   ones arrive here as this is the base case for Karatsuba's recursive
   algorithm.  */

void
mpn_mul_basecase (mp_ptr rp,
		  mp_srcptr up, mp_size_t un,
		  mp_srcptr vp, mp_size_t vn)
{
  ASSERT (un >= vn);
  ASSERT (vn >= 1);
  ASSERT (! MPN_OVERLAP_P (rp, un+vn, up, un));
  ASSERT (! MPN_OVERLAP_P (rp, un+vn, vp, vn));

  /* We first multiply by the low order limb (or depending on optional function
     availability, limbs).  This result can be stored, not added, to rp.  We
     also avoid a loop for zeroing this way.  */

#if HAVE_NATIVE_mpn_mul_2
  if (vn >= 2)
    {
      rp[un + 1] = mpn_mul_2 (rp, up, un, vp);
      rp += 2, vp += 2, vn -= 2;
    }
  else
    {
      rp[un] = mpn_mul_1 (rp, up, un, vp[0]);
      return;
    }
#else
  rp[un] = mpn_mul_1 (rp, up, un, vp[0]);
  rp += 1, vp += 1, vn -= 1;
#endif

  /* Now accumulate the product of up[] and the next higher limb (or depending
     on optional function availability, limbs) from vp[].  */

#define MAX_LEFT MP_SIZE_T_MAX	/* Used to simplify loops into if statements */


#if HAVE_NATIVE_mpn_addmul_6
  while (vn >= 6)
    {
      rp[un + 6 - 1] = mpn_addmul_6 (rp, up, un, vp);
      if (MAX_LEFT == 6)
	return;
      rp += 6, vp += 6, vn -= 6;
      if (MAX_LEFT < 2 * 6)
	break;
    }
#undef MAX_LEFT
#define MAX_LEFT (6 - 1)
#endif

#if HAVE_NATIVE_mpn_addmul_5
  while (vn >= 5)
    {
      rp[un + 5 - 1] = mpn_addmul_5 (rp, up, un, vp);
      if (MAX_LEFT == 5)
	return;
      rp += 5, vp += 5, vn -= 5;
      if (MAX_LEFT < 2 * 5)
	break;
    }
#undef MAX_LEFT
#define MAX_LEFT (5 - 1)
#endif

#if HAVE_NATIVE_mpn_addmul_4
  while (vn >= 4)
    {
      rp[un + 4 - 1] = mpn_addmul_4 (rp, up, un, vp);
      if (MAX_LEFT == 4)
	return;
      rp += 4, vp += 4, vn -= 4;
      if (MAX_LEFT < 2 * 4)
	break;
    }
#undef MAX_LEFT
#define MAX_LEFT (4 - 1)
#endif

#if HAVE_NATIVE_mpn_addmul_3
  while (vn >= 3)
    {
      rp[un + 3 - 1] = mpn_addmul_3 (rp, up, un, vp);
      if (MAX_LEFT == 3)
	return;
      rp += 3, vp += 3, vn -= 3;
      if (MAX_LEFT < 2 * 3)
	break;
    }
#undef MAX_LEFT
#define MAX_LEFT (3 - 1)
#endif

#if HAVE_NATIVE_mpn_addmul_2
  while (vn >= 2)
    {
      rp[un + 2 - 1] = mpn_addmul_2 (rp, up, un, vp);
      if (MAX_LEFT == 2)
	return;
      rp += 2, vp += 2, vn -= 2;
      if (MAX_LEFT < 2 * 2)
	break;
    }
#undef MAX_LEFT
#define MAX_LEFT (2 - 1)
#endif

  while (vn >= 1)
    {
      rp[un] = mpn_addmul_1 (rp, up, un, vp[0]);
      if (MAX_LEFT == 1)
	return;
      rp += 1, vp += 1, vn -= 1;
    }
}
