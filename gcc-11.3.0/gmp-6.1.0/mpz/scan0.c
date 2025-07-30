/* mpz_scan0 -- search for a 0 bit.

Copyright 2000-2002, 2004, 2012 Free Software Foundation, Inc.

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


/* mpn_scan0 can't be used for the u>0 search since there might not be a 0
   bit before the end of the data.  mpn_scan1 could be used for the inverted
   search under u<0, but usually the search won't go very far so it seems
   reasonable to inline that code.  */

mp_bitcnt_t
mpz_scan0 (mpz_srcptr u, mp_bitcnt_t starting_bit) __GMP_NOTHROW
{
  mp_srcptr      u_ptr = PTR(u);
  mp_size_t      size = SIZ(u);
  mp_size_t      abs_size = ABS(size);
  mp_srcptr      u_end = u_ptr + abs_size;
  mp_size_t      starting_limb = starting_bit / GMP_NUMB_BITS;
  mp_srcptr      p = u_ptr + starting_limb;
  mp_limb_t      limb;
  int            cnt;

  /* When past end, there's an immediate 0 bit for u>=0, or no 0 bits for
     u<0.  Notice this test picks up all cases of u==0 too. */
  if (starting_limb >= abs_size)
    return (size >= 0 ? starting_bit : ~(mp_bitcnt_t) 0);

  limb = *p;

  if (size >= 0)
    {
      /* Mask to 1 all bits before starting_bit, thus ignoring them. */
      limb |= (CNST_LIMB(1) << (starting_bit % GMP_NUMB_BITS)) - 1;

      /* Search for a limb which isn't all ones.  If the end is reached then
	 the zero bit immediately past the end is returned.  */
      while (limb == GMP_NUMB_MAX)
	{
	  p++;
	  if (p == u_end)
	    return (mp_bitcnt_t) abs_size * GMP_NUMB_BITS;
	  limb = *p;
	}

      /* Now seek low 1 bit. */
      limb = ~limb;
    }
  else
    {
      mp_srcptr  q;

      /* If there's a non-zero limb before ours then we're in the ones
	 complement region.  Search from *(p-1) downwards since that might
	 give better cache locality, and since a non-zero in the middle of a
	 number is perhaps a touch more likely than at the end.  */
      q = p;
      while (q != u_ptr)
	{
	  q--;
	  if (*q != 0)
	    goto inverted;
	}

      /* Adjust so ~limb implied by searching for 1 bit below becomes -limb.
	 If limb==0 here then this isn't the beginning of twos complement
	 inversion, but that doesn't matter because limb==0 is a zero bit
	 immediately (-1 is all ones for below).  */
      limb--;

    inverted:
      /* Now seeking a 1 bit. */

      /* Mask to 0 all bits before starting_bit, thus ignoring them. */
      limb &= (MP_LIMB_T_MAX << (starting_bit % GMP_NUMB_BITS));

      if (limb == 0)
	{
	  /* If the high limb is zero after masking, then no 1 bits past
	     starting_bit.  */
	  p++;
	  if (p == u_end)
	    return ~(mp_bitcnt_t) 0;

	  /* Search further for a non-zero limb.  The high limb is non-zero,
	     if nothing else.  */
	  for (;;)
	    {
	      limb = *p;
	      if (limb != 0)
		break;
	      p++;
	      ASSERT (p < u_end);
	    }
	}
    }

  ASSERT (limb != 0);
  count_trailing_zeros (cnt, limb);
  return (mp_bitcnt_t) (p - u_ptr) * GMP_NUMB_BITS + cnt;
}
