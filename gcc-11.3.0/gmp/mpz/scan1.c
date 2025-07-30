/* mpz_scan1 -- search for a 1 bit.

Copyright 2000-2002, 2004, 2012, 2015 Free Software Foundation, Inc.

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


/* mpn_scan0 can't be used for the inverted u<0 search since there might not
   be a 0 bit before the end of the data.  mpn_scan1 could be used under u>0
   (except when in the high limb), but usually the search won't go very far
   so it seems reasonable to inline that code.  */

mp_bitcnt_t
mpz_scan1 (mpz_srcptr u, mp_bitcnt_t starting_bit) __GMP_NOTHROW
{
  mp_srcptr      u_ptr = PTR(u);
  mp_size_t      size = SIZ(u);
  mp_size_t      abs_size = ABS(size);
  mp_srcptr      u_end = u_ptr + abs_size - 1;
  mp_size_t      starting_limb = starting_bit / GMP_NUMB_BITS;
  mp_srcptr      p = u_ptr + starting_limb;
  mp_limb_t      limb;
  int            cnt;

  /* Past the end there's no 1 bits for u>=0, or an immediate 1 bit for u<0.
     Notice this test picks up any u==0 too. */
  if (starting_limb >= abs_size)
    return (size >= 0 ? ~(mp_bitcnt_t) 0 : starting_bit);

  /* This is an important case, where sign is not relevant! */
  if (starting_bit == 0)
    goto short_cut;

  limb = *p;

  if (size >= 0)
    {
      /* Mask to 0 all bits before starting_bit, thus ignoring them. */
      limb &= (MP_LIMB_T_MAX << (starting_bit % GMP_NUMB_BITS));

      if (limb == 0)
	{
	  /* If it's the high limb which is zero after masking, then there's
	     no 1 bits after starting_bit.  */
	  if (p == u_end)
	    return ~(mp_bitcnt_t) 0;

	  /* Otherwise search further for a non-zero limb.  The high limb is
	     non-zero, if nothing else.  */
	search_nonzero:
	  do
	    {
	      ASSERT (p != u_end);
	      p++;
	    short_cut:
	      limb = *p;
	    }
	  while (limb == 0);
	}
    }
  else
    {
      /* If there's a non-zero limb before ours then we're in the ones
	 complement region.  */
      if (starting_limb == 0 || mpn_zero_p (u_ptr, starting_limb)) {
	if (limb == 0)
	  /* Seeking for the first non-zero bit, it is the same for u and -u. */
	  goto search_nonzero;

	/* Adjust so ~limb implied by searching for 0 bit becomes -limb.  */
	limb--;
      }

      /* Now seeking a 0 bit. */

      /* Mask to 1 all bits before starting_bit, thus ignoring them. */
      limb |= (CNST_LIMB(1) << (starting_bit % GMP_NUMB_BITS)) - 1;

      /* Search for a limb which is not all ones.  If the end is reached
	 then the zero immediately past the end is the result.  */
      while (limb == GMP_NUMB_MAX)
	{
	  if (p == u_end)
	    return (mp_bitcnt_t) abs_size * GMP_NUMB_BITS;
	  p++;
	  limb = *p;
	}

      /* Now seeking low 1 bit. */
      limb = ~limb;
    }

  ASSERT (limb != 0);
  count_trailing_zeros (cnt, limb);
  return (mp_bitcnt_t) (p - u_ptr) * GMP_NUMB_BITS + cnt;
}
