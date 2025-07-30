/* mpz_combit -- complement a specified bit.

Copyright 2002, 2003, 2012, 2015 Free Software Foundation, Inc.

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

void
mpz_combit (mpz_ptr d, mp_bitcnt_t bit_index)
{
  mp_size_t dsize = SIZ(d);
  mp_ptr dp = PTR(d);

  mp_size_t limb_index = bit_index / GMP_NUMB_BITS;
  mp_limb_t bit = (CNST_LIMB (1) << (bit_index % GMP_NUMB_BITS));

  /* Check for the most common case: Positive input, no realloc or
     normalization needed. */
  if (limb_index + 1 < dsize)
    dp[limb_index] ^= bit;

  /* Check for the hairy case. d < 0, and we have all zero bits to the
     right of the bit to toggle. */
  else if (limb_index < -dsize
	   && (limb_index == 0 || mpn_zero_p (dp, limb_index))
	   && (dp[limb_index] & (bit - 1)) == 0)
    {
      ASSERT (dsize < 0);
      dsize = -dsize;

      if (dp[limb_index] & bit)
	{
	  /* We toggle the least significant one bit. Corresponds to
	     an add, with potential carry propagation, on the absolute
	     value. */
	  dp = MPZ_REALLOC (d, 1 + dsize);
	  dp[dsize] = 0;
	  MPN_INCR_U (dp + limb_index, 1 + dsize - limb_index, bit);
	  SIZ(d) = - dsize - dp[dsize];
	}
      else
	{
	  /* We toggle a zero bit, subtract from the absolute value. */
	  MPN_DECR_U (dp + limb_index, dsize - limb_index, bit);
	  /* The absolute value shrinked by at most one bit. */
	  dsize -= dp[dsize - 1] == 0;
	  ASSERT (dsize > 0 && dp[dsize - 1] != 0);
	  SIZ (d) = -dsize;
	}
    }
  else
    {
      /* Simple case: Toggle the bit in the absolute value. */
      dsize = ABS(dsize);
      if (limb_index < dsize)
	{
	  mp_limb_t	 dlimb;
	  dlimb = dp[limb_index] ^ bit;
	  dp[limb_index] = dlimb;

	  /* Can happen only when limb_index = dsize - 1. Avoid SIZ(d)
	     bookkeeping in the common case. */
	  if (UNLIKELY ((dlimb == 0) + limb_index == dsize)) /* dsize == limb_index + 1 */
	    {
	      /* high limb became zero, must normalize */
	      MPN_NORMALIZE (dp, limb_index);
	      SIZ (d) = SIZ (d) >= 0 ? limb_index : -limb_index;
	    }
	}
      else
	{
	  dp = MPZ_REALLOC (d, limb_index + 1);
	  MPN_ZERO(dp + dsize, limb_index - dsize);
	  dp[limb_index++] = bit;
	  SIZ(d) = SIZ(d) >= 0 ? limb_index : -limb_index;
	}
    }
}
