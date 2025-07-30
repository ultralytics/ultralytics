/* mpz_setbit -- set a specified bit.

Copyright 1991, 1993-1995, 1997, 1999, 2001, 2002, 2012 Free Software
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

void
mpz_setbit (mpz_ptr d, mp_bitcnt_t bit_idx)
{
  mp_size_t dsize = SIZ (d);
  mp_ptr dp = PTR (d);
  mp_size_t limb_idx;
  mp_limb_t mask;

  limb_idx = bit_idx / GMP_NUMB_BITS;
  mask = CNST_LIMB(1) << (bit_idx % GMP_NUMB_BITS);
  if (dsize >= 0)
    {
      if (limb_idx < dsize)
	{
	  dp[limb_idx] |= mask;
	}
      else
	{
	  /* Ugh.  The bit should be set outside of the end of the
	     number.  We have to increase the size of the number.  */
	  dp = MPZ_REALLOC (d, limb_idx + 1);
	  SIZ (d) = limb_idx + 1;
	  MPN_ZERO (dp + dsize, limb_idx - dsize);
	  dp[limb_idx] = mask;
	}
    }
  else
    {
      /* Simulate two's complement arithmetic, i.e. simulate
	 1. Set OP = ~(OP - 1) [with infinitely many leading ones].
	 2. Set the bit.
	 3. Set OP = ~OP + 1.  */

      dsize = -dsize;

      if (limb_idx < dsize)
	{
	  mp_size_t zero_bound;
	  /* No index upper bound on this loop, we're sure there's a non-zero limb
	     sooner or later.  */
	  zero_bound = 0;
	  while (dp[zero_bound] == 0)
	    zero_bound++;

	  if (limb_idx > zero_bound)
	    {
	      mp_limb_t	 dlimb;
	      dlimb = dp[limb_idx] & ~mask;
	      dp[limb_idx] = dlimb;

	      if (UNLIKELY ((dlimb == 0) + limb_idx == dsize)) /* dsize == limb_idx + 1 */
		{
		  /* high limb became zero, must normalize */
		  MPN_NORMALIZE (dp, limb_idx);
		  SIZ (d) = -limb_idx;
		}
	    }
	  else if (limb_idx == zero_bound)
	    {
	      dp[limb_idx] = ((dp[limb_idx] - 1) & ~mask) + 1;
	      ASSERT (dp[limb_idx] != 0);
	    }
	  else
	    {
	      MPN_DECR_U (dp + limb_idx, dsize - limb_idx, mask);
	      dsize -= dp[dsize - 1] == 0;
	      SIZ (d) = -dsize;
	    }
	}
    }
}
