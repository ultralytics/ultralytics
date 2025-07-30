/* mpn_gcd_1 -- mpn and limb greatest common divisor.

Copyright 1994, 1996, 2000, 2001, 2009, 2012 Free Software Foundation, Inc.

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

#ifndef GCD_1_METHOD
#define GCD_1_METHOD 2
#endif

#define USE_ZEROTAB 0

#if USE_ZEROTAB
#define MAXSHIFT 4
#define MASK ((1 << MAXSHIFT) - 1)
static const unsigned char zerotab[1 << MAXSHIFT] =
{
#if MAXSHIFT > 4
  5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
#endif
  4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0
};
#endif

/* Does not work for U == 0 or V == 0.  It would be tough to make it work for
   V == 0 since gcd(x,0) = x, and U does not generally fit in an mp_limb_t.

   The threshold for doing u%v when size==1 will vary by CPU according to
   the speed of a division and the code generated for the main loop.  Any
   tuning for this is left to a CPU specific implementation.  */

mp_limb_t
mpn_gcd_1 (mp_srcptr up, mp_size_t size, mp_limb_t vlimb)
{
  mp_limb_t      ulimb;
  unsigned long  zero_bits, u_low_zero_bits;

  ASSERT (size >= 1);
  ASSERT (vlimb != 0);
  ASSERT_MPN_NONZERO_P (up, size);

  ulimb = up[0];

  /* Need vlimb odd for modexact, want it odd to get common zeros. */
  count_trailing_zeros (zero_bits, vlimb);
  vlimb >>= zero_bits;

  if (size > 1)
    {
      /* Must get common zeros before the mod reduction.  If ulimb==0 then
	 vlimb already gives the common zeros.  */
      if (ulimb != 0)
	{
	  count_trailing_zeros (u_low_zero_bits, ulimb);
	  zero_bits = MIN (zero_bits, u_low_zero_bits);
	}

      ulimb = MPN_MOD_OR_MODEXACT_1_ODD (up, size, vlimb);
      if (ulimb == 0)
	goto done;

      goto strip_u_maybe;
    }

  /* size==1, so up[0]!=0 */
  count_trailing_zeros (u_low_zero_bits, ulimb);
  ulimb >>= u_low_zero_bits;
  zero_bits = MIN (zero_bits, u_low_zero_bits);

  /* make u bigger */
  if (vlimb > ulimb)
    MP_LIMB_T_SWAP (ulimb, vlimb);

  /* if u is much bigger than v, reduce using a division rather than
     chipping away at it bit-by-bit */
  if ((ulimb >> 16) > vlimb)
    {
      ulimb %= vlimb;
      if (ulimb == 0)
	goto done;
      goto strip_u_maybe;
    }

  ASSERT (ulimb & 1);
  ASSERT (vlimb & 1);

#if GCD_1_METHOD == 1
  while (ulimb != vlimb)
    {
      ASSERT (ulimb & 1);
      ASSERT (vlimb & 1);

      if (ulimb > vlimb)
	{
	  ulimb -= vlimb;
	  do
	    {
	      ulimb >>= 1;
	      ASSERT (ulimb != 0);
	    strip_u_maybe:
	      ;
	    }
	  while ((ulimb & 1) == 0);
	}
      else /*  vlimb > ulimb.  */
	{
	  vlimb -= ulimb;
	  do
	    {
	      vlimb >>= 1;
	      ASSERT (vlimb != 0);
	    }
	  while ((vlimb & 1) == 0);
	}
    }
#else
# if GCD_1_METHOD  == 2

  ulimb >>= 1;
  vlimb >>= 1;

  while (ulimb != vlimb)
    {
      int c;
      mp_limb_t t;
      mp_limb_t vgtu;

      t = ulimb - vlimb;
      vgtu = LIMB_HIGHBIT_TO_MASK (t);

      /* v <-- min (u, v) */
      vlimb += (vgtu & t);

      /* u <-- |u - v| */
      ulimb = (t ^ vgtu) - vgtu;

#if USE_ZEROTAB
      /* Number of trailing zeros is the same no matter if we look at
       * t or ulimb, but using t gives more parallelism. */
      c = zerotab[t & MASK];

      while (UNLIKELY (c == MAXSHIFT))
	{
	  ulimb >>= MAXSHIFT;
	  if (0)
	  strip_u_maybe:
	    vlimb >>= 1;

	  c = zerotab[ulimb & MASK];
	}
#else
      if (0)
	{
	strip_u_maybe:
	  vlimb >>= 1;
	  t = ulimb;
	}
      count_trailing_zeros (c, t);
#endif
      ulimb >>= (c + 1);
    }

  vlimb = (vlimb << 1) | 1;
# else
#  error Unknown GCD_1_METHOD
# endif
#endif

 done:
  return vlimb << zero_bits;
}
