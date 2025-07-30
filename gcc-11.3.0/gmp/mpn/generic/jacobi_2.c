/* jacobi_2.c

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 1996, 1998, 2000-2004, 2008, 2010 Free Software Foundation, Inc.

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

#ifndef JACOBI_2_METHOD
#define JACOBI_2_METHOD 2
#endif

/* Computes (a / b) where b is odd, and a and b are otherwise arbitrary
   two-limb numbers. */
#if JACOBI_2_METHOD == 1
int
mpn_jacobi_2 (mp_srcptr ap, mp_srcptr bp, unsigned bit)
{
  mp_limb_t ah, al, bh, bl;
  int c;

  al = ap[0];
  ah = ap[1];
  bl = bp[0];
  bh = bp[1];

  ASSERT (bl & 1);

  bl = ((bh << (GMP_NUMB_BITS - 1)) & GMP_NUMB_MASK) | (bl >> 1);
  bh >>= 1;

  if ( (bh | bl) == 0)
    return 1 - 2*(bit & 1);

  if ( (ah | al) == 0)
    return 0;

  if (al == 0)
    {
      al = ah;
      ah = 0;
      bit ^= GMP_NUMB_BITS & (bl ^ (bl >> 1));
    }
  count_trailing_zeros (c, al);
  bit ^= c & (bl ^ (bl >> 1));

  c++;
  if (UNLIKELY (c == GMP_NUMB_BITS))
    {
      al = ah;
      ah = 0;
    }
  else
    {
      al = ((ah << (GMP_NUMB_BITS - c)) & GMP_NUMB_MASK) | (al >> c);
      ah >>= c;
    }
  while ( (ah | bh) > 0)
    {
      mp_limb_t th, tl;
      mp_limb_t bgta;

      sub_ddmmss (th, tl, ah, al, bh, bl);
      if ( (tl | th) == 0)
	return 0;

      bgta = LIMB_HIGHBIT_TO_MASK (th);

      /* If b > a, invoke reciprocity */
      bit ^= (bgta & al & bl);

      /* b <-- min (a, b) */
      add_ssaaaa (bh, bl, bh, bl, th & bgta, tl & bgta);

      if ( (bh | bl) == 0)
	return 1 - 2*(bit & 1);

      /* a <-- |a - b| */
      al = (bgta ^ tl) - bgta;
      ah = (bgta ^ th);

      if (UNLIKELY (al == 0))
	{
	  /* If b > a, al == 0 implies that we have a carry to
	     propagate. */
	  al = ah - bgta;
	  ah = 0;
	  bit ^= GMP_NUMB_BITS & (bl ^ (bl >> 1));
	}
      count_trailing_zeros (c, al);
      c++;
      bit ^= c & (bl ^ (bl >> 1));

      if (UNLIKELY (c == GMP_NUMB_BITS))
	{
	  al = ah;
	  ah = 0;
	}
      else
	{
	  al = ((ah << (GMP_NUMB_BITS - c)) & GMP_NUMB_MASK) | (al >> c);
	  ah >>= c;
	}
    }

  ASSERT (bl > 0);

  while ( (al | bl) & GMP_LIMB_HIGHBIT)
    {
      /* Need an extra comparison to get the mask. */
      mp_limb_t t = al - bl;
      mp_limb_t bgta = - (bl > al);

      if (t == 0)
	return 0;

      /* If b > a, invoke reciprocity */
      bit ^= (bgta & al & bl);

      /* b <-- min (a, b) */
      bl += (bgta & t);

      /* a <-- |a - b| */
      al = (t ^ bgta) - bgta;

      /* Number of trailing zeros is the same no matter if we look at
       * t or a, but using t gives more parallelism. */
      count_trailing_zeros (c, t);
      c ++;
      /* (2/b) = -1 if b = 3 or 5 mod 8 */
      bit ^= c & (bl ^ (bl >> 1));

      if (UNLIKELY (c == GMP_NUMB_BITS))
	return 1 - 2*(bit & 1);

      al >>= c;
    }

  /* Here we have a little impedance mismatch. Better to inline it? */
  return mpn_jacobi_base (2*al+1, 2*bl+1, bit << 1);
}
#elif JACOBI_2_METHOD == 2
int
mpn_jacobi_2 (mp_srcptr ap, mp_srcptr bp, unsigned bit)
{
  mp_limb_t ah, al, bh, bl;
  int c;

  al = ap[0];
  ah = ap[1];
  bl = bp[0];
  bh = bp[1];

  ASSERT (bl & 1);

  /* Use bit 1. */
  bit <<= 1;

  if (bh == 0 && bl == 1)
    /* (a|1) = 1 */
    return 1 - (bit & 2);

  if (al == 0)
    {
      if (ah == 0)
	/* (0|b) = 0, b > 1 */
	return 0;

      count_trailing_zeros (c, ah);
      bit ^= ((GMP_NUMB_BITS + c) << 1) & (bl ^ (bl >> 1));

      al = bl;
      bl = ah >> c;

      if (bl == 1)
	/* (1|b) = 1 */
	return 1 - (bit & 2);

      ah = bh;

      bit ^= al & bl;

      goto b_reduced;
    }
  if ( (al & 1) == 0)
    {
      count_trailing_zeros (c, al);

      al = ((ah << (GMP_NUMB_BITS - c)) & GMP_NUMB_MASK) | (al >> c);
      ah >>= c;
      bit ^= (c << 1) & (bl ^ (bl >> 1));
    }
  if (ah == 0)
    {
      if (bh > 0)
	{
	  bit ^= al & bl;
	  MP_LIMB_T_SWAP (al, bl);
	  ah = bh;
	  goto b_reduced;
	}
      goto ab_reduced;
    }

  while (bh > 0)
    {
      /* Compute (a|b) */
      while (ah > bh)
	{
	  sub_ddmmss (ah, al, ah, al, bh, bl);
	  if (al == 0)
	    {
	      count_trailing_zeros (c, ah);
	      bit ^= ((GMP_NUMB_BITS + c) << 1) & (bl ^ (bl >> 1));

	      al = bl;
	      bl = ah >> c;
	      ah = bh;

	      bit ^= al & bl;
	      goto b_reduced;
	    }
	  count_trailing_zeros (c, al);
	  bit ^= (c << 1) & (bl ^ (bl >> 1));
	  al = ((ah << (GMP_NUMB_BITS - c)) & GMP_NUMB_MASK) | (al >> c);
	  ah >>= c;
	}
      if (ah == bh)
	goto cancel_hi;

      if (ah == 0)
	{
	  bit ^= al & bl;
	  MP_LIMB_T_SWAP (al, bl);
	  ah = bh;
	  break;
	}

      bit ^= al & bl;

      /* Compute (b|a) */
      while (bh > ah)
	{
	  sub_ddmmss (bh, bl, bh, bl, ah, al);
	  if (bl == 0)
	    {
	      count_trailing_zeros (c, bh);
	      bit ^= ((GMP_NUMB_BITS + c) << 1) & (al ^ (al >> 1));

	      bl = bh >> c;
	      bit ^= al & bl;
	      goto b_reduced;
	    }
	  count_trailing_zeros (c, bl);
	  bit ^= (c << 1) & (al ^ (al >> 1));
	  bl = ((bh << (GMP_NUMB_BITS - c)) & GMP_NUMB_MASK) | (bl >> c);
	  bh >>= c;
	}
      bit ^= al & bl;

      /* Compute (a|b) */
      if (ah == bh)
	{
	cancel_hi:
	  if (al < bl)
	    {
	      MP_LIMB_T_SWAP (al, bl);
	      bit ^= al & bl;
	    }
	  al -= bl;
	  if (al == 0)
	    return 0;

	  count_trailing_zeros (c, al);
	  bit ^= (c << 1) & (bl ^ (bl >> 1));
	  al >>= c;

	  if (al == 1)
	    return 1 - (bit & 2);

	  MP_LIMB_T_SWAP (al, bl);
	  bit ^= al & bl;
	  break;
	}
    }

 b_reduced:
  /* Compute (a|b), with b a single limb. */
  ASSERT (bl & 1);

  if (bl == 1)
    /* (a|1) = 1 */
    return 1 - (bit & 2);

  while (ah > 0)
    {
      ah -= (al < bl);
      al -= bl;
      if (al == 0)
	{
	  if (ah == 0)
	    return 0;
	  count_trailing_zeros (c, ah);
	  bit ^= ((GMP_NUMB_BITS + c) << 1) & (bl ^ (bl >> 1));
	  al = ah >> c;
	  goto ab_reduced;
	}
      count_trailing_zeros (c, al);

      al = ((ah << (GMP_NUMB_BITS - c)) & GMP_NUMB_MASK) | (al >> c);
      ah >>= c;
      bit ^= (c << 1) & (bl ^ (bl >> 1));
    }
 ab_reduced:
  ASSERT (bl & 1);
  ASSERT (bl > 1);

  return mpn_jacobi_base (al, bl, bit);
}
#else
#error Unsupported value for JACOBI_2_METHOD
#endif
