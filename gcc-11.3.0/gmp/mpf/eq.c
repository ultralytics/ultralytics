/* mpf_eq -- Compare two floats up to a specified bit #.

Copyright 1993, 1995, 1996, 2001, 2002, 2008, 2009, 2012 Free Software
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
#include "longlong.h"

int
mpf_eq (mpf_srcptr u, mpf_srcptr v, mp_bitcnt_t n_bits)
{
  mp_srcptr up, vp, p;
  mp_size_t usize, vsize, minsize, maxsize, n_limbs, i, size;
  mp_exp_t uexp, vexp;
  mp_limb_t diff;
  int cnt;

  uexp = u->_mp_exp;
  vexp = v->_mp_exp;

  usize = u->_mp_size;
  vsize = v->_mp_size;

  /* 1. Are the signs different?  */
  if ((usize ^ vsize) >= 0)
    {
      /* U and V are both non-negative or both negative.  */
      if (usize == 0)
	return vsize == 0;
      if (vsize == 0)
	return 0;

      /* Fall out.  */
    }
  else
    {
      /* Either U or V is negative, but not both.  */
      return 0;
    }

  /* U and V have the same sign and are both non-zero.  */

  /* 2. Are the exponents different?  */
  if (uexp != vexp)
    return 0;

  usize = ABS (usize);
  vsize = ABS (vsize);

  up = u->_mp_d;
  vp = v->_mp_d;

  up += usize;			/* point just above most significant limb */
  vp += vsize;			/* point just above most significant limb */

  count_leading_zeros (cnt, up[-1]);
  if ((vp[-1] >> (GMP_LIMB_BITS - 1 - cnt)) != 1)
    return 0;			/* msb positions different */

  n_bits += cnt - GMP_NAIL_BITS;
  n_limbs = (n_bits + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS;

  usize = MIN (usize, n_limbs);
  vsize = MIN (vsize, n_limbs);

#if 0
  /* Ignore zeros at the low end of U and V.  */
  while (up[0] == 0)
    up++, usize--;
  while (vp[0] == 0)
    vp++, vsize--;
#endif

  minsize = MIN (usize, vsize);
  maxsize = usize + vsize - minsize;

  up -= minsize;		/* point at most significant common limb */
  vp -= minsize;		/* point at most significant common limb */

  /* Compare the most significant part which has explicit limbs for U and V. */
  for (i = minsize - 1; i > 0; i--)
    {
      if (up[i] != vp[i])
	return 0;
    }

  n_bits -= (maxsize - 1) * GMP_NUMB_BITS;

  size = maxsize - minsize;
  if (size != 0)
    {
      if (up[0] != vp[0])
	return 0;

      /* Now either U or V has its limbs consumed, i.e, continues with an
	 infinite number of implicit zero limbs.  Check that the other operand
	 has just zeros in the corresponding, relevant part.  */

      if (usize > vsize)
	p = up - size;
      else
	p = vp - size;

      for (i = size - 1; i > 0; i--)
	{
	  if (p[i] != 0)
	    return 0;
	}

      diff = p[0];
    }
  else
    {
      /* Both U or V has its limbs consumed.  */

      diff = up[0] ^ vp[0];
    }

  if (n_bits < GMP_NUMB_BITS)
    diff >>= GMP_NUMB_BITS - n_bits;

  return diff == 0;
}
