/* mpn_pow_1 -- Compute powers R = U^exp.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2002, 2014 Free Software Foundation, Inc.

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

mp_size_t
mpn_pow_1 (mp_ptr rp, mp_srcptr bp, mp_size_t bn, mp_limb_t exp, mp_ptr tp)
{
  mp_limb_t x;
  int cnt, i;
  mp_size_t rn;
  int par;

  ASSERT (bn >= 1);
  /* FIXME: Add operand overlap criteria */

  if (exp <= 1)
    {
      if (exp == 0)
	{
	  rp[0] = 1;
	  return 1;
	}
      else
	{
	  MPN_COPY (rp, bp, bn);
	  return bn;
	}
    }

  /* Count number of bits in exp, and compute where to put initial square in
     order to magically get results in the entry rp.  Use simple code,
     optimized for small exp.  For large exp, the bignum operations will take
     so much time that the slowness of this code will be negligible.  */
  par = 0;
  cnt = GMP_LIMB_BITS;
  x = exp;
  do
    {
      par ^= x;
      cnt--;
      x >>= 1;
    } while (x != 0);
  exp <<= cnt;

  if (bn == 1)
    {
      mp_limb_t bl = bp[0];

      if ((cnt & 1) != 0)
	MP_PTR_SWAP (rp, tp);

      mpn_sqr (rp, bp, bn);
      rn = 2 * bn; rn -= rp[rn - 1] == 0;

      for (i = GMP_LIMB_BITS - cnt - 1;;)
	{
	  exp <<= 1;
	  if ((exp & GMP_LIMB_HIGHBIT) != 0)
	    {
	      rp[rn] = mpn_mul_1 (rp, rp, rn, bl);
	      rn += rp[rn] != 0;
	    }

	  if (--i == 0)
	    break;

	  mpn_sqr (tp, rp, rn);
	  rn = 2 * rn; rn -= tp[rn - 1] == 0;
	  MP_PTR_SWAP (rp, tp);
	}
    }
  else
    {
      if (((par ^ cnt) & 1) == 0)
	MP_PTR_SWAP (rp, tp);

      mpn_sqr (rp, bp, bn);
      rn = 2 * bn; rn -= rp[rn - 1] == 0;

      for (i = GMP_LIMB_BITS - cnt - 1;;)
	{
	  exp <<= 1;
	  if ((exp & GMP_LIMB_HIGHBIT) != 0)
	    {
	      rn = rn + bn - (mpn_mul (tp, rp, rn, bp, bn) == 0);
	      MP_PTR_SWAP (rp, tp);
	    }

	  if (--i == 0)
	    break;

	  mpn_sqr (tp, rp, rn);
	  rn = 2 * rn; rn -= tp[rn - 1] == 0;
	  MP_PTR_SWAP (rp, tp);
	}
    }

  return rn;
}
