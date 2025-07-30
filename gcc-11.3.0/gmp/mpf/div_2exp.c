/* mpf_div_2exp -- Divide a float by 2^n.

Copyright 1993, 1994, 1996, 2000-2002, 2004 Free Software Foundation, Inc.

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


/* Multiples of GMP_NUMB_BITS in exp simply mean an amount subtracted from
   EXP(u) to set EXP(r).  The remainder exp%GMP_NUMB_BITS is then a right
   shift for the limb data.

   If exp%GMP_NUMB_BITS == 0 then there's no shifting, we effectively just
   do an mpz_set with changed EXP(r).  Like mpz_set we take prec+1 limbs in
   this case.  Although just prec would suffice, it's nice to have
   mpf_div_2exp with exp==0 come out the same as mpz_set.

   When shifting we take up to prec many limbs from the input.  Our shift is
   cy = mpn_rshift (PTR(r)+1, PTR(u)+k, ...), where k is the number of low
   limbs dropped from u, and the carry out is stored to PTR(r)[0].  We don't
   try to work extra bits from PTR(u)[k-1] (when k>=1 makes it available)
   into that low carry limb.  Just prec limbs (with the high non-zero) from
   the input is enough bits for the application requested precision, no need
   to do extra work.

   If r==u the shift will have overlapping operands.  When k>=1 (ie. when
   usize > prec), the overlap is in the style supported by rshift (ie. dst
   <= src).

   But when r==u and k==0 (ie. usize <= prec), we would have an invalid
   overlap (mpn_rshift (rp+1, rp, ...)).  In this case we must instead use
   mpn_lshift (PTR(r), PTR(u), size, NUMB-shift).  An lshift by NUMB-shift
   bits gives identical data of course, it's just its overlap restrictions
   which differ.

   In both shift cases, the resulting data is abs_usize+1 limbs.  "adj" is
   used to add +1 to that size if the high is non-zero (it may of course
   have become zero by the shifting).  EXP(u) is the exponent just above
   those abs_usize+1 limbs, so it gets -1+adj, which means -1 if the high is
   zero, or no change if the high is non-zero.

   Enhancements:

   The way mpn_lshift is used means successive mpf_div_2exp calls on the
   same operand will accumulate low zero limbs, until prec+1 limbs is
   reached.  This is wasteful for subsequent operations.  When abs_usize <=
   prec, we should test the low exp%GMP_NUMB_BITS many bits of PTR(u)[0],
   ie. those which would be shifted out by an mpn_rshift.  If they're zero
   then use that mpn_rshift.  */

void
mpf_div_2exp (mpf_ptr r, mpf_srcptr u, mp_bitcnt_t exp)
{
  mp_srcptr up;
  mp_ptr rp = r->_mp_d;
  mp_size_t usize;
  mp_size_t abs_usize;
  mp_size_t prec = r->_mp_prec;
  mp_exp_t uexp = u->_mp_exp;

  usize = u->_mp_size;

  if (UNLIKELY (usize == 0))
    {
      r->_mp_size = 0;
      r->_mp_exp = 0;
      return;
    }

  abs_usize = ABS (usize);
  up = u->_mp_d;

  if (exp % GMP_NUMB_BITS == 0)
    {
      prec++;			/* retain more precision here as we don't need
				   to account for carry-out here */
      if (abs_usize > prec)
	{
	  up += abs_usize - prec;
	  abs_usize = prec;
	}
      if (rp != up)
	MPN_COPY_INCR (rp, up, abs_usize);
      r->_mp_exp = uexp - exp / GMP_NUMB_BITS;
    }
  else
    {
      mp_limb_t cy_limb;
      mp_size_t adj;
      if (abs_usize > prec)
	{
	  up += abs_usize - prec;
	  abs_usize = prec;
	  /* Use mpn_rshift since mpn_lshift operates downwards, and we
	     therefore would clobber part of U before using that part, in case
	     R is the same variable as U.  */
	  cy_limb = mpn_rshift (rp + 1, up, abs_usize, exp % GMP_NUMB_BITS);
	  rp[0] = cy_limb;
	  adj = rp[abs_usize] != 0;
	}
      else
	{
	  cy_limb = mpn_lshift (rp, up, abs_usize,
				GMP_NUMB_BITS - exp % GMP_NUMB_BITS);
	  rp[abs_usize] = cy_limb;
	  adj = cy_limb != 0;
	}

      abs_usize += adj;
      r->_mp_exp = uexp - exp / GMP_NUMB_BITS - 1 + adj;
    }
  r->_mp_size = usize >= 0 ? abs_usize : -abs_usize;
}
