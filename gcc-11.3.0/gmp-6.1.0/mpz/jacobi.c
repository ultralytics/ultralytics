/* mpz_jacobi, mpz_legendre, mpz_kronecker -- mpz/mpz Jacobi symbols.

Copyright 2000-2002, 2005, 2010-2012 Free Software Foundation, Inc.

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

#include <stdio.h>
#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"


/* This code does triple duty as mpz_jacobi, mpz_legendre and
   mpz_kronecker. For ABI compatibility, the link symbol is
   __gmpz_jacobi, not __gmpz_kronecker, even though the latter would
   be more logical.

   mpz_jacobi could assume b is odd, but the improvements from that seem
   small compared to other operations, and anything significant should be
   checked at run-time since we'd like odd b to go fast in mpz_kronecker
   too.

   mpz_legendre could assume b is an odd prime, but knowing this doesn't
   present any obvious benefits.  Result 0 wouldn't arise (unless "a" is a
   multiple of b), but the checking for that takes little time compared to
   other operations.

   Enhancements:

   mpn_bdiv_qr should be used instead of mpn_tdiv_qr.

*/

int
mpz_jacobi (mpz_srcptr a, mpz_srcptr b)
{
  mp_srcptr  asrcp, bsrcp;
  mp_size_t  asize, bsize;
  mp_limb_t  alow, blow;
  mp_ptr     ap, bp;
  unsigned   btwos;
  int        result_bit1;
  int        res;
  TMP_DECL;

  asize = SIZ(a);
  asrcp = PTR(a);
  alow = asrcp[0];

  bsize = SIZ(b);
  bsrcp = PTR(b);
  blow = bsrcp[0];

  /* The MPN jacobi functions require positive a and b, and b odd. So
     we must to handle the cases of a or b zero, then signs, and then
     the case of even b.
  */

  if (bsize == 0)
    /* (a/0) = [ a = 1 or a = -1 ] */
    return JACOBI_LS0 (alow, asize);

  if (asize == 0)
    /* (0/b) = [ b = 1 or b = - 1 ] */
    return JACOBI_0LS (blow, bsize);

  if ( (((alow | blow) & 1) == 0))
    /* Common factor of 2 ==> (a/b) = 0 */
    return 0;

  if (bsize < 0)
    {
      /* (a/-1) = -1 if a < 0, +1 if a >= 0 */
      result_bit1 = (asize < 0) << 1;
      bsize = -bsize;
    }
  else
    result_bit1 = 0;

  JACOBI_STRIP_LOW_ZEROS (result_bit1, alow, bsrcp, bsize, blow);

  count_trailing_zeros (btwos, blow);
  blow >>= btwos;

  if (bsize > 1 && btwos > 0)
    {
      mp_limb_t b1 = bsrcp[1];
      blow |= b1 << (GMP_NUMB_BITS - btwos);
      if (bsize == 2 && (b1 >> btwos) == 0)
	bsize = 1;
    }

  if (asize < 0)
    {
      /* (-1/b) = -1 iff b = 3 (mod 4) */
      result_bit1 ^= JACOBI_N1B_BIT1(blow);
      asize = -asize;
    }

  JACOBI_STRIP_LOW_ZEROS (result_bit1, blow, asrcp, asize, alow);

  /* Ensure asize >= bsize. Take advantage of the generalized
     reciprocity law (a/b*2^n) = (b*2^n / a) * RECIP(a,b) */

  if (asize < bsize)
    {
      MPN_SRCPTR_SWAP (asrcp, asize, bsrcp, bsize);
      MP_LIMB_T_SWAP (alow, blow);

      /* NOTE: The value of alow (old blow) is a bit subtle. For this code
	 path, we get alow as the low, always odd, limb of shifted A. Which is
	 what we need for the reciprocity update below.

	 However, all other uses of alow assumes that it is *not*
	 shifted. Luckily, alow matters only when either

	 + btwos > 0, in which case A is always odd

	 + asize == bsize == 1, in which case this code path is never
	   taken. */

      count_trailing_zeros (btwos, blow);
      blow >>= btwos;

      if (bsize > 1 && btwos > 0)
	{
	  mp_limb_t b1 = bsrcp[1];
	  blow |= b1 << (GMP_NUMB_BITS - btwos);
	  if (bsize == 2 && (b1 >> btwos) == 0)
	    bsize = 1;
	}

      result_bit1 ^= JACOBI_RECIP_UU_BIT1 (alow, blow);
    }

  if (bsize == 1)
    {
      result_bit1 ^= JACOBI_TWOS_U_BIT1(btwos, alow);

      if (blow == 1)
	return JACOBI_BIT1_TO_PN (result_bit1);

      if (asize > 1)
	JACOBI_MOD_OR_MODEXACT_1_ODD (result_bit1, alow, asrcp, asize, blow);

      return mpn_jacobi_base (alow, blow, result_bit1);
    }

  /* Allocation strategy: For A, we allocate a working copy only for A % B, but
     when A is much larger than B, we have to allocate space for the large
     quotient. We use the same area, pointed to by bp, for both the quotient
     A/B and the working copy of B. */

  TMP_MARK;

  if (asize >= 2*bsize)
    TMP_ALLOC_LIMBS_2 (ap, bsize, bp, asize - bsize + 1);
  else
    TMP_ALLOC_LIMBS_2 (ap, bsize, bp, bsize);

  /* In the case of even B, we conceptually shift out the powers of two first,
     and then divide A mod B. Hence, when taking those powers of two into
     account, we must use alow *before* the division. Doing the actual division
     first is ok, because the point is to remove multiples of B from A, and
     multiples of 2^k B are good enough. */
  if (asize > bsize)
    mpn_tdiv_qr (bp, ap, 0, asrcp, asize, bsrcp, bsize);
  else
    MPN_COPY (ap, asrcp, bsize);

  if (btwos > 0)
    {
      result_bit1 ^= JACOBI_TWOS_U_BIT1(btwos, alow);

      ASSERT_NOCARRY (mpn_rshift (bp, bsrcp, bsize, btwos));
      bsize -= (ap[bsize-1] | bp[bsize-1]) == 0;
    }
  else
    MPN_COPY (bp, bsrcp, bsize);

  ASSERT (blow == bp[0]);
  res = mpn_jacobi_n (ap, bp, bsize,
		      mpn_jacobi_init (ap[0], blow, (result_bit1>>1) & 1));

  TMP_FREE;
  return res;
}
