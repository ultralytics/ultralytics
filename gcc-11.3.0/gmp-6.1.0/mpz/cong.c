/* mpz_congruent_p -- test congruence of two mpz's.

Copyright 2001, 2002, 2005 Free Software Foundation, Inc.

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


/* For big divisors this code is only very slightly better than the user
   doing a combination of mpz_sub and mpz_tdiv_r, but it's quite convenient,
   and perhaps in the future can be improved, in similar ways to
   mpn_divisible_p perhaps.

   The csize==1 / dsize==1 special case makes mpz_congruent_p as good as
   mpz_congruent_ui_p on relevant operands, though such a combination
   probably doesn't occur often.

   Alternatives:

   If c<d then it'd work to just form a%d and compare a and c (either as
   a==c or a+c==d depending on the signs), but the saving from avoiding the
   abs(a-c) calculation would be small compared to the division.

   Similarly if both a<d and c<d then it would work to just compare a and c
   (a==c or a+c==d), but this isn't considered a particularly important case
   and so isn't done for the moment.

   Low zero limbs on d could be stripped and the corresponding limbs of a
   and c tested and skipped, but doing so would introduce a borrow when a
   and c differ in sign and have non-zero skipped limbs.  It doesn't seem
   worth the complications to do this, since low zero limbs on d should
   occur only rarely.  */

int
mpz_congruent_p (mpz_srcptr a, mpz_srcptr c, mpz_srcptr d)
{
  mp_size_t  asize, csize, dsize, sign;
  mp_srcptr  ap, cp, dp;
  mp_ptr     xp;
  mp_limb_t  alow, clow, dlow, dmask, r;
  int        result;
  TMP_DECL;

  dsize = SIZ(d);
  if (UNLIKELY (dsize == 0))
    return (mpz_cmp (a, c) == 0);

  dsize = ABS(dsize);
  dp = PTR(d);

  if (ABSIZ(a) < ABSIZ(c))
    MPZ_SRCPTR_SWAP (a, c);

  asize = SIZ(a);
  csize = SIZ(c);
  sign = (asize ^ csize);

  asize = ABS(asize);
  ap = PTR(a);

  if (csize == 0)
    return mpn_divisible_p (ap, asize, dp, dsize);

  csize = ABS(csize);
  cp = PTR(c);

  alow = ap[0];
  clow = cp[0];
  dlow = dp[0];

  /* Check a==c mod low zero bits of dlow.  This might catch a few cases of
     a!=c quickly, and it helps the csize==1 special cases below.  */
  dmask = LOW_ZEROS_MASK (dlow) & GMP_NUMB_MASK;
  alow = (sign >= 0 ? alow : -alow);
  if (((alow-clow) & dmask) != 0)
    return 0;

  if (csize == 1)
    {
      if (dsize == 1)
	{
	cong_1:
	  if (sign < 0)
	    NEG_MOD (clow, clow, dlow);

	  if (ABOVE_THRESHOLD (asize, BMOD_1_TO_MOD_1_THRESHOLD))
	    {
	      r = mpn_mod_1 (ap, asize, dlow);
	      if (clow < dlow)
		return r == clow;
	      else
		return r == (clow % dlow);
	    }

	  if ((dlow & 1) == 0)
	    {
	      /* Strip low zero bits to get odd d required by modexact.  If
		 d==e*2^n then a==c mod d if and only if both a==c mod e and
		 a==c mod 2^n, the latter having been done above.  */
	      unsigned	twos;
	      count_trailing_zeros (twos, dlow);
	      dlow >>= twos;
	    }

	  r = mpn_modexact_1c_odd (ap, asize, dlow, clow);
	  return r == 0 || r == dlow;
	}

      /* dlow==0 is avoided since we don't want to bother handling extra low
	 zero bits if dsecond is even (would involve borrow if a,c differ in
	 sign and alow,clow!=0).  */
      if (dsize == 2 && dlow != 0)
	{
	  mp_limb_t  dsecond = dp[1];

	  if (dsecond <= dmask)
	    {
	      unsigned	 twos;
	      count_trailing_zeros (twos, dlow);
	      dlow = (dlow >> twos) | (dsecond << (GMP_NUMB_BITS-twos));
	      ASSERT_LIMB (dlow);

	      /* dlow will be odd here, so the test for it even under cong_1
		 is unnecessary, but the rest of that code is wanted. */
	      goto cong_1;
	    }
	}
    }

  TMP_MARK;
  xp = TMP_ALLOC_LIMBS (asize+1);

  /* calculate abs(a-c) */
  if (sign >= 0)
    {
      /* same signs, subtract */
      if (asize > csize || mpn_cmp (ap, cp, asize) >= 0)
	ASSERT_NOCARRY (mpn_sub (xp, ap, asize, cp, csize));
      else
	ASSERT_NOCARRY (mpn_sub_n (xp, cp, ap, asize));
      MPN_NORMALIZE (xp, asize);
    }
  else
    {
      /* different signs, add */
      mp_limb_t  carry;
      carry = mpn_add (xp, ap, asize, cp, csize);
      xp[asize] = carry;
      asize += (carry != 0);
    }

  result = mpn_divisible_p (xp, asize, dp, dsize);

  TMP_FREE;
  return result;
}
