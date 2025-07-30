/* mpn_divisible_p -- mpn by mpn divisibility test

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2001, 2002, 2005, 2009, 2014 Free Software Foundation, Inc.

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


/* Determine whether A={ap,an} is divisible by D={dp,dn}.  Must have both
   operands normalized, meaning high limbs non-zero, except that an==0 is
   allowed.

   There usually won't be many low zero bits on D, but the checks for this
   are fast and might pick up a few operand combinations, in particular they
   might reduce D to fit the single-limb mod_1/modexact_1 code.

   Future:

   Getting the remainder limb by limb would make an early exit possible on
   finding a non-zero.  This would probably have to be bdivmod style so
   there's no addback, but it would need a multi-precision inverse and so
   might be slower than the plain method (on small sizes at least).

   When D must be normalized (shifted to low bit set), it's possible to
   suppress the bit-shifting of A down, as long as it's already been checked
   that A has at least as many trailing zero bits as D.  */

int
mpn_divisible_p (mp_srcptr ap, mp_size_t an,
		 mp_srcptr dp, mp_size_t dn)
{
  mp_limb_t  alow, dlow, dmask;
  mp_ptr     qp, rp, tp;
  mp_size_t  i;
  mp_limb_t di;
  unsigned  twos;
  TMP_DECL;

  ASSERT (an >= 0);
  ASSERT (an == 0 || ap[an-1] != 0);
  ASSERT (dn >= 1);
  ASSERT (dp[dn-1] != 0);
  ASSERT_MPN (ap, an);
  ASSERT_MPN (dp, dn);

  /* When a<d only a==0 is divisible.
     Notice this test covers all cases of an==0. */
  if (an < dn)
    return (an == 0);

  /* Strip low zero limbs from d, requiring a==0 on those. */
  for (;;)
    {
      alow = *ap;
      dlow = *dp;

      if (dlow != 0)
	break;

      if (alow != 0)
	return 0;  /* a has fewer low zero limbs than d, so not divisible */

      /* a!=0 and d!=0 so won't get to n==0 */
      an--; ASSERT (an >= 1);
      dn--; ASSERT (dn >= 1);
      ap++;
      dp++;
    }

  /* a must have at least as many low zero bits as d */
  dmask = LOW_ZEROS_MASK (dlow);
  if ((alow & dmask) != 0)
    return 0;

  if (dn == 1)
    {
      if (ABOVE_THRESHOLD (an, BMOD_1_TO_MOD_1_THRESHOLD))
	return mpn_mod_1 (ap, an, dlow) == 0;

      count_trailing_zeros (twos, dlow);
      dlow >>= twos;
      return mpn_modexact_1_odd (ap, an, dlow) == 0;
    }

  count_trailing_zeros (twos, dlow);
  if (dn == 2)
    {
      mp_limb_t  dsecond = dp[1];
      if (dsecond <= dmask)
	{
	  dlow = (dlow >> twos) | (dsecond << (GMP_NUMB_BITS-twos));
	  ASSERT_LIMB (dlow);
	  return MPN_MOD_OR_MODEXACT_1_ODD (ap, an, dlow) == 0;
	}
    }

  /* Should we compute Q = A * D^(-1) mod B^k,
                       R = A - Q * D  mod B^k
     here, for some small values of k?  Then check if R = 0 (mod B^k).  */

  /* We could also compute A' = A mod T and D' = D mod P, for some
     P = 3 * 5 * 7 * 11 ..., and then check if any prime factor from P
     dividing D' also divides A'.  */

  TMP_MARK;

  TMP_ALLOC_LIMBS_2 (rp, an + 1,
		     qp, an - dn + 1); /* FIXME: Could we avoid this? */

  if (twos != 0)
    {
      tp = TMP_ALLOC_LIMBS (dn);
      ASSERT_NOCARRY (mpn_rshift (tp, dp, dn, twos));
      dp = tp;

      ASSERT_NOCARRY (mpn_rshift (rp, ap, an, twos));
    }
  else
    {
      MPN_COPY (rp, ap, an);
    }
  if (rp[an - 1] >= dp[dn - 1])
    {
      rp[an] = 0;
      an++;
    }
  else if (an == dn)
    {
      TMP_FREE;
      return 0;
    }

  ASSERT (an > dn);		/* requirement of functions below */

  if (BELOW_THRESHOLD (dn, DC_BDIV_QR_THRESHOLD) ||
      BELOW_THRESHOLD (an - dn, DC_BDIV_QR_THRESHOLD))
    {
      binvert_limb (di, dp[0]);
      mpn_sbpi1_bdiv_qr (qp, rp, an, dp, dn, -di);
      rp += an - dn;
    }
  else if (BELOW_THRESHOLD (dn, MU_BDIV_QR_THRESHOLD))
    {
      binvert_limb (di, dp[0]);
      mpn_dcpi1_bdiv_qr (qp, rp, an, dp, dn, -di);
      rp += an - dn;
    }
  else
    {
      tp = TMP_ALLOC_LIMBS (mpn_mu_bdiv_qr_itch (an, dn));
      mpn_mu_bdiv_qr (qp, rp, rp, an, dp, dn, tp);
    }

  /* test for {rp,dn} zero or non-zero */
  i = 0;
  do
    {
      if (rp[i] != 0)
	{
	  TMP_FREE;
	  return 0;
	}
    }
  while (++i < dn);

  TMP_FREE;
  return 1;
}
