/* mpn_divexact(qp,np,nn,dp,dn,tp) -- Divide N = {np,nn} by D = {dp,dn} storing
   the result in Q = {qp,nn-dn+1} expecting no remainder.  Overlap allowed
   between Q and N; all other overlap disallowed.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2006, 2007, 2009 Free Software Foundation, Inc.

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

#if 1
void
mpn_divexact (mp_ptr qp,
	      mp_srcptr np, mp_size_t nn,
	      mp_srcptr dp, mp_size_t dn)
{
  unsigned shift;
  mp_size_t qn;
  mp_ptr tp;
  TMP_DECL;

  ASSERT (dn > 0);
  ASSERT (nn >= dn);
  ASSERT (dp[dn-1] > 0);

  while (dp[0] == 0)
    {
      ASSERT (np[0] == 0);
      dp++;
      np++;
      dn--;
      nn--;
    }

  if (dn == 1)
    {
      MPN_DIVREM_OR_DIVEXACT_1 (qp, np, nn, dp[0]);
      return;
    }

  TMP_MARK;

  qn = nn + 1 - dn;
  count_trailing_zeros (shift, dp[0]);

  if (shift > 0)
    {
      mp_ptr wp;
      mp_size_t ss;
      ss = (dn > qn) ? qn + 1 : dn;

      tp = TMP_ALLOC_LIMBS (ss);
      mpn_rshift (tp, dp, ss, shift);
      dp = tp;

      /* Since we have excluded dn == 1, we have nn > qn, and we need
	 to shift one limb beyond qn. */
      wp = TMP_ALLOC_LIMBS (qn + 1);
      mpn_rshift (wp, np, qn + 1, shift);
      np = wp;
    }

  if (dn > qn)
    dn = qn;

  tp = TMP_ALLOC_LIMBS (mpn_bdiv_q_itch (qn, dn));
  mpn_bdiv_q (qp, np, qn, dp, dn, tp);
  TMP_FREE;
}

#else

/* We use the Jebelean's bidirectional exact division algorithm.  This is
   somewhat naively implemented, with equal quotient parts done by 2-adic
   division and truncating division.  Since 2-adic division is faster, it
   should be used for a larger chunk.

   This code is horrendously ugly, in all sorts of ways.

   * It was hacked without much care or thought, but with a testing program.
   * It handles scratch space frivolously, and furthermore the itch function
     is broken.
   * Doesn't provide any measures to deal with mu_divappr_q's +3 error.  We
     have yet to provoke an error due to this, though.
   * Algorithm selection leaves a lot to be desired.  In particular, the choice
     between DC and MU isn't a point, but we treat it like one.
   * It makes the msb part 1 or 2 limbs larger than the lsb part, in spite of
     that the latter is faster.  We should at least reverse this, but perhaps
     we should make the lsb part considerably larger.  (How do we tune this?)
*/

mp_size_t
mpn_divexact_itch (mp_size_t nn, mp_size_t dn)
{
  return nn + dn;		/* FIXME this is not right */
}

void
mpn_divexact (mp_ptr qp,
	      mp_srcptr np, mp_size_t nn,
	      mp_srcptr dp, mp_size_t dn,
	      mp_ptr scratch)
{
  mp_size_t qn;
  mp_size_t nn0, qn0;
  mp_size_t nn1, qn1;
  mp_ptr tp;
  mp_limb_t qml;
  mp_limb_t qh;
  int cnt;
  mp_ptr xdp;
  mp_limb_t di;
  mp_limb_t cy;
  gmp_pi1_t dinv;
  TMP_DECL;

  TMP_MARK;

  qn = nn - dn + 1;

  /* For small divisors, and small quotients, don't use Jebelean's algorithm. */
  if (dn < DIVEXACT_JEB_THRESHOLD || qn < DIVEXACT_JEB_THRESHOLD)
    {
      tp = scratch;
      MPN_COPY (tp, np, qn);
      binvert_limb (di, dp[0]);  di = -di;
      dn = MIN (dn, qn);
      mpn_sbpi1_bdiv_q (qp, tp, qn, dp, dn, di);
      TMP_FREE;
      return;
    }

  qn0 = ((nn - dn) >> 1) + 1;	/* low quotient size */

  /* If quotient is much larger than the divisor, the bidirectional algorithm
     does not work as currently implemented.  Fall back to plain bdiv.  */
  if (qn0 > dn)
    {
      if (BELOW_THRESHOLD (dn, DC_BDIV_Q_THRESHOLD))
	{
	  tp = scratch;
	  MPN_COPY (tp, np, qn);
	  binvert_limb (di, dp[0]);  di = -di;
	  dn = MIN (dn, qn);
	  mpn_sbpi1_bdiv_q (qp, tp, qn, dp, dn, di);
	}
      else if (BELOW_THRESHOLD (dn, MU_BDIV_Q_THRESHOLD))
	{
	  tp = scratch;
	  MPN_COPY (tp, np, qn);
	  binvert_limb (di, dp[0]);  di = -di;
	  mpn_dcpi1_bdiv_q (qp, tp, qn, dp, dn, di);
	}
      else
	{
	  mpn_mu_bdiv_q (qp, np, qn, dp, dn, scratch);
	}
      TMP_FREE;
      return;
    }

  nn0 = qn0 + qn0;

  nn1 = nn0 - 1 + ((nn-dn) & 1);
  qn1 = qn0;
  if (LIKELY (qn0 != dn))
    {
      nn1 = nn1 + 1;
      qn1 = qn1 + 1;
      if (UNLIKELY (dp[dn - 1] == 1 && qn1 != dn))
	{
	  /* If the leading divisor limb == 1, i.e. has just one bit, we have
	     to include an extra limb in order to get the needed overlap.  */
	  /* FIXME: Now with the mu_divappr_q function, we should really need
	     more overlap. That indicates one of two things: (1) The test code
	     is not good. (2) We actually overlap too much by default.  */
	  nn1 = nn1 + 1;
	  qn1 = qn1 + 1;
	}
    }

  tp = TMP_ALLOC_LIMBS (nn1 + 1);

  count_leading_zeros (cnt, dp[dn - 1]);

  /* Normalize divisor, store into tmp area.  */
  if (cnt != 0)
    {
      xdp = TMP_ALLOC_LIMBS (qn1);
      mpn_lshift (xdp, dp + dn - qn1, qn1, cnt);
    }
  else
    {
      xdp = (mp_ptr) dp + dn - qn1;
    }

  /* Shift dividend according to the divisor normalization.  */
  /* FIXME: We compute too much here for XX_divappr_q, but these functions'
     interfaces want a pointer to the imaginative least significant limb, not
     to the least significant *used* limb.  Of course, we could leave nn1-qn1
     rubbish limbs in the low part, to save some time.  */
  if (cnt != 0)
    {
      cy = mpn_lshift (tp, np + nn - nn1, nn1, cnt);
      if (cy != 0)
	{
	  tp[nn1] = cy;
	  nn1++;
	}
    }
  else
    {
      /* FIXME: This copy is not needed for mpn_mu_divappr_q, except when the
	 mpn_sub_n right before is executed.  */
      MPN_COPY (tp, np + nn - nn1, nn1);
    }

  invert_pi1 (dinv, xdp[qn1 - 1], xdp[qn1 - 2]);
  if (BELOW_THRESHOLD (qn1, DC_DIVAPPR_Q_THRESHOLD))
    {
      qp[qn0 - 1 + nn1 - qn1] = mpn_sbpi1_divappr_q (qp + qn0 - 1, tp, nn1, xdp, qn1, dinv.inv32);
    }
  else if (BELOW_THRESHOLD (qn1, MU_DIVAPPR_Q_THRESHOLD))
    {
      qp[qn0 - 1 + nn1 - qn1] = mpn_dcpi1_divappr_q (qp + qn0 - 1, tp, nn1, xdp, qn1, &dinv);
    }
  else
    {
      /* FIXME: mpn_mu_divappr_q doesn't handle qh != 0.  Work around it with a
	 conditional subtraction here.  */
      qh = mpn_cmp (tp + nn1 - qn1, xdp, qn1) >= 0;
      if (qh)
	mpn_sub_n (tp + nn1 - qn1, tp + nn1 - qn1, xdp, qn1);
      mpn_mu_divappr_q (qp + qn0 - 1, tp, nn1, xdp, qn1, scratch);
      qp[qn0 - 1 + nn1 - qn1] = qh;
    }
  qml = qp[qn0 - 1];

  binvert_limb (di, dp[0]);  di = -di;

  if (BELOW_THRESHOLD (qn0, DC_BDIV_Q_THRESHOLD))
    {
      MPN_COPY (tp, np, qn0);
      mpn_sbpi1_bdiv_q (qp, tp, qn0, dp, qn0, di);
    }
  else if (BELOW_THRESHOLD (qn0, MU_BDIV_Q_THRESHOLD))
    {
      MPN_COPY (tp, np, qn0);
      mpn_dcpi1_bdiv_q (qp, tp, qn0, dp, qn0, di);
    }
  else
    {
      mpn_mu_bdiv_q (qp, np, qn0, dp, qn0, scratch);
    }

  if (qml < qp[qn0 - 1])
    mpn_decr_u (qp + qn0, 1);

  TMP_FREE;
}
#endif
