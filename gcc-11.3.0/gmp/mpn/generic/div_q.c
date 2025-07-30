/* mpn_div_q -- division for arbitrary size operands.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2009, 2010 Free Software Foundation, Inc.

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


/* Compute Q = N/D with truncation.
     N = {np,nn}
     D = {dp,dn}
     Q = {qp,nn-dn+1}
     T = {scratch,nn+1} is scratch space
   N and D are both untouched by the computation.
   N and T may overlap; pass the same space if N is irrelevant after the call,
   but note that tp needs an extra limb.

   Operand requirements:
     N >= D > 0
     dp[dn-1] != 0
     No overlap between the N, D, and Q areas.

   This division function does not clobber its input operands, since it is
   intended to support average-O(qn) division, and for that to be effective, it
   cannot put requirements on callers to copy a O(nn) operand.

   If a caller does not care about the value of {np,nn+1} after calling this
   function, it should pass np also for the scratch argument.  This function
   will then save some time and space by avoiding allocation and copying.
   (FIXME: Is this a good design?  We only really save any copying for
   already-normalised divisors, which should be rare.  It also prevents us from
   reasonably asking for all scratch space we need.)

   We write nn-dn+1 limbs for the quotient, but return void.  Why not return
   the most significant quotient limb?  Look at the 4 main code blocks below
   (consisting of an outer if-else where each arm contains an if-else). It is
   tricky for the first code block, since the mpn_*_div_q calls will typically
   generate all nn-dn+1 and return 0 or 1.  I don't see how to fix that unless
   we generate the most significant quotient limb here, before calling
   mpn_*_div_q, or put the quotient in a temporary area.  Since this is a
   critical division case (the SB sub-case in particular) copying is not a good
   idea.

   It might make sense to split the if-else parts of the (qn + FUDGE
   >= dn) blocks into separate functions, since we could promise quite
   different things to callers in these two cases.  The 'then' case
   benefits from np=scratch, and it could perhaps even tolerate qp=np,
   saving some headache for many callers.

   FIXME: Scratch allocation leaves a lot to be desired.  E.g., for the MU size
   operands, we do not reuse the huge scratch for adjustments.  This can be a
   serious waste of memory for the largest operands.
*/

/* FUDGE determines when to try getting an approximate quotient from the upper
   parts of the dividend and divisor, then adjust.  N.B. FUDGE must be >= 2
   for the code to be correct.  */
#define FUDGE 5			/* FIXME: tune this */

#define DC_DIV_Q_THRESHOLD      DC_DIVAPPR_Q_THRESHOLD
#define MU_DIV_Q_THRESHOLD      MU_DIVAPPR_Q_THRESHOLD
#define MUPI_DIV_Q_THRESHOLD  MUPI_DIVAPPR_Q_THRESHOLD
#ifndef MUPI_DIVAPPR_Q_THRESHOLD
#define MUPI_DIVAPPR_Q_THRESHOLD  MUPI_DIV_QR_THRESHOLD
#endif

void
mpn_div_q (mp_ptr qp,
	   mp_srcptr np, mp_size_t nn,
	   mp_srcptr dp, mp_size_t dn, mp_ptr scratch)
{
  mp_ptr new_dp, new_np, tp, rp;
  mp_limb_t cy, dh, qh;
  mp_size_t new_nn, qn;
  gmp_pi1_t dinv;
  int cnt;
  TMP_DECL;
  TMP_MARK;

  ASSERT (nn >= dn);
  ASSERT (dn > 0);
  ASSERT (dp[dn - 1] != 0);
  ASSERT (! MPN_OVERLAP_P (qp, nn - dn + 1, np, nn));
  ASSERT (! MPN_OVERLAP_P (qp, nn - dn + 1, dp, dn));
  ASSERT (MPN_SAME_OR_SEPARATE_P (np, scratch, nn));

  ASSERT_ALWAYS (FUDGE >= 2);

  if (dn == 1)
    {
      mpn_divrem_1 (qp, 0L, np, nn, dp[dn - 1]);
      return;
    }

  qn = nn - dn + 1;		/* Quotient size, high limb might be zero */

  if (qn + FUDGE >= dn)
    {
      /* |________________________|
                          |_______|  */
      new_np = scratch;

      dh = dp[dn - 1];
      if (LIKELY ((dh & GMP_NUMB_HIGHBIT) == 0))
	{
	  count_leading_zeros (cnt, dh);

	  cy = mpn_lshift (new_np, np, nn, cnt);
	  new_np[nn] = cy;
	  new_nn = nn + (cy != 0);

	  new_dp = TMP_ALLOC_LIMBS (dn);
	  mpn_lshift (new_dp, dp, dn, cnt);

	  if (dn == 2)
	    {
	      qh = mpn_divrem_2 (qp, 0L, new_np, new_nn, new_dp);
	    }
	  else if (BELOW_THRESHOLD (dn, DC_DIV_Q_THRESHOLD) ||
		   BELOW_THRESHOLD (new_nn - dn, DC_DIV_Q_THRESHOLD))
	    {
	      invert_pi1 (dinv, new_dp[dn - 1], new_dp[dn - 2]);
	      qh = mpn_sbpi1_div_q (qp, new_np, new_nn, new_dp, dn, dinv.inv32);
	    }
	  else if (BELOW_THRESHOLD (dn, MUPI_DIV_Q_THRESHOLD) ||   /* fast condition */
		   BELOW_THRESHOLD (nn, 2 * MU_DIV_Q_THRESHOLD) || /* fast condition */
		   (double) (2 * (MU_DIV_Q_THRESHOLD - MUPI_DIV_Q_THRESHOLD)) * dn /* slow... */
		   + (double) MUPI_DIV_Q_THRESHOLD * nn > (double) dn * nn)   /* ...condition */
	    {
	      invert_pi1 (dinv, new_dp[dn - 1], new_dp[dn - 2]);
	      qh = mpn_dcpi1_div_q (qp, new_np, new_nn, new_dp, dn, &dinv);
	    }
	  else
	    {
	      mp_size_t itch = mpn_mu_div_q_itch (new_nn, dn, 0);
	      mp_ptr scratch = TMP_ALLOC_LIMBS (itch);
	      qh = mpn_mu_div_q (qp, new_np, new_nn, new_dp, dn, scratch);
	    }
	  if (cy == 0)
	    qp[qn - 1] = qh;
	  else if (UNLIKELY (qh != 0))
	    {
	      /* This happens only when the quotient is close to B^n and
		 mpn_*_divappr_q returned B^n.  */
	      mp_size_t i, n;
	      n = new_nn - dn;
	      for (i = 0; i < n; i++)
		qp[i] = GMP_NUMB_MAX;
	      qh = 0;		/* currently ignored */
	    }
	}
      else  /* divisor is already normalised */
	{
	  if (new_np != np)
	    MPN_COPY (new_np, np, nn);

	  if (dn == 2)
	    {
	      qh = mpn_divrem_2 (qp, 0L, new_np, nn, dp);
	    }
	  else if (BELOW_THRESHOLD (dn, DC_DIV_Q_THRESHOLD) ||
		   BELOW_THRESHOLD (nn - dn, DC_DIV_Q_THRESHOLD))
	    {
	      invert_pi1 (dinv, dh, dp[dn - 2]);
	      qh = mpn_sbpi1_div_q (qp, new_np, nn, dp, dn, dinv.inv32);
	    }
	  else if (BELOW_THRESHOLD (dn, MUPI_DIV_Q_THRESHOLD) ||   /* fast condition */
		   BELOW_THRESHOLD (nn, 2 * MU_DIV_Q_THRESHOLD) || /* fast condition */
		   (double) (2 * (MU_DIV_Q_THRESHOLD - MUPI_DIV_Q_THRESHOLD)) * dn /* slow... */
		   + (double) MUPI_DIV_Q_THRESHOLD * nn > (double) dn * nn)   /* ...condition */
	    {
	      invert_pi1 (dinv, dh, dp[dn - 2]);
	      qh = mpn_dcpi1_div_q (qp, new_np, nn, dp, dn, &dinv);
	    }
	  else
	    {
	      mp_size_t itch = mpn_mu_div_q_itch (nn, dn, 0);
	      mp_ptr scratch = TMP_ALLOC_LIMBS (itch);
	      qh = mpn_mu_div_q (qp, np, nn, dp, dn, scratch);
	    }
	  qp[nn - dn] = qh;
	}
    }
  else
    {
      /* |________________________|
                |_________________|  */
      tp = TMP_ALLOC_LIMBS (qn + 1);

      new_np = scratch;
      new_nn = 2 * qn + 1;
      if (new_np == np)
	/* We need {np,nn} to remain untouched until the final adjustment, so
	   we need to allocate separate space for new_np.  */
	new_np = TMP_ALLOC_LIMBS (new_nn + 1);


      dh = dp[dn - 1];
      if (LIKELY ((dh & GMP_NUMB_HIGHBIT) == 0))
	{
	  count_leading_zeros (cnt, dh);

	  cy = mpn_lshift (new_np, np + nn - new_nn, new_nn, cnt);
	  new_np[new_nn] = cy;

	  new_nn += (cy != 0);

	  new_dp = TMP_ALLOC_LIMBS (qn + 1);
	  mpn_lshift (new_dp, dp + dn - (qn + 1), qn + 1, cnt);
	  new_dp[0] |= dp[dn - (qn + 1) - 1] >> (GMP_NUMB_BITS - cnt);

	  if (qn + 1 == 2)
	    {
	      qh = mpn_divrem_2 (tp, 0L, new_np, new_nn, new_dp);
	    }
	  else if (BELOW_THRESHOLD (qn, DC_DIVAPPR_Q_THRESHOLD - 1))
	    {
	      invert_pi1 (dinv, new_dp[qn], new_dp[qn - 1]);
	      qh = mpn_sbpi1_divappr_q (tp, new_np, new_nn, new_dp, qn + 1, dinv.inv32);
	    }
	  else if (BELOW_THRESHOLD (qn, MU_DIVAPPR_Q_THRESHOLD - 1))
	    {
	      invert_pi1 (dinv, new_dp[qn], new_dp[qn - 1]);
	      qh = mpn_dcpi1_divappr_q (tp, new_np, new_nn, new_dp, qn + 1, &dinv);
	    }
	  else
	    {
	      mp_size_t itch = mpn_mu_divappr_q_itch (new_nn, qn + 1, 0);
	      mp_ptr scratch = TMP_ALLOC_LIMBS (itch);
	      qh = mpn_mu_divappr_q (tp, new_np, new_nn, new_dp, qn + 1, scratch);
	    }
	  if (cy == 0)
	    tp[qn] = qh;
	  else if (UNLIKELY (qh != 0))
	    {
	      /* This happens only when the quotient is close to B^n and
		 mpn_*_divappr_q returned B^n.  */
	      mp_size_t i, n;
	      n = new_nn - (qn + 1);
	      for (i = 0; i < n; i++)
		tp[i] = GMP_NUMB_MAX;
	      qh = 0;		/* currently ignored */
	    }
	}
      else  /* divisor is already normalised */
	{
	  MPN_COPY (new_np, np + nn - new_nn, new_nn); /* pointless if MU will be used */

	  new_dp = (mp_ptr) dp + dn - (qn + 1);

	  if (qn == 2 - 1)
	    {
	      qh = mpn_divrem_2 (tp, 0L, new_np, new_nn, new_dp);
	    }
	  else if (BELOW_THRESHOLD (qn, DC_DIVAPPR_Q_THRESHOLD - 1))
	    {
	      invert_pi1 (dinv, dh, new_dp[qn - 1]);
	      qh = mpn_sbpi1_divappr_q (tp, new_np, new_nn, new_dp, qn + 1, dinv.inv32);
	    }
	  else if (BELOW_THRESHOLD (qn, MU_DIVAPPR_Q_THRESHOLD - 1))
	    {
	      invert_pi1 (dinv, dh, new_dp[qn - 1]);
	      qh = mpn_dcpi1_divappr_q (tp, new_np, new_nn, new_dp, qn + 1, &dinv);
	    }
	  else
	    {
	      mp_size_t itch = mpn_mu_divappr_q_itch (new_nn, qn + 1, 0);
	      mp_ptr scratch = TMP_ALLOC_LIMBS (itch);
	      qh = mpn_mu_divappr_q (tp, new_np, new_nn, new_dp, qn + 1, scratch);
	    }
	  tp[qn] = qh;
	}

      MPN_COPY (qp, tp + 1, qn);
      if (tp[0] <= 4)
        {
	  mp_size_t rn;

          rp = TMP_ALLOC_LIMBS (dn + qn);
          mpn_mul (rp, dp, dn, tp + 1, qn);
	  rn = dn + qn;
	  rn -= rp[rn - 1] == 0;

          if (rn > nn || mpn_cmp (np, rp, nn) < 0)
            mpn_decr_u (qp, 1);
        }
    }

  TMP_FREE;
}
