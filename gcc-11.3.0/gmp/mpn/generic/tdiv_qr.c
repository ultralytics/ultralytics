/* mpn_tdiv_qr -- Divide the numerator (np,nn) by the denominator (dp,dn) and
   write the nn-dn+1 quotient limbs at qp and the dn remainder limbs at rp.  If
   qxn is non-zero, generate that many fraction limbs and append them after the
   other quotient limbs, and update the remainder accordingly.  The input
   operands are unaffected.

   Preconditions:
   1. The most significant limb of of the divisor must be non-zero.
   2. nn >= dn, even if qxn is non-zero.  (??? relax this ???)

   The time complexity of this is O(qn*qn+M(dn,qn)), where M(m,n) is the time
   complexity of multiplication.

Copyright 1997, 2000-2002, 2005, 2009 Free Software Foundation, Inc.

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


void
mpn_tdiv_qr (mp_ptr qp, mp_ptr rp, mp_size_t qxn,
	     mp_srcptr np, mp_size_t nn, mp_srcptr dp, mp_size_t dn)
{
  ASSERT_ALWAYS (qxn == 0);

  ASSERT (nn >= 0);
  ASSERT (dn >= 0);
  ASSERT (dn == 0 || dp[dn - 1] != 0);
  ASSERT (! MPN_OVERLAP_P (qp, nn - dn + 1 + qxn, np, nn));
  ASSERT (! MPN_OVERLAP_P (qp, nn - dn + 1 + qxn, dp, dn));

  switch (dn)
    {
    case 0:
      DIVIDE_BY_ZERO;

    case 1:
      {
	rp[0] = mpn_divrem_1 (qp, (mp_size_t) 0, np, nn, dp[0]);
	return;
      }

    case 2:
      {
	mp_ptr n2p, d2p;
	mp_limb_t qhl, cy;
	TMP_DECL;
	TMP_MARK;
	if ((dp[1] & GMP_NUMB_HIGHBIT) == 0)
	  {
	    int cnt;
	    mp_limb_t dtmp[2];
	    count_leading_zeros (cnt, dp[1]);
	    cnt -= GMP_NAIL_BITS;
	    d2p = dtmp;
	    d2p[1] = (dp[1] << cnt) | (dp[0] >> (GMP_NUMB_BITS - cnt));
	    d2p[0] = (dp[0] << cnt) & GMP_NUMB_MASK;
	    n2p = TMP_ALLOC_LIMBS (nn + 1);
	    cy = mpn_lshift (n2p, np, nn, cnt);
	    n2p[nn] = cy;
	    qhl = mpn_divrem_2 (qp, 0L, n2p, nn + (cy != 0), d2p);
	    if (cy == 0)
	      qp[nn - 2] = qhl;	/* always store nn-2+1 quotient limbs */
	    rp[0] = (n2p[0] >> cnt)
	      | ((n2p[1] << (GMP_NUMB_BITS - cnt)) & GMP_NUMB_MASK);
	    rp[1] = (n2p[1] >> cnt);
	  }
	else
	  {
	    d2p = (mp_ptr) dp;
	    n2p = TMP_ALLOC_LIMBS (nn);
	    MPN_COPY (n2p, np, nn);
	    qhl = mpn_divrem_2 (qp, 0L, n2p, nn, d2p);
	    qp[nn - 2] = qhl;	/* always store nn-2+1 quotient limbs */
	    rp[0] = n2p[0];
	    rp[1] = n2p[1];
	  }
	TMP_FREE;
	return;
      }

    default:
      {
	int adjust;
	gmp_pi1_t dinv;
	TMP_DECL;
	TMP_MARK;
	adjust = np[nn - 1] >= dp[dn - 1];	/* conservative tests for quotient size */
	if (nn + adjust >= 2 * dn)
	  {
	    mp_ptr n2p, d2p;
	    mp_limb_t cy;
	    int cnt;

	    qp[nn - dn] = 0;			  /* zero high quotient limb */
	    if ((dp[dn - 1] & GMP_NUMB_HIGHBIT) == 0) /* normalize divisor */
	      {
		count_leading_zeros (cnt, dp[dn - 1]);
		cnt -= GMP_NAIL_BITS;
		d2p = TMP_ALLOC_LIMBS (dn);
		mpn_lshift (d2p, dp, dn, cnt);
		n2p = TMP_ALLOC_LIMBS (nn + 1);
		cy = mpn_lshift (n2p, np, nn, cnt);
		n2p[nn] = cy;
		nn += adjust;
	      }
	    else
	      {
		cnt = 0;
		d2p = (mp_ptr) dp;
		n2p = TMP_ALLOC_LIMBS (nn + 1);
		MPN_COPY (n2p, np, nn);
		n2p[nn] = 0;
		nn += adjust;
	      }

	    invert_pi1 (dinv, d2p[dn - 1], d2p[dn - 2]);
	    if (BELOW_THRESHOLD (dn, DC_DIV_QR_THRESHOLD))
	      mpn_sbpi1_div_qr (qp, n2p, nn, d2p, dn, dinv.inv32);
	    else if (BELOW_THRESHOLD (dn, MUPI_DIV_QR_THRESHOLD) ||   /* fast condition */
		     BELOW_THRESHOLD (nn, 2 * MU_DIV_QR_THRESHOLD) || /* fast condition */
		     (double) (2 * (MU_DIV_QR_THRESHOLD - MUPI_DIV_QR_THRESHOLD)) * dn /* slow... */
		     + (double) MUPI_DIV_QR_THRESHOLD * nn > (double) dn * nn)    /* ...condition */
	      mpn_dcpi1_div_qr (qp, n2p, nn, d2p, dn, &dinv);
	    else
	      {
		mp_size_t itch = mpn_mu_div_qr_itch (nn, dn, 0);
		mp_ptr scratch = TMP_ALLOC_LIMBS (itch);
		mpn_mu_div_qr (qp, rp, n2p, nn, d2p, dn, scratch);
		n2p = rp;
	      }

	    if (cnt != 0)
	      mpn_rshift (rp, n2p, dn, cnt);
	    else
	      MPN_COPY (rp, n2p, dn);
	    TMP_FREE;
	    return;
	  }

	/* When we come here, the numerator/partial remainder is less
	   than twice the size of the denominator.  */

	  {
	    /* Problem:

	       Divide a numerator N with nn limbs by a denominator D with dn
	       limbs forming a quotient of qn=nn-dn+1 limbs.  When qn is small
	       compared to dn, conventional division algorithms perform poorly.
	       We want an algorithm that has an expected running time that is
	       dependent only on qn.

	       Algorithm (very informally stated):

	       1) Divide the 2 x qn most significant limbs from the numerator
		  by the qn most significant limbs from the denominator.  Call
		  the result qest.  This is either the correct quotient, but
		  might be 1 or 2 too large.  Compute the remainder from the
		  division.  (This step is implemented by an mpn_divrem call.)

	       2) Is the most significant limb from the remainder < p, where p
		  is the product of the most significant limb from the quotient
		  and the next(d)?  (Next(d) denotes the next ignored limb from
		  the denominator.)  If it is, decrement qest, and adjust the
		  remainder accordingly.

	       3) Is the remainder >= qest?  If it is, qest is the desired
		  quotient.  The algorithm terminates.

	       4) Subtract qest x next(d) from the remainder.  If there is
		  borrow out, decrement qest, and adjust the remainder
		  accordingly.

	       5) Skip one word from the denominator (i.e., let next(d) denote
		  the next less significant limb.  */

	    mp_size_t qn;
	    mp_ptr n2p, d2p;
	    mp_ptr tp;
	    mp_limb_t cy;
	    mp_size_t in, rn;
	    mp_limb_t quotient_too_large;
	    unsigned int cnt;

	    qn = nn - dn;
	    qp[qn] = 0;				/* zero high quotient limb */
	    qn += adjust;			/* qn cannot become bigger */

	    if (qn == 0)
	      {
		MPN_COPY (rp, np, dn);
		TMP_FREE;
		return;
	      }

	    in = dn - qn;		/* (at least partially) ignored # of limbs in ops */
	    /* Normalize denominator by shifting it to the left such that its
	       most significant bit is set.  Then shift the numerator the same
	       amount, to mathematically preserve quotient.  */
	    if ((dp[dn - 1] & GMP_NUMB_HIGHBIT) == 0)
	      {
		count_leading_zeros (cnt, dp[dn - 1]);
		cnt -= GMP_NAIL_BITS;

		d2p = TMP_ALLOC_LIMBS (qn);
		mpn_lshift (d2p, dp + in, qn, cnt);
		d2p[0] |= dp[in - 1] >> (GMP_NUMB_BITS - cnt);

		n2p = TMP_ALLOC_LIMBS (2 * qn + 1);
		cy = mpn_lshift (n2p, np + nn - 2 * qn, 2 * qn, cnt);
		if (adjust)
		  {
		    n2p[2 * qn] = cy;
		    n2p++;
		  }
		else
		  {
		    n2p[0] |= np[nn - 2 * qn - 1] >> (GMP_NUMB_BITS - cnt);
		  }
	      }
	    else
	      {
		cnt = 0;
		d2p = (mp_ptr) dp + in;

		n2p = TMP_ALLOC_LIMBS (2 * qn + 1);
		MPN_COPY (n2p, np + nn - 2 * qn, 2 * qn);
		if (adjust)
		  {
		    n2p[2 * qn] = 0;
		    n2p++;
		  }
	      }

	    /* Get an approximate quotient using the extracted operands.  */
	    if (qn == 1)
	      {
		mp_limb_t q0, r0;
		udiv_qrnnd (q0, r0, n2p[1], n2p[0] << GMP_NAIL_BITS, d2p[0] << GMP_NAIL_BITS);
		n2p[0] = r0 >> GMP_NAIL_BITS;
		qp[0] = q0;
	      }
	    else if (qn == 2)
	      mpn_divrem_2 (qp, 0L, n2p, 4L, d2p); /* FIXME: obsolete function */
	    else
	      {
		invert_pi1 (dinv, d2p[qn - 1], d2p[qn - 2]);
		if (BELOW_THRESHOLD (qn, DC_DIV_QR_THRESHOLD))
		  mpn_sbpi1_div_qr (qp, n2p, 2 * qn, d2p, qn, dinv.inv32);
		else if (BELOW_THRESHOLD (qn, MU_DIV_QR_THRESHOLD))
		  mpn_dcpi1_div_qr (qp, n2p, 2 * qn, d2p, qn, &dinv);
		else
		  {
		    mp_size_t itch = mpn_mu_div_qr_itch (2 * qn, qn, 0);
		    mp_ptr scratch = TMP_ALLOC_LIMBS (itch);
		    mp_ptr r2p = rp;
		    if (np == r2p)	/* If N and R share space, put ... */
		      r2p += nn - qn;	/* intermediate remainder at N's upper end. */
		    mpn_mu_div_qr (qp, r2p, n2p, 2 * qn, d2p, qn, scratch);
		    MPN_COPY (n2p, r2p, qn);
		  }
	      }

	    rn = qn;
	    /* Multiply the first ignored divisor limb by the most significant
	       quotient limb.  If that product is > the partial remainder's
	       most significant limb, we know the quotient is too large.  This
	       test quickly catches most cases where the quotient is too large;
	       it catches all cases where the quotient is 2 too large.  */
	    {
	      mp_limb_t dl, x;
	      mp_limb_t h, dummy;

	      if (in - 2 < 0)
		dl = 0;
	      else
		dl = dp[in - 2];

#if GMP_NAIL_BITS == 0
	      x = (dp[in - 1] << cnt) | ((dl >> 1) >> ((~cnt) % GMP_LIMB_BITS));
#else
	      x = (dp[in - 1] << cnt) & GMP_NUMB_MASK;
	      if (cnt != 0)
		x |= dl >> (GMP_NUMB_BITS - cnt);
#endif
	      umul_ppmm (h, dummy, x, qp[qn - 1] << GMP_NAIL_BITS);

	      if (n2p[qn - 1] < h)
		{
		  mp_limb_t cy;

		  mpn_decr_u (qp, (mp_limb_t) 1);
		  cy = mpn_add_n (n2p, n2p, d2p, qn);
		  if (cy)
		    {
		      /* The partial remainder is safely large.  */
		      n2p[qn] = cy;
		      ++rn;
		    }
		}
	    }

	    quotient_too_large = 0;
	    if (cnt != 0)
	      {
		mp_limb_t cy1, cy2;

		/* Append partially used numerator limb to partial remainder.  */
		cy1 = mpn_lshift (n2p, n2p, rn, GMP_NUMB_BITS - cnt);
		n2p[0] |= np[in - 1] & (GMP_NUMB_MASK >> cnt);

		/* Update partial remainder with partially used divisor limb.  */
		cy2 = mpn_submul_1 (n2p, qp, qn, dp[in - 1] & (GMP_NUMB_MASK >> cnt));
		if (qn != rn)
		  {
		    ASSERT_ALWAYS (n2p[qn] >= cy2);
		    n2p[qn] -= cy2;
		  }
		else
		  {
		    n2p[qn] = cy1 - cy2; /* & GMP_NUMB_MASK; */

		    quotient_too_large = (cy1 < cy2);
		    ++rn;
		  }
		--in;
	      }
	    /* True: partial remainder now is neutral, i.e., it is not shifted up.  */

	    tp = TMP_ALLOC_LIMBS (dn);

	    if (in < qn)
	      {
		if (in == 0)
		  {
		    MPN_COPY (rp, n2p, rn);
		    ASSERT_ALWAYS (rn == dn);
		    goto foo;
		  }
		mpn_mul (tp, qp, qn, dp, in);
	      }
	    else
	      mpn_mul (tp, dp, in, qp, qn);

	    cy = mpn_sub (n2p, n2p, rn, tp + in, qn);
	    MPN_COPY (rp + in, n2p, dn - in);
	    quotient_too_large |= cy;
	    cy = mpn_sub_n (rp, np, tp, in);
	    cy = mpn_sub_1 (rp + in, rp + in, rn, cy);
	    quotient_too_large |= cy;
	  foo:
	    if (quotient_too_large)
	      {
		mpn_decr_u (qp, (mp_limb_t) 1);
		mpn_add_n (rp, rp, dp, dn);
	      }
	  }
	TMP_FREE;
	return;
      }
    }
}
