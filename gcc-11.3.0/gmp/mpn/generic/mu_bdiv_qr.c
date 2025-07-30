/* mpn_mu_bdiv_qr(qp,rp,np,nn,dp,dn,tp) -- Compute {np,nn} / {dp,dn} mod B^qn,
   where qn = nn-dn, storing the result in {qp,qn}.  Overlap allowed between Q
   and N; all other overlap disallowed.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2005-2007, 2009, 2010, 2012 Free Software Foundation, Inc.

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


/*
   The idea of the algorithm used herein is to compute a smaller inverted value
   than used in the standard Barrett algorithm, and thus save time in the
   Newton iterations, and pay just a small price when using the inverted value
   for developing quotient bits.  This algorithm was presented at ICMS 2006.
*/

#include "gmp.h"
#include "gmp-impl.h"


/* N = {np,nn}
   D = {dp,dn}

   Requirements: N >= D
		 D >= 1
		 D odd
		 dn >= 2
		 nn >= 2
		 scratch space as determined by mpn_mu_bdiv_qr_itch(nn,dn).

   Write quotient to Q = {qp,nn-dn}.

   FIXME: When iterating, perhaps do the small step before loop, not after.
   FIXME: Try to avoid the scalar divisions when computing inverse size.
   FIXME: Trim allocation for (qn > dn) case, 3*dn might be possible.  In
	  particular, when dn==in, tp and rp could use the same space.
*/
mp_limb_t
mpn_mu_bdiv_qr (mp_ptr qp,
		mp_ptr rp,
		mp_srcptr np, mp_size_t nn,
		mp_srcptr dp, mp_size_t dn,
		mp_ptr scratch)
{
  mp_size_t qn;
  mp_size_t in;
  mp_limb_t cy, c0;
  mp_size_t tn, wn;

  qn = nn - dn;

  ASSERT (dn >= 2);
  ASSERT (qn >= 2);

  if (qn > dn)
    {
      mp_size_t b;

      /* |_______________________|   dividend
			|________|   divisor  */

#define ip           scratch		/* in */
#define tp           (scratch + in)	/* dn+in or next_size(dn) or rest >= binvert_itch(in) */
#define scratch_out  (scratch + in + tn)/* mulmod_bnm1_itch(next_size(dn)) */

      /* Compute an inverse size that is a nice partition of the quotient.  */
      b = (qn - 1) / dn + 1;	/* ceil(qn/dn), number of blocks */
      in = (qn - 1) / b + 1;	/* ceil(qn/b) = ceil(qn / ceil(qn/dn)) */

      /* Some notes on allocation:

	 When in = dn, R dies when mpn_mullo returns, if in < dn the low in
	 limbs of R dies at that point.  We could save memory by letting T live
	 just under R, and let the upper part of T expand into R. These changes
	 should reduce itch to perhaps 3dn.
       */

      mpn_binvert (ip, dp, in, tp);

      MPN_COPY (rp, np, dn);
      np += dn;
      cy = 0;

      while (qn > in)
	{
	  mpn_mullo_n (qp, rp, ip, in);

	  if (BELOW_THRESHOLD (in, MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD))
	    mpn_mul (tp, dp, dn, qp, in);	/* mulhi, need tp[dn+in-1...in] */
	  else
	    {
	      tn = mpn_mulmod_bnm1_next_size (dn);
	      mpn_mulmod_bnm1 (tp, tn, dp, dn, qp, in, scratch_out);
	      wn = dn + in - tn;		/* number of wrapped limbs */
	      if (wn > 0)
		{
		  c0 = mpn_sub_n (tp + tn, tp, rp, wn);
		  mpn_decr_u (tp + wn, c0);
		}
	    }

	  qp += in;
	  qn -= in;

	  if (dn != in)
	    {
	      /* Subtract tp[dn-1...in] from partial remainder.  */
	      cy += mpn_sub_n (rp, rp + in, tp + in, dn - in);
	      if (cy == 2)
		{
		  mpn_incr_u (tp + dn, 1);
		  cy = 1;
		}
	    }
	  /* Subtract tp[dn+in-1...dn] from dividend.  */
	  cy = mpn_sub_nc (rp + dn - in, np, tp + dn, in, cy);
	  np += in;
	}

      /* Generate last qn limbs.  */
      mpn_mullo_n (qp, rp, ip, qn);

      if (BELOW_THRESHOLD (qn, MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD))
	mpn_mul (tp, dp, dn, qp, qn);		/* mulhi, need tp[qn+in-1...in] */
      else
	{
	  tn = mpn_mulmod_bnm1_next_size (dn);
	  mpn_mulmod_bnm1 (tp, tn, dp, dn, qp, qn, scratch_out);
	  wn = dn + qn - tn;			/* number of wrapped limbs */
	  if (wn > 0)
	    {
	      c0 = mpn_sub_n (tp + tn, tp, rp, wn);
	      mpn_decr_u (tp + wn, c0);
	    }
	}

      if (dn != qn)
	{
	  cy += mpn_sub_n (rp, rp + qn, tp + qn, dn - qn);
	  if (cy == 2)
	    {
	      mpn_incr_u (tp + dn, 1);
	      cy = 1;
	    }
	}
      return mpn_sub_nc (rp + dn - qn, np, tp + dn, qn, cy);

#undef ip
#undef tp
#undef scratch_out
    }
  else
    {
      /* |_______________________|   dividend
		|________________|   divisor  */

#define ip           scratch		/* in */
#define tp           (scratch + in)	/* dn+in or next_size(dn) or rest >= binvert_itch(in) */
#define scratch_out  (scratch + in + tn)/* mulmod_bnm1_itch(next_size(dn)) */

      /* Compute half-sized inverse.  */
      in = qn - (qn >> 1);

      mpn_binvert (ip, dp, in, tp);

      mpn_mullo_n (qp, np, ip, in);		/* low `in' quotient limbs */

      if (BELOW_THRESHOLD (in, MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD))
	mpn_mul (tp, dp, dn, qp, in);		/* mulhigh */
      else
	{
	  tn = mpn_mulmod_bnm1_next_size (dn);
	  mpn_mulmod_bnm1 (tp, tn, dp, dn, qp, in, scratch_out);
	  wn = dn + in - tn;			/* number of wrapped limbs */
	  if (wn > 0)
	    {
	      c0 = mpn_sub_n (tp + tn, tp, np, wn);
	      mpn_decr_u (tp + wn, c0);
	    }
	}

      qp += in;
      qn -= in;

      cy = mpn_sub_n (rp, np + in, tp + in, dn);
      mpn_mullo_n (qp, rp, ip, qn);		/* high qn quotient limbs */

      if (BELOW_THRESHOLD (qn, MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD))
	mpn_mul (tp, dp, dn, qp, qn);		/* mulhigh */
      else
	{
	  tn = mpn_mulmod_bnm1_next_size (dn);
	  mpn_mulmod_bnm1 (tp, tn, dp, dn, qp, qn, scratch_out);
	  wn = dn + qn - tn;			/* number of wrapped limbs */
	  if (wn > 0)
	    {
	      c0 = mpn_sub_n (tp + tn, tp, rp, wn);
	      mpn_decr_u (tp + wn, c0);
	    }
	}

      cy += mpn_sub_n (rp, rp + qn, tp + qn, dn - qn);
      if (cy == 2)
	{
	  mpn_incr_u (tp + dn, 1);
	  cy = 1;
	}
      return mpn_sub_nc (rp + dn - qn, np + dn + in, tp + dn, qn, cy);

#undef ip
#undef tp
#undef scratch_out
    }
}

mp_size_t
mpn_mu_bdiv_qr_itch (mp_size_t nn, mp_size_t dn)
{
  mp_size_t qn, in, tn, itch_binvert, itch_out, itches;
  mp_size_t b;

  ASSERT_ALWAYS (DC_BDIV_Q_THRESHOLD < MU_BDIV_Q_THRESHOLD);

  qn = nn - dn;

  if (qn > dn)
    {
      b = (qn - 1) / dn + 1;	/* ceil(qn/dn), number of blocks */
      in = (qn - 1) / b + 1;	/* ceil(qn/b) = ceil(qn / ceil(qn/dn)) */
    }
  else
    {
      in = qn - (qn >> 1);
    }

  if (BELOW_THRESHOLD (in, MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD))
    {
      tn = dn + in;
      itch_out = 0;
    }
  else
    {
      tn = mpn_mulmod_bnm1_next_size (dn);
      itch_out = mpn_mulmod_bnm1_itch (tn, dn, in);
    }

  itch_binvert = mpn_binvert_itch (in);
  itches = tn + itch_out;
  return in + MAX (itches, itch_binvert);
}
