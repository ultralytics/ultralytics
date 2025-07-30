/* mpn_mu_div_q.

   Contributed to the GNU project by Torbjorn Granlund and Marco Bodrato.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2005-2007, 2009, 2010, 2013 Free Software Foundation, Inc.

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

/*
  Things to work on:

  1. This is a rudimentary implementation of mpn_mu_div_q.  The algorithm is
     probably close to optimal, except when mpn_mu_divappr_q fails.

  2. We used to fall back to mpn_mu_div_qr when we detect a possible
     mpn_mu_divappr_q rounding problem, now we multiply and compare.
     Unfortunately, since mpn_mu_divappr_q does not return the partial
     remainder, this also doesn't become optimal.  A mpn_mu_divappr_qr could
     solve that.

  3. The allocations done here should be made from the scratch area, which
     then would need to be amended.
*/

#include <stdlib.h>		/* for NULL */
#include "gmp.h"
#include "gmp-impl.h"


mp_limb_t
mpn_mu_div_q (mp_ptr qp,
	      mp_srcptr np, mp_size_t nn,
	      mp_srcptr dp, mp_size_t dn,
	      mp_ptr scratch)
{
  mp_ptr tp, rp;
  mp_size_t qn;
  mp_limb_t cy, qh;
  TMP_DECL;

  TMP_MARK;

  qn = nn - dn;

  tp = TMP_BALLOC_LIMBS (qn + 1);

  if (qn >= dn)			/* nn >= 2*dn + 1 */
    {
       /* |_______________________|   dividend
			 |________|   divisor  */

      rp = TMP_BALLOC_LIMBS (nn + 1);
      MPN_COPY (rp + 1, np, nn);
      rp[0] = 0;

      qh = mpn_cmp (rp + 1 + nn - dn, dp, dn) >= 0;
      if (qh != 0)
	mpn_sub_n (rp + 1 + nn - dn, rp + 1 + nn - dn, dp, dn);

      cy = mpn_mu_divappr_q (tp, rp, nn + 1, dp, dn, scratch);

      if (UNLIKELY (cy != 0))
	{
	  /* Since the partial remainder fed to mpn_preinv_mu_divappr_q was
	     canonically reduced, replace the returned value of B^(qn-dn)+eps
	     by the largest possible value.  */
	  mp_size_t i;
	  for (i = 0; i < qn + 1; i++)
	    tp[i] = GMP_NUMB_MAX;
	}

      /* The max error of mpn_mu_divappr_q is +4.  If the low quotient limb is
	 smaller than the max error, we cannot trust the quotient.  */
      if (tp[0] > 4)
	{
	  MPN_COPY (qp, tp + 1, qn);
	}
      else
	{
	  mp_limb_t cy;
	  mp_ptr pp;

	  pp = rp;
	  mpn_mul (pp, tp + 1, qn, dp, dn);

	  cy = (qh != 0) ? mpn_add_n (pp + qn, pp + qn, dp, dn) : 0;

	  if (cy || mpn_cmp (pp, np, nn) > 0) /* At most is wrong by one, no cycle. */
	    qh -= mpn_sub_1 (qp, tp + 1, qn, 1);
	  else /* Same as above */
	    MPN_COPY (qp, tp + 1, qn);
	}
    }
  else
    {
       /* |_______________________|   dividend
		 |________________|   divisor  */

      /* FIXME: When nn = 2dn-1, qn becomes dn-1, and the numerator size passed
	 here becomes 2dn, i.e., more than nn.  This shouldn't hurt, since only
	 the most significant dn-1 limbs will actually be read, but it is not
	 pretty.  */

      qh = mpn_mu_divappr_q (tp, np + nn - (2 * qn + 2), 2 * qn + 2,
			     dp + dn - (qn + 1), qn + 1, scratch);

      /* The max error of mpn_mu_divappr_q is +4, but we get an additional
         error from the divisor truncation.  */
      if (tp[0] > 6)
	{
	  MPN_COPY (qp, tp + 1, qn);
	}
      else
	{
	  mp_limb_t cy;

	  /* FIXME: a shorter product should be enough; we may use already
	     allocated space... */
	  rp = TMP_BALLOC_LIMBS (nn);
	  mpn_mul (rp, dp, dn, tp + 1, qn);

	  cy = (qh != 0) ? mpn_add_n (rp + qn, rp + qn, dp, dn) : 0;

	  if (cy || mpn_cmp (rp, np, nn) > 0) /* At most is wrong by one, no cycle. */
	    qh -= mpn_sub_1 (qp, tp + 1, qn, 1);
	  else /* Same as above */
	    MPN_COPY (qp, tp + 1, qn);
	}
    }

  TMP_FREE;
  return qh;
}

mp_size_t
mpn_mu_div_q_itch (mp_size_t nn, mp_size_t dn, int mua_k)
{
  mp_size_t qn;

  qn = nn - dn;
  if (qn >= dn)
    {
      return mpn_mu_divappr_q_itch (nn + 1, dn, mua_k);
    }
  else
    {
      return mpn_mu_divappr_q_itch (2 * qn + 2, qn + 1, mua_k);
    }
}
