/* mpn_sbpi1_div_q -- Schoolbook division using the Möller-Granlund 3/2
   division algorithm.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2007, 2009 Free Software Foundation, Inc.

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

mp_limb_t
mpn_sbpi1_div_q (mp_ptr qp,
		 mp_ptr np, mp_size_t nn,
		 mp_srcptr dp, mp_size_t dn,
		 mp_limb_t dinv)
{
  mp_limb_t qh;
  mp_size_t qn, i;
  mp_limb_t n1, n0;
  mp_limb_t d1, d0;
  mp_limb_t cy, cy1;
  mp_limb_t q;
  mp_limb_t flag;

  mp_size_t dn_orig = dn;
  mp_srcptr dp_orig = dp;
  mp_ptr np_orig = np;

  ASSERT (dn > 2);
  ASSERT (nn >= dn);
  ASSERT ((dp[dn-1] & GMP_NUMB_HIGHBIT) != 0);

  np += nn;

  qn = nn - dn;
  if (qn + 1 < dn)
    {
      dp += dn - (qn + 1);
      dn = qn + 1;
    }

  qh = mpn_cmp (np - dn, dp, dn) >= 0;
  if (qh != 0)
    mpn_sub_n (np - dn, np - dn, dp, dn);

  qp += qn;

  dn -= 2;			/* offset dn by 2 for main division loops,
				   saving two iterations in mpn_submul_1.  */
  d1 = dp[dn + 1];
  d0 = dp[dn + 0];

  np -= 2;

  n1 = np[1];

  for (i = qn - (dn + 2); i >= 0; i--)
    {
      np--;
      if (UNLIKELY (n1 == d1) && np[1] == d0)
	{
	  q = GMP_NUMB_MASK;
	  mpn_submul_1 (np - dn, dp, dn + 2, q);
	  n1 = np[1];		/* update n1, last loop's value will now be invalid */
	}
      else
	{
	  udiv_qr_3by2 (q, n1, n0, n1, np[1], np[0], d1, d0, dinv);

	  cy = mpn_submul_1 (np - dn, dp, dn, q);

	  cy1 = n0 < cy;
	  n0 = (n0 - cy) & GMP_NUMB_MASK;
	  cy = n1 < cy1;
	  n1 -= cy1;
	  np[0] = n0;

	  if (UNLIKELY (cy != 0))
	    {
	      n1 += d1 + mpn_add_n (np - dn, np - dn, dp, dn + 1);
	      q--;
	    }
	}

      *--qp = q;
    }

  flag = ~CNST_LIMB(0);

  if (dn >= 0)
    {
      for (i = dn; i > 0; i--)
	{
	  np--;
	  if (UNLIKELY (n1 >= (d1 & flag)))
	    {
	      q = GMP_NUMB_MASK;
	      cy = mpn_submul_1 (np - dn, dp, dn + 2, q);

	      if (UNLIKELY (n1 != cy))
		{
		  if (n1 < (cy & flag))
		    {
		      q--;
		      mpn_add_n (np - dn, np - dn, dp, dn + 2);
		    }
		  else
		    flag = 0;
		}
	      n1 = np[1];
	    }
	  else
	    {
	      udiv_qr_3by2 (q, n1, n0, n1, np[1], np[0], d1, d0, dinv);

	      cy = mpn_submul_1 (np - dn, dp, dn, q);

	      cy1 = n0 < cy;
	      n0 = (n0 - cy) & GMP_NUMB_MASK;
	      cy = n1 < cy1;
	      n1 -= cy1;
	      np[0] = n0;

	      if (UNLIKELY (cy != 0))
		{
		  n1 += d1 + mpn_add_n (np - dn, np - dn, dp, dn + 1);
		  q--;
		}
	    }

	  *--qp = q;

	  /* Truncate operands.  */
	  dn--;
	  dp++;
	}

      np--;
      if (UNLIKELY (n1 >= (d1 & flag)))
	{
	  q = GMP_NUMB_MASK;
	  cy = mpn_submul_1 (np, dp, 2, q);

	  if (UNLIKELY (n1 != cy))
	    {
	      if (n1 < (cy & flag))
		{
		  q--;
		  add_ssaaaa (np[1], np[0], np[1], np[0], dp[1], dp[0]);
		}
	      else
		flag = 0;
	    }
	  n1 = np[1];
	}
      else
	{
	  udiv_qr_3by2 (q, n1, n0, n1, np[1], np[0], d1, d0, dinv);

	  np[0] = n0;
	  np[1] = n1;
	}

      *--qp = q;
    }
  ASSERT_ALWAYS (np[1] == n1);
  np += 2;


  dn = dn_orig;
  if (UNLIKELY (n1 < (dn & flag)))
    {
      mp_limb_t q, x;

      /* The quotient may be too large if the remainder is small.  Recompute
	 for above ignored operand parts, until the remainder spills.

	 FIXME: The quality of this code isn't the same as the code above.
	 1. We don't compute things in an optimal order, high-to-low, in order
	    to terminate as quickly as possible.
	 2. We mess with pointers and sizes, adding and subtracting and
	    adjusting to get things right.  It surely could be streamlined.
	 3. The only termination criteria are that we determine that the
	    quotient needs to be adjusted, or that we have recomputed
	    everything.  We should stop when the remainder is so large
	    that no additional subtracting could make it spill.
	 4. If nothing else, we should not do two loops of submul_1 over the
	    data, instead handle both the triangularization and chopping at
	    once.  */

      x = n1;

      if (dn > 2)
	{
	  /* Compensate for triangularization.  */
	  mp_limb_t y;

	  dp = dp_orig;
	  if (qn + 1 < dn)
	    {
	      dp += dn - (qn + 1);
	      dn = qn + 1;
	    }

	  y = np[-2];

	  for (i = dn - 3; i >= 0; i--)
	    {
	      q = qp[i];
	      cy = mpn_submul_1 (np - (dn - i), dp, dn - i - 2, q);

	      if (y < cy)
		{
		  if (x == 0)
		    {
		      cy = mpn_sub_1 (qp, qp, qn, 1);
		      ASSERT_ALWAYS (cy == 0);
		      return qh - cy;
		    }
		  x--;
		}
	      y -= cy;
	    }
	  np[-2] = y;
	}

      dn = dn_orig;
      if (qn + 1 < dn)
	{
	  /* Compensate for ignored dividend and divisor tails.  */

	  dp = dp_orig;
	  np = np_orig;

	  if (qh != 0)
	    {
	      cy = mpn_sub_n (np + qn, np + qn, dp, dn - (qn + 1));
	      if (cy != 0)
		{
		  if (x == 0)
		    {
		      if (qn != 0)
			cy = mpn_sub_1 (qp, qp, qn, 1);
		      return qh - cy;
		    }
		  x--;
		}
	    }

	  if (qn == 0)
	    return qh;

	  for (i = dn - qn - 2; i >= 0; i--)
	    {
	      cy = mpn_submul_1 (np + i, qp, qn, dp[i]);
	      cy = mpn_sub_1 (np + qn + i, np + qn + i, dn - qn - i - 1, cy);
	      if (cy != 0)
		{
		  if (x == 0)
		    {
		      cy = mpn_sub_1 (qp, qp, qn, 1);
		      return qh;
		    }
		  x--;
		}
	    }
	}
    }

  return qh;
}
