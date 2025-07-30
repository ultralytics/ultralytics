/* mpn_sbpi1_div_qr -- Schoolbook division using the Möller-Granlund 3/2
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
mpn_sbpi1_div_qr (mp_ptr qp,
		  mp_ptr np, mp_size_t nn,
		  mp_srcptr dp, mp_size_t dn,
		  mp_limb_t dinv)
{
  mp_limb_t qh;
  mp_size_t i;
  mp_limb_t n1, n0;
  mp_limb_t d1, d0;
  mp_limb_t cy, cy1;
  mp_limb_t q;

  ASSERT (dn > 2);
  ASSERT (nn >= dn);
  ASSERT ((dp[dn-1] & GMP_NUMB_HIGHBIT) != 0);

  np += nn;

  qh = mpn_cmp (np - dn, dp, dn) >= 0;
  if (qh != 0)
    mpn_sub_n (np - dn, np - dn, dp, dn);

  qp += nn - dn;

  dn -= 2;			/* offset dn by 2 for main division loops,
				   saving two iterations in mpn_submul_1.  */
  d1 = dp[dn + 1];
  d0 = dp[dn + 0];

  np -= 2;

  n1 = np[1];

  for (i = nn - (dn + 2); i > 0; i--)
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
	  n1 = (n1 - cy1) & GMP_NUMB_MASK;
	  np[0] = n0;

	  if (UNLIKELY (cy != 0))
	    {
	      n1 += d1 + mpn_add_n (np - dn, np - dn, dp, dn + 1);
	      q--;
	    }
	}

      *--qp = q;
    }
  np[1] = n1;

  return qh;
}
