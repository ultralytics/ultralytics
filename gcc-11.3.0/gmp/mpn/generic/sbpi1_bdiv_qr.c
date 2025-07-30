/* mpn_sbpi1_bdiv_qr -- schoolbook Hensel division with precomputed inverse,
   returning quotient and remainder.

   Contributed to the GNU project by Niels Möller.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL FUNCTIONS WITH MUTABLE INTERFACES.
   IT IS ONLY SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS
   ALMOST GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2006, 2009, 2011, 2012 Free Software Foundation, Inc.

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


/* Computes a binary quotient of size qn = nn - dn.
   Output:

      Q = N * D^{-1} mod B^qn,

      R = (N - Q * D) * B^(-qn)

   Stores the dn least significant limbs of R at {np + nn - dn, dn},
   and returns the borrow from the subtraction N - Q*D.

   D must be odd. dinv is (-D)^-1 mod B. */

mp_limb_t
mpn_sbpi1_bdiv_qr (mp_ptr qp,
		   mp_ptr np, mp_size_t nn,
		   mp_srcptr dp, mp_size_t dn, mp_limb_t dinv)
{
  mp_size_t qn;
  mp_size_t i;
  mp_limb_t rh;
  mp_limb_t ql;

  ASSERT (dn > 0);
  ASSERT (nn > dn);
  ASSERT ((dp[0] & 1) != 0);
  /* FIXME: Add ASSERTs for allowable overlapping; i.e., that qp = np is OK,
     but some over N/Q overlaps will not work.  */

  qn = nn - dn;

  rh = 0;

  /* To complete the negation, this value is added to q. */
  ql = 1;
  while (qn > dn)
    {
      for (i = 0; i < dn; i++)
	{
	  mp_limb_t q;

	  q = dinv * np[i];
	  np[i] = mpn_addmul_1 (np + i, dp, dn, q);
	  qp[i] = ~q;
	}
      rh += mpn_add (np + dn, np + dn, qn, np, dn);
      ql = mpn_add_1 (qp, qp, dn, ql);

      qp += dn; qn -= dn;
      np += dn; nn -= dn;
    }

  for (i = 0; i < qn; i++)
    {
      mp_limb_t q;

      q = dinv * np[i];
      np[i] = mpn_addmul_1 (np + i, dp, dn, q);
      qp[i] = ~q;
    }

  rh += mpn_add_n (np + dn, np + dn, np, qn);
  ql = mpn_add_1 (qp, qp, qn, ql);

  if (UNLIKELY (ql > 0))
    {
      /* q == 0 */
      ASSERT (rh == 0);
      return 0;
    }
  else
    {
      mp_limb_t cy;

      cy = mpn_sub_n (np + qn, np + qn, dp, dn);
      ASSERT (cy >= rh);
      return cy - rh;
    }
}
