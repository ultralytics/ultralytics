/* mpn_divrem -- Divide natural numbers, producing both remainder and
   quotient.  This is now just a middle layer calling mpn_tdiv_qr.

Copyright 1993-1997, 1999-2002, 2005 Free Software Foundation, Inc.

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
mpn_divrem (mp_ptr qp, mp_size_t qxn,
	    mp_ptr np, mp_size_t nn,
	    mp_srcptr dp, mp_size_t dn)
{
  ASSERT (qxn >= 0);
  ASSERT (nn >= dn);
  ASSERT (dn >= 1);
  ASSERT (dp[dn-1] & GMP_NUMB_HIGHBIT);
  ASSERT (! MPN_OVERLAP_P (np, nn, dp, dn));
  ASSERT (! MPN_OVERLAP_P (qp, nn-dn+qxn, np, nn) || qp==np+dn+qxn);
  ASSERT (! MPN_OVERLAP_P (qp, nn-dn+qxn, dp, dn));
  ASSERT_MPN (np, nn);
  ASSERT_MPN (dp, dn);

  if (dn == 1)
    {
      mp_limb_t ret;
      mp_ptr q2p;
      mp_size_t qn;
      TMP_DECL;

      TMP_MARK;
      q2p = TMP_ALLOC_LIMBS (nn + qxn);

      np[0] = mpn_divrem_1 (q2p, qxn, np, nn, dp[0]);
      qn = nn + qxn - 1;
      MPN_COPY (qp, q2p, qn);
      ret = q2p[qn];

      TMP_FREE;
      return ret;
    }
  else if (dn == 2)
    {
      return mpn_divrem_2 (qp, qxn, np, nn, dp);
    }
  else
    {
      mp_ptr rp, q2p;
      mp_limb_t qhl;
      mp_size_t qn;
      TMP_DECL;

      TMP_MARK;
      if (UNLIKELY (qxn != 0))
	{
	  mp_ptr n2p;
	  n2p = TMP_ALLOC_LIMBS (nn + qxn);
	  MPN_ZERO (n2p, qxn);
	  MPN_COPY (n2p + qxn, np, nn);
	  q2p = TMP_ALLOC_LIMBS (nn - dn + qxn + 1);
	  rp = TMP_ALLOC_LIMBS (dn);
	  mpn_tdiv_qr (q2p, rp, 0L, n2p, nn + qxn, dp, dn);
	  MPN_COPY (np, rp, dn);
	  qn = nn - dn + qxn;
	  MPN_COPY (qp, q2p, qn);
	  qhl = q2p[qn];
	}
      else
	{
	  q2p = TMP_ALLOC_LIMBS (nn - dn + 1);
	  rp = TMP_ALLOC_LIMBS (dn);
	  mpn_tdiv_qr (q2p, rp, 0L, np, nn, dp, dn);
	  MPN_COPY (np, rp, dn);	/* overwrite np area with remainder */
	  qn = nn - dn;
	  MPN_COPY (qp, q2p, qn);
	  qhl = q2p[qn];
	}
      TMP_FREE;
      return qhl;
    }
}
