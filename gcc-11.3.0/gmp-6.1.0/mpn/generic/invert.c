/* invert.c -- Compute floor((B^{2n}-1)/U) - B^n.

   Contributed to the GNU project by Marco Bodrato.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright (C) 2007, 2009, 2010, 2012, 2014-2015 Free Software Foundation, Inc.

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
mpn_invert (mp_ptr ip, mp_srcptr dp, mp_size_t n, mp_ptr scratch)
{
  ASSERT (n > 0);
  ASSERT (dp[n-1] & GMP_NUMB_HIGHBIT);
  ASSERT (! MPN_OVERLAP_P (ip, n, dp, n));
  ASSERT (! MPN_OVERLAP_P (ip, n, scratch, mpn_invertappr_itch(n)));
  ASSERT (! MPN_OVERLAP_P (dp, n, scratch, mpn_invertappr_itch(n)));

  if (n == 1)
    invert_limb (*ip, *dp);
  else if (BELOW_THRESHOLD (n, INV_APPR_THRESHOLD))
    {
	/* Maximum scratch needed by this branch: 2*n */
	mp_size_t i;
	mp_ptr xp;

	xp = scratch;				/* 2 * n limbs */
	/* n > 1 here */
	i = n;
	do
	  xp[--i] = GMP_NUMB_MAX;
	while (i);
	mpn_com (xp + n, dp, n);
	if (n == 2) {
	  mpn_divrem_2 (ip, 0, xp, 4, dp);
	} else {
	  gmp_pi1_t inv;
	  invert_pi1 (inv, dp[n-1], dp[n-2]);
	  /* FIXME: should we use dcpi1_div_q, for big sizes? */
	  mpn_sbpi1_div_q (ip, xp, 2 * n, dp, n, inv.inv32);
	}
    }
  else { /* Use approximated inverse; correct the result if needed. */
      mp_limb_t e; /* The possible error in the approximate inverse */

      ASSERT ( mpn_invert_itch (n) >= mpn_invertappr_itch (n) );
      e = mpn_ni_invertappr (ip, dp, n, scratch);

      if (UNLIKELY (e)) { /* Assume the error can only be "0" (no error) or "1". */
	/* Code to detect and correct the "off by one" approximation. */
	mpn_mul_n (scratch, ip, dp, n);
	e = mpn_add_n (scratch, scratch, dp, n); /* FIXME: we only need e.*/
	if (LIKELY(e)) /* The high part can not give a carry by itself. */
	  e = mpn_add_nc (scratch + n, scratch + n, dp, n, e); /* FIXME:e */
	/* If the value was wrong (no carry), correct it (increment). */
	e ^= CNST_LIMB (1);
	MPN_INCR_U (ip, n, e);
      }
  }
}
