/* mpn_redc_n.  Set rp[] <- up[]/R^n mod mp[].  Clobber up[].
   mp[] is n limbs; up[] is 2n limbs, the inverse ip[] is n limbs.

   THIS IS AN INTERNAL FUNCTION WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH THIS FUNCTION THROUGH DOCUMENTED INTERFACES.

Copyright 2009, 2012 Free Software Foundation, Inc.

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

/*
  TODO

  * We assume mpn_mulmod_bnm1 is always faster than plain mpn_mul_n (or a
    future mpn_mulhi) for the range we will be called.  Follow up that
    assumption.

  * Decrease scratch usage.

  * Consider removing the residue canonicalisation.
*/

void
mpn_redc_n (mp_ptr rp, mp_ptr up, mp_srcptr mp, mp_size_t n, mp_srcptr ip)
{
  mp_ptr xp, yp, scratch;
  mp_limb_t cy;
  mp_size_t rn;
  TMP_DECL;
  TMP_MARK;

  ASSERT (n > 8);

  rn = mpn_mulmod_bnm1_next_size (n);

  scratch = TMP_ALLOC_LIMBS (n + rn + mpn_mulmod_bnm1_itch (rn, n, n));

  xp = scratch;
  mpn_mullo_n (xp, up, ip, n);

  yp = scratch + n;
  mpn_mulmod_bnm1 (yp, rn, xp, n, mp, n, scratch + n + rn);

  ASSERT_ALWAYS (2 * n > rn);				/* could handle this */

  cy = mpn_sub_n (yp + rn, yp, up, 2*n - rn);		/* undo wrap around */
  MPN_DECR_U (yp + 2*n - rn, rn, cy);

  cy = mpn_sub_n (rp, up + n, yp + n, n);
  if (cy != 0)
    mpn_add_n (rp, rp, mp, n);

  TMP_FREE;
}
