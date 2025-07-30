/* mpn_sub_err1_n -- sub_n with one error term

   Contributed by David Harvey.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2011 Free Software Foundation, Inc.

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
  Computes:

  (1) {rp,n} := {up,n} - {vp,n} (just like mpn_sub_n) with incoming borrow cy,
  return value is borrow out.

  (2) Let c[i+1] = borrow from i-th limb subtraction (c[0] = cy).
  Computes c[1]*yp[n-1] + ... + c[n]*yp[0], stores two-limb result at ep.

  Requires n >= 1.

  None of the outputs may overlap each other or any of the inputs, except
  that {rp,n} may be equal to {up,n} or {vp,n}.
*/
mp_limb_t
mpn_sub_err1_n (mp_ptr rp, mp_srcptr up, mp_srcptr vp,
		mp_ptr ep, mp_srcptr yp,
                mp_size_t n, mp_limb_t cy)
{
  mp_limb_t el, eh, ul, vl, yl, zl, rl, sl, cy1, cy2;

  ASSERT (n >= 1);
  ASSERT (MPN_SAME_OR_SEPARATE_P (rp, up, n));
  ASSERT (MPN_SAME_OR_SEPARATE_P (rp, vp, n));
  ASSERT (! MPN_OVERLAP_P (rp, n, yp, n));
  ASSERT (! MPN_OVERLAP_P (ep, 2, up, n));
  ASSERT (! MPN_OVERLAP_P (ep, 2, vp, n));
  ASSERT (! MPN_OVERLAP_P (ep, 2, yp, n));
  ASSERT (! MPN_OVERLAP_P (ep, 2, rp, n));

  yp += n - 1;
  el = eh = 0;

  do
    {
      yl = *yp--;
      ul = *up++;
      vl = *vp++;

      /* ordinary sub_n */
      SUBC_LIMB (cy1, sl, ul, vl);
      SUBC_LIMB (cy2, rl, sl, cy);
      cy = cy1 | cy2;
      *rp++ = rl;

      /* update (eh:el) */
      zl = (-cy) & yl;
      el += zl;
      eh += el < zl;
    }
  while (--n);

#if GMP_NAIL_BITS != 0
  eh = (eh << GMP_NAIL_BITS) + (el >> GMP_NUMB_BITS);
  el &= GMP_NUMB_MASK;
#endif

  ep[0] = el;
  ep[1] = eh;

  return cy;
}
