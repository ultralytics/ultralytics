/* mpn_mul_1 -- Multiply a limb vector with a single limb and store the
   product in a second limb vector.

Copyright 1991-1994, 1996, 2000-2002 Free Software Foundation, Inc.

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


#if GMP_NAIL_BITS == 0

mp_limb_t
mpn_mul_1 (mp_ptr rp, mp_srcptr up, mp_size_t n, mp_limb_t vl)
{
  mp_limb_t ul, cl, hpl, lpl;

  ASSERT (n >= 1);
  ASSERT (MPN_SAME_OR_INCR_P (rp, up, n));

  cl = 0;
  do
    {
      ul = *up++;
      umul_ppmm (hpl, lpl, ul, vl);

      lpl += cl;
      cl = (lpl < cl) + hpl;

      *rp++ = lpl;
    }
  while (--n != 0);

  return cl;
}

#endif

#if GMP_NAIL_BITS >= 1

mp_limb_t
mpn_mul_1 (mp_ptr rp, mp_srcptr up, mp_size_t n, mp_limb_t vl)
{
  mp_limb_t shifted_vl, ul, lpl, hpl, prev_hpl, xw, cl, xl;

  ASSERT (n >= 1);
  ASSERT (MPN_SAME_OR_INCR_P (rp, up, n));
  ASSERT_MPN (up, n);
  ASSERT_LIMB (vl);

  shifted_vl = vl << GMP_NAIL_BITS;
  cl = 0;
  prev_hpl = 0;
  do
    {
      ul = *up++;

      umul_ppmm (hpl, lpl, ul, shifted_vl);
      lpl >>= GMP_NAIL_BITS;
      xw = prev_hpl + lpl + cl;
      cl = xw >> GMP_NUMB_BITS;
      xl = xw & GMP_NUMB_MASK;
      *rp++ = xl;
      prev_hpl = hpl;
    }
  while (--n != 0);

  return prev_hpl + cl;
}

#endif
