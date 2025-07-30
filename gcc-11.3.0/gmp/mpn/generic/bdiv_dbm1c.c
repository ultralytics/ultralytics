/* mpn_bdiv_dbm1c -- divide an mpn number by a divisor of B-1, where B is the
   limb base.  The dbm1c moniker means "Divisor of B Minus 1 with Carry".

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2008, 2009 Free Software Foundation, Inc.

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
mpn_bdiv_dbm1c (mp_ptr qp, mp_srcptr ap, mp_size_t n, mp_limb_t bd, mp_limb_t h)
{
  mp_limb_t a, p0, p1, cy;
  mp_size_t i;

  for (i = 0; i < n; i++)
    {
      a = ap[i];
      umul_ppmm (p1, p0, a, bd << GMP_NAIL_BITS);
      p0 >>= GMP_NAIL_BITS;
      cy = h < p0;
      h = (h - p0) & GMP_NUMB_MASK;
      qp[i] = h;
      h = h - p1 - cy;
    }

  return h;
}
