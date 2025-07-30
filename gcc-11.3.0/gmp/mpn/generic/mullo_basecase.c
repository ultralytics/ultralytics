/* mpn_mullo_basecase -- Internal routine to multiply two natural
   numbers of length n and return the low part.

   THIS IS AN INTERNAL FUNCTION WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH THIS FUNCTION THROUGH DOCUMENTED INTERFACES.


Copyright (C) 2000, 2002, 2004, 2015 Free Software Foundation, Inc.

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

/* FIXME: Should optionally use mpn_mul_2/mpn_addmul_2.  */

#ifndef MULLO_VARIANT
#define MULLO_VARIANT 2
#endif


#if MULLO_VARIANT == 1
void
mpn_mullo_basecase (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
{
  mp_size_t i;

  mpn_mul_1 (rp, up, n, vp[0]);

  for (i = n - 1; i > 0; i--)
    {
      vp++;
      rp++;
      mpn_addmul_1 (rp, up, i, vp[0]);
    }
}
#endif


#if MULLO_VARIANT == 2
void
mpn_mullo_basecase (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
{
  mp_limb_t h;

  h = up[0] * vp[n - 1];

  if (n != 1)
    {
      mp_size_t i;
      mp_limb_t v0;

      v0 = *vp++;
      h += up[n - 1] * v0 + mpn_mul_1 (rp, up, n - 1, v0);
      rp++;

      for (i = n - 2; i > 0; i--)
	{
	  v0 = *vp++;
	  h += up[i] * v0 + mpn_addmul_1 (rp, up, i, v0);
	  rp++;
	}
    }

  rp[0] = h;
}
#endif
