/* mpz_ui_sub -- Subtract an unsigned one-word integer and an mpz_t.

Copyright 2002, 2004 Free Software Foundation, Inc.

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

void
mpz_ui_sub (mpz_ptr w, unsigned long int uval, mpz_srcptr v)
{
  mp_ptr vp, wp;
  mp_size_t vn, wn;
  mp_limb_t cy;

#if BITS_PER_ULONG > GMP_NUMB_BITS  /* avoid warnings about shift amount */
  if (uval > GMP_NUMB_MAX)
    {
      mpz_t u;
      mp_limb_t ul[2];
      PTR(u) = ul;
      ul[0] = uval & GMP_NUMB_MASK;
      ul[1] = uval >> GMP_NUMB_BITS;
      SIZ(u) = 2;
      mpz_sub (w, u, v);
      return;
    }
#endif

  vp = PTR(v);
  vn = SIZ(v);

  wp = PTR(w);

  if (vn > 1)
    {
      wp = MPZ_REALLOC (w, vn);
      vp = PTR(v);
      mpn_sub_1 (wp, vp, vn, (mp_limb_t) uval);
      wn = -(vn - (wp[vn - 1] == 0));
    }
  else if (vn == 1)
    {
      if (uval >= vp[0])
	{
	  wp[0] = uval - vp[0];
	  wn = wp[0] != 0;
	}
      else
	{
	  wp[0] = vp[0] - uval;
	  wn = -1;
	}
    }
  else if (vn == 0)
    {
      wp[0] = uval;
      wn = uval != 0;
    }
  else /* (vn < 0) */
    {
      vn = -vn;
      wp = MPZ_REALLOC (w, vn + 1);
      vp = PTR(v);
      cy = mpn_add_1 (wp, vp, vn, (mp_limb_t) uval);
      wp[vn] = cy;
      wn = vn + (cy != 0);
    }

  SIZ(w) = wn;
}
