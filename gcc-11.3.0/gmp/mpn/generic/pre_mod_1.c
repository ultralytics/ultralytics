/* mpn_preinv_mod_1 (up, un, d, dinv) -- Divide (UP,,UN) by the normalized D.
   DINV should be 2^(2*GMP_LIMB_BITS) / D - 2^GMP_LIMB_BITS.
   Return the single-limb remainder.

Copyright 1991, 1993, 1994, 2000-2002, 2004, 2005 Free Software Foundation,
Inc.

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


/* This function used to be documented, but is now considered obsolete.  It
   continues to exist for binary compatibility, even when not required
   internally.  */

mp_limb_t
mpn_preinv_mod_1 (mp_srcptr up, mp_size_t un, mp_limb_t d, mp_limb_t dinv)
{
  mp_size_t i;
  mp_limb_t n0, r;

  ASSERT (un >= 1);
  ASSERT (d & GMP_LIMB_HIGHBIT);

  r = up[un - 1];
  if (r >= d)
    r -= d;

  for (i = un - 2; i >= 0; i--)
    {
      n0 = up[i];
      udiv_rnnd_preinv (r, r, n0, d, dinv);
    }
  return r;
}
