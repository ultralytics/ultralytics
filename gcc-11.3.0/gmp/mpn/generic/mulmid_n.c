/* mpn_mulmid_n -- balanced middle product

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


void
mpn_mulmid_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  ASSERT (n >= 1);
  ASSERT (! MPN_OVERLAP_P (rp, n + 2, ap, 2*n - 1));
  ASSERT (! MPN_OVERLAP_P (rp, n + 2, bp, n));

  if (n < MULMID_TOOM42_THRESHOLD)
    {
      mpn_mulmid_basecase (rp, ap, 2*n - 1, bp, n);
    }
  else
    {
      mp_ptr scratch;
      TMP_DECL;
      TMP_MARK;
      scratch = TMP_ALLOC_LIMBS (mpn_toom42_mulmid_itch (n));
      mpn_toom42_mulmid (rp, ap, bp, n, scratch);
      TMP_FREE;
    }
}
