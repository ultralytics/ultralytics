/* mpn_rshift -- Shift right low level for Cray vector processors.

Copyright (C) 2000 Free Software Foundation, Inc.

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

#include <intrinsics.h>
#include "gmp.h"
#include "gmp-impl.h"

mp_limb_t
mpn_rshift (mp_ptr wp, mp_srcptr up, mp_size_t n, unsigned int cnt)
{
  unsigned sh_1, sh_2;
  mp_size_t i;
  mp_limb_t retval;

  sh_1 = cnt;
  sh_2 = GMP_LIMB_BITS - sh_1;
  retval = up[0] << sh_2;

#pragma _CRI ivdep
  for (i = 0; i < n - 1; i++)
    {
#if 1
      wp[i] = (up[i] >> sh_1) | (up[i + 1] << sh_2);
#else
      /* This is the recommended way, but at least on SV1 it is slower.  */
      wp[i] = _dshiftr (up[i + 1], up[i], sh_1);
#endif
    }

  wp[n - 1] = up[n - 1] >> sh_1;
  return retval;
}
