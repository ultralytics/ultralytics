/* mpfr_dim -- positive difference

Copyright 2001-2002, 2004, 2006-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include "mpfr-impl.h"

/* dim (x,y) is defined as:

    x-y if x >  y
    +0    if x <= y
*/

int
mpfr_dim (mpfr_ptr z, mpfr_srcptr x, mpfr_srcptr y, mpfr_rnd_t rnd_mode)
{
  if (MPFR_IS_NAN(x) || MPFR_IS_NAN(y))
    {
      MPFR_SET_NAN(z);
      MPFR_RET_NAN;
    }

  if (mpfr_cmp (x,y) > 0)
    return mpfr_sub (z, x, y, rnd_mode);
  else
    {
      MPFR_SET_ZERO(z);
      MPFR_SET_POS(z);
      MPFR_RET(0);
    }
}
