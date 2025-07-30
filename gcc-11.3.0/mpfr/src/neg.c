/* mpfr_neg -- change the sign of a floating-point number

Copyright 1999-2001, 2004, 2006-2017 Free Software Foundation, Inc.
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

int
mpfr_neg (mpfr_ptr a, mpfr_srcptr b, mpfr_rnd_t rnd_mode)
{
  if (MPFR_UNLIKELY(a != b))
    return mpfr_set4 (a, b, rnd_mode, -MPFR_SIGN(b));
  else if (MPFR_UNLIKELY(MPFR_IS_NAN (b)))
    {
      MPFR_RET_NAN;
    }
  else
    {
      MPFR_CHANGE_SIGN(a);
      MPFR_RET(0);
    }
}
