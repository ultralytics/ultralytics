/* mpfr_setmin -- minimum representable floating-point number (raw version)

Copyright 2002-2004, 2006-2017 Free Software Foundation, Inc.
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

/* Note: the flags are not cleared and the current sign is kept. */

void
mpfr_setmin (mpfr_ptr x, mpfr_exp_t e)
{
  mp_size_t xn;
  mp_limb_t *xp;

  MPFR_SET_EXP (x, e);
  xn = (MPFR_PREC(x) - 1) / GMP_NUMB_BITS;
  xp = MPFR_MANT(x);
  xp[xn] = MPFR_LIMB_HIGHBIT;
  MPN_ZERO(xp, xn);
}
