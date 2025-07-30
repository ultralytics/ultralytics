/* mpfr_powerof2_raw -- test whether a floating-point number is a power of 2

Copyright 2002-2017 Free Software Foundation, Inc.
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

/* This is an internal function and one assumes that x is a non-special
 * number (more precisely, only its significand is considered and this
 * function can be used even if the exponent field of x has not been
 * initialized). It returns 1 (true) if |x| is a power of 2, else 0.
 */

int
mpfr_powerof2_raw (mpfr_srcptr x)
{
  /* This is an internal function, and we may call it with some
     wrong numbers (ie good mantissa but wrong flags or exp)
     So we don't want to test if it is a pure FP.
     MPFR_ASSERTN(MPFR_IS_PURE_FP(x)); */
  return mpfr_powerof2_raw2 (MPFR_MANT(x), MPFR_LIMB_SIZE(x));
}

int
mpfr_powerof2_raw2 (const mp_limb_t *xp, mp_size_t xn)
{
  if (xp[--xn] != MPFR_LIMB_HIGHBIT)
    return 0;
  while (xn > 0)
    if (xp[--xn] != 0)
      return 0;
  return 1;
}
