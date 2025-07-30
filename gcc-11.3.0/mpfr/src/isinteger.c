/* mpfr_integer_p -- test if a mpfr variable is integer.

Copyright 2001-2004, 2006-2017 Free Software Foundation, Inc.
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
mpfr_integer_p (mpfr_srcptr x)
{
  mpfr_exp_t expo;
  mpfr_prec_t prec;
  mp_size_t xn;
  mp_limb_t *xp;

  if (MPFR_UNLIKELY( MPFR_IS_SINGULAR(x)))
    return (MPFR_IS_ZERO(x));

  expo = MPFR_GET_EXP (x);
  if (expo <= 0)
    return 0;

  prec = MPFR_PREC(x);
  if ((mpfr_uexp_t) expo >= (mpfr_uexp_t) prec)
    return 1;

  /* 0 < expo < prec */

  xn =  (mp_size_t) ((prec - 1) / GMP_NUMB_BITS);  /* index of last limb */
  xn -= (mp_size_t) (expo / GMP_NUMB_BITS);
  /* now the index of the last limb containing bits of the fractional part */

  xp = MPFR_MANT(x);
  MPFR_ASSERTN(xn >= 0);
  if (xp[xn] << (expo % GMP_NUMB_BITS) != 0)
    return 0;
  while (--xn >= 0)
    if (xp[xn] != 0)
      return 0;
  return 1;
}
