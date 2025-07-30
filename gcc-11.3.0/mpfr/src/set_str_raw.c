/* mpfr_set_str_binary -- set a floating-point number from a binary string

Copyright 1999-2017 Free Software Foundation, Inc.
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

/* Currently the number should be of the form +/- xxxx.xxxxxxEyy, with
   decimal exponent. The mantissa of x is supposed to be large enough
   to hold all the bits of str. */

void
mpfr_set_str_binary (mpfr_ptr x, const char *str)
{
  int has_sign;
  int res;

  if (*str == 'N')
    {
      MPFR_SET_NAN(x);
      __gmpfr_flags |= MPFR_FLAGS_NAN;
      return;
    }

  has_sign = *str == '-' || *str == '+';
  if (str[has_sign] == 'I')
    {
      MPFR_SET_INF(x);
      if (*str == '-')
        MPFR_SET_NEG(x);
      else
        MPFR_SET_POS(x);
      return;
    }

  res = mpfr_strtofr (x, str, 0, 2, MPFR_RNDZ);
  MPFR_ASSERTN (res == 0);
}
