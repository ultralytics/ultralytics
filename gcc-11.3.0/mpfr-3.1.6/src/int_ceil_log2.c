/* __gmpfr_int_ceil_log2 -- Integer ceil of log2(x)

Copyright 2004-2017 Free Software Foundation, Inc.
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

#define MPFR_NEED_LONGLONG_H /* for count_leading_zeros */
#include "mpfr-impl.h"

int
__gmpfr_int_ceil_log2 (unsigned long n)
{
  if (MPFR_UNLIKELY (n == 1))
    return 0;
  else
    {
      int b;
      mp_limb_t limb;

      MPFR_ASSERTN (n > 1);
      limb = n - 1;
      MPFR_ASSERTN (limb == n - 1);
      count_leading_zeros (b, limb);
      return GMP_NUMB_BITS - b;
    }
}
