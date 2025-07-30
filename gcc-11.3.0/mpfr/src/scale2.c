/* mpfr_scale2 -- multiply a double float by 2^exp

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

#include <float.h> /* for DBL_EPSILON */
#include "mpfr-impl.h"

/* Note: we could use the ldexp function, but since we want not to depend on
   math.h, we write our own implementation. */

/* multiplies 1/2 <= d <= 1 by 2^exp */
double
mpfr_scale2 (double d, int exp)
{
#if _GMP_IEEE_FLOATS
  {
    union ieee_double_extract x;

    if (MPFR_UNLIKELY (d == 1.0))
      {
        d = 0.5;
        exp ++;
      }

    /* now 1/2 <= d < 1 */

    /* infinities and zeroes have already been checked */
    MPFR_ASSERTD (-1073 <= exp && exp <= 1025);

    x.d = d;
    if (MPFR_UNLIKELY (exp < -1021)) /* subnormal case */
      {
        x.s.exp += exp + 52;
        x.d *= DBL_EPSILON;
      }
    else /* normalized case */
      {
        x.s.exp += exp;
      }
    return x.d;
  }
#else /* _GMP_IEEE_FLOATS */
  {
    double factor;

    /* An overflow may occurs (example: 0.5*2^1024) */
    if (d < 1.0)
      {
        d += d;
        exp--;
      }
    /* Now 1.0 <= d < 2.0 */

    if (exp < 0)
      {
        factor = 0.5;
        exp = -exp;
      }
    else
      {
        factor = 2.0;
      }
    while (exp != 0)
      {
        if ((exp & 1) != 0)
          d *= factor;
        exp >>= 1;
        factor *= factor;
      }
    return d;
  }
#endif
}
