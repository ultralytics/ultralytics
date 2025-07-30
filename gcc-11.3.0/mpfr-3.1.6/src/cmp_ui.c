/* mpfr_cmp_ui_2exp -- compare a floating-point number with an unsigned
machine integer multiplied by a power of 2

Copyright 1999, 2001-2004, 2006-2017 Free Software Foundation, Inc.
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

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

/* returns a positive value if b > i*2^f,
           a negative value if b < i*2^f,
           zero if b = i*2^f.
   b must not be NaN
*/

int
mpfr_cmp_ui_2exp (mpfr_srcptr b, unsigned long int i, mpfr_exp_t f)
{
  if (MPFR_UNLIKELY( MPFR_IS_SINGULAR(b) ))
    {
      if (MPFR_IS_NAN (b))
        {
          MPFR_SET_ERANGE ();
          return 0;
        }
      else if (MPFR_IS_INF(b))
        return MPFR_INT_SIGN (b);
      else /* since b cannot be NaN, b=0 here */
        return i != 0 ? -1 : 0;
    }

  if (MPFR_IS_NEG (b))
    return -1;
  /* now b > 0 */
  else if (MPFR_UNLIKELY(i == 0))
    return 1;
  else /* b > 0, i > 0 */
    {
      mpfr_exp_t e;
      int k;
      mp_size_t bn;
      mp_limb_t c, *bp;

      /* i must be representable in a mp_limb_t */
      MPFR_ASSERTN(i == (mp_limb_t) i);

      e = MPFR_GET_EXP (b); /* 2^(e-1) <= b < 2^e */
      if (e <= f)
        return -1;
      if (f < MPFR_EMAX_MAX - GMP_NUMB_BITS &&
          e > f + GMP_NUMB_BITS)
        return 1;

      /* now f < e <= f + GMP_NUMB_BITS */
      c = (mp_limb_t) i;
      count_leading_zeros(k, c);
      if ((int) (e - f) > GMP_NUMB_BITS - k)
        return 1;
      if ((int) (e - f) < GMP_NUMB_BITS - k)
        return -1;

      /* now b and i*2^f have the same exponent */
      c <<= k;
      bn = (MPFR_PREC(b) - 1) / GMP_NUMB_BITS;
      bp = MPFR_MANT(b);
      if (bp[bn] > c)
        return 1;
      if (bp[bn] < c)
        return -1;

      /* most significant limbs agree, check remaining limbs from b */
      while (bn > 0)
        if (bp[--bn] != 0)
          return 1;
      return 0;
    }
}

#undef mpfr_cmp_ui
int
mpfr_cmp_ui (mpfr_srcptr b, unsigned long int i)
{
  return mpfr_cmp_ui_2exp (b, i, 0);
}
