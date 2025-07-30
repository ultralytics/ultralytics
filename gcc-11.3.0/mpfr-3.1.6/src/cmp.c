/* mpfr_cmp -- compare two floating-point numbers

Copyright 1999, 2001, 2003-2017 Free Software Foundation, Inc.
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

/* returns 0 iff b = sign(s) * c
           a positive value iff b > sign(s) * c
           a negative value iff b < sign(s) * c
   returns 0 and sets erange flag if b and/or c is NaN.
*/

int
mpfr_cmp3 (mpfr_srcptr b, mpfr_srcptr c, int s)
{
  mpfr_exp_t be, ce;
  mp_size_t bn, cn;
  mp_limb_t *bp, *cp;

  s = MPFR_MULT_SIGN( s , MPFR_SIGN(c) );

  if (MPFR_ARE_SINGULAR(b, c))
    {
      if (MPFR_IS_NAN (b) || MPFR_IS_NAN (c))
        {
          MPFR_SET_ERANGE ();
          return 0;
        }
      else if (MPFR_IS_INF(b))
        {
          if (MPFR_IS_INF(c) && s == MPFR_SIGN(b) )
            return 0;
          else
            return MPFR_SIGN(b);
        }
      else if (MPFR_IS_INF(c))
        return -s;
      else if (MPFR_IS_ZERO(b))
        return MPFR_IS_ZERO(c) ? 0 : -s;
      else /* necessarily c=0 */
        return MPFR_SIGN(b);
    }
  /* b and c are real numbers */
  if (s != MPFR_SIGN(b))
    return MPFR_SIGN(b);

  /* now signs are equal */

  be = MPFR_GET_EXP (b);
  ce = MPFR_GET_EXP (c);
  if (be > ce)
    return s;
  if (be < ce)
    return -s;

  /* both signs and exponents are equal */

  bn = (MPFR_PREC(b)-1)/GMP_NUMB_BITS;
  cn = (MPFR_PREC(c)-1)/GMP_NUMB_BITS;

  bp = MPFR_MANT(b);
  cp = MPFR_MANT(c);

  for ( ; bn >= 0 && cn >= 0; bn--, cn--)
    {
      if (bp[bn] > cp[cn])
        return s;
      if (bp[bn] < cp[cn])
        return -s;
    }
  for ( ; bn >= 0; bn--)
    if (bp[bn])
      return s;
  for ( ; cn >= 0; cn--)
    if (cp[cn])
      return -s;

   return 0;
}

#undef mpfr_cmp
int
mpfr_cmp (mpfr_srcptr b, mpfr_srcptr c)
{
  return mpfr_cmp3 (b, c, 1);
}
