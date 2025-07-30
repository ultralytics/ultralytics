/* mpfr_cmpabs -- compare the absolute values of two FP numbers

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

#include "mpfr-impl.h"

/* Return a positive value if abs(b) > abs(c), 0 if abs(b) = abs(c), and
   a negative value if abs(b) < abs(c). Neither b nor c may be NaN. */

int
mpfr_cmpabs (mpfr_srcptr b, mpfr_srcptr c)
{
  mpfr_exp_t be, ce;
  mp_size_t bn, cn;
  mp_limb_t *bp, *cp;

  if (MPFR_ARE_SINGULAR (b, c))
    {
      if (MPFR_IS_NAN (b) || MPFR_IS_NAN (c))
        {
          MPFR_SET_ERANGE ();
          return 0;
        }
      else if (MPFR_IS_INF (b))
        return ! MPFR_IS_INF (c);
      else if (MPFR_IS_INF (c))
        return -1;
      else if (MPFR_IS_ZERO (c))
        return ! MPFR_IS_ZERO (b);
      else /* b == 0 */
        return -1;
    }

  MPFR_ASSERTD (MPFR_IS_PURE_FP (b));
  MPFR_ASSERTD (MPFR_IS_PURE_FP (c));

  /* Now that we know that b and c are pure FP numbers (i.e. they have
     a meaningful exponent), we use MPFR_EXP instead of MPFR_GET_EXP to
     allow exponents outside the current exponent range. For instance,
     this is useful for mpfr_pow, which compares values to __gmpfr_one.
     This is for internal use only! For compatibility with other MPFR
     versions, the user must still provide values that are representable
     in the current exponent range. */
  be = MPFR_EXP (b);
  ce = MPFR_EXP (c);
  if (be > ce)
    return 1;
  if (be < ce)
    return -1;

  /* exponents are equal */

  bn = MPFR_LIMB_SIZE(b)-1;
  cn = MPFR_LIMB_SIZE(c)-1;

  bp = MPFR_MANT(b);
  cp = MPFR_MANT(c);

  for ( ; bn >= 0 && cn >= 0; bn--, cn--)
    {
      if (bp[bn] > cp[cn])
        return 1;
      if (bp[bn] < cp[cn])
        return -1;
    }

  for ( ; bn >= 0; bn--)
    if (bp[bn])
      return 1;

  for ( ; cn >= 0; cn--)
    if (cp[cn])
      return -1;

   return 0;
}
