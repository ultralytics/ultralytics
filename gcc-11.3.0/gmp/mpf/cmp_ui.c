/* mpf_cmp_ui -- Compare a float with an unsigned integer.

Copyright 1993-1995, 1999, 2001, 2002, 2015 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

#include "gmp.h"
#include "gmp-impl.h"

int
mpf_cmp_ui (mpf_srcptr u, unsigned long int vval) __GMP_NOTHROW
{
  mp_srcptr up;
  mp_size_t usize;
  mp_exp_t uexp;
  mp_limb_t ulimb;

  usize = SIZ (u);

  /* 1. Is U negative?  */
  if (usize < 0)
    return -1;
  /* We rely on usize being non-negative in the code that follows.  */

  if (vval == 0)
    return usize != 0;

  /* 2. Are the exponents different (V's exponent == 1)?  */
  uexp = EXP (u);

#if GMP_NAIL_BITS != 0
  if (uexp != 1 + (vval > GMP_NUMB_MAX))
    return (uexp < 1 + (vval > GMP_NUMB_MAX)) ? -1 : 1;
#else
  if (uexp != 1)
    return (uexp < 1) ? -1 : 1;
#endif

  up = PTR (u);

  ASSERT (usize > 0);
  ulimb = up[--usize];
#if GMP_NAIL_BITS != 0
  if (uexp == 2)
    {
      if ((ulimb >> GMP_NAIL_BITS) != 0)
	return 1;
      ulimb = (ulimb << GMP_NUMB_BITS);
      if (usize != 0) ulimb |= up[--usize];
    }
#endif

  /* 3. Compare the most significant mantissa limb with V.  */
  if (ulimb != vval)
    return (ulimb < vval) ? -1 : 1;

  /* Ignore zeroes at the low end of U.  */
  for (; *up == 0; ++up)
    --usize;

  /* 4. Now, if the number of limbs are different, we have a difference
     since we have made sure the trailing limbs are not zero.  */
  return (usize > 0);
}
