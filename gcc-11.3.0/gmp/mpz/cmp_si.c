/* mpz_cmp_si(u,v) -- Compare an integer U with a single-word int V.
   Return positive, zero, or negative based on if U > V, U == V, or U < V.

Copyright 1991, 1993-1996, 2000-2002, 2012, 2013 Free Software Foundation, Inc.

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
_mpz_cmp_si (mpz_srcptr u, signed long int v_digit) __GMP_NOTHROW
{
#if GMP_NAIL_BITS != 0
  /* FIXME.  This isn't very pretty.  */
  mpz_t tmp;
  mp_limb_t tt[2];
  PTR(tmp) = tt;
  ALLOC(tmp) = 2;
  mpz_set_si (tmp, v_digit);
  return mpz_cmp (u, tmp);
#else

  mp_size_t vsize, usize;

  usize = SIZ (u);
  vsize = (v_digit > 0) - (v_digit < 0);

  if ((usize == 0) | (usize != vsize))
    return usize - vsize;
  else {
    mp_limb_t u_digit, absv_digit;

    u_digit = PTR (u)[0];
    absv_digit = ABS_CAST (unsigned long, v_digit);

    if (u_digit == absv_digit)
      return 0;

    if (u_digit > absv_digit)
      return usize;
    else
      return -usize;
  }
#endif
}
