/* mpz_invert (inv, x, n).  Find multiplicative inverse of X in Z(N).
   If X has an inverse, return non-zero and store inverse in INVERSE,
   otherwise, return 0 and put garbage in INVERSE.

Copyright 1996-2001, 2005, 2012, 2014 Free Software Foundation, Inc.

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
mpz_invert (mpz_ptr inverse, mpz_srcptr x, mpz_srcptr n)
{
  mpz_t gcd, tmp;
  mp_size_t xsize, nsize, size;
  TMP_DECL;

  xsize = ABSIZ (x);
  nsize = ABSIZ (n);

  size = MAX (xsize, nsize) + 1;
  TMP_MARK;

  MPZ_TMP_INIT (gcd, size);
  MPZ_TMP_INIT (tmp, size);
  mpz_gcdext (gcd, tmp, (mpz_ptr) 0, x, n);

  /* If no inverse existed, return with an indication of that.  */
  if (!MPZ_EQUAL_1_P (gcd))
    {
      TMP_FREE;
      return 0;
    }

  /* Make sure we return a positive inverse.  */
  if (SIZ (tmp) < 0)
    {
      if (SIZ (n) < 0)
	mpz_sub (inverse, tmp, n);
      else
	mpz_add (inverse, tmp, n);
    }
  else
    mpz_set (inverse, tmp);

  TMP_FREE;
  return 1;
}
