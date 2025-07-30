/* mpz_mod -- The mathematical mod function.

Copyright 1991, 1993-1996, 2001, 2002, 2005, 2010, 2012 Free Software
Foundation, Inc.

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

void
mpz_mod (mpz_ptr rem, mpz_srcptr dividend, mpz_srcptr divisor)
{
  mp_size_t rn, bn;
  mpz_t temp_divisor;
  TMP_DECL;

  TMP_MARK;

  bn = ABSIZ(divisor);

  /* We need the original value of the divisor after the remainder has been
     preliminary calculated.  We have to copy it to temporary space if it's
     the same variable as REM.  */
  if (rem == divisor)
    {
      PTR(temp_divisor) = TMP_ALLOC_LIMBS (bn);
      MPN_COPY (PTR(temp_divisor), PTR(divisor), bn);
    }
  else
    {
      PTR(temp_divisor) = PTR(divisor);
    }
  SIZ(temp_divisor) = bn;
  divisor = temp_divisor;

  mpz_tdiv_r (rem, dividend, divisor);

  rn = SIZ (rem);
  if (rn < 0)
    mpz_add (rem, rem, divisor);

  TMP_FREE;
}
