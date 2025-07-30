/* mpq_canonicalize(op) -- Remove common factors of the denominator and
   numerator in OP.

Copyright 1991, 1994-1996, 2000, 2001, 2005, 2014 Free Software Foundation, Inc.

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
mpq_canonicalize (mpq_t op)
{
  mpz_t gcd;
  TMP_DECL;

  if (SIZ(DEN(op)) < 0)
    {
      SIZ(NUM(op)) = -SIZ(NUM(op));
      SIZ(DEN(op)) = -SIZ(DEN(op));
    }
  else if (UNLIKELY (SIZ(DEN(op)) == 0))
    DIVIDE_BY_ZERO;

  TMP_MARK;

  /* ??? Dunno if the 1+ is needed.  */
  MPZ_TMP_INIT (gcd, 1 + MAX (ABSIZ(NUM(op)),
			      SIZ(DEN(op))));

  mpz_gcd (gcd, NUM(op), DEN(op));
  if (! MPZ_EQUAL_1_P (gcd))
    {
      mpz_divexact_gcd (NUM(op), NUM(op), gcd);
      mpz_divexact_gcd (DEN(op), DEN(op), gcd);
    }
  TMP_FREE;
}
