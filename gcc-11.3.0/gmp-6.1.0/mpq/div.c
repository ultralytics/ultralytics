/* mpq_div -- divide two rational numbers.

Copyright 1991, 1994-1996, 2000, 2001, 2015 Free Software Foundation, Inc.

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
mpq_div (mpq_ptr quot, mpq_srcptr op1, mpq_srcptr op2)
{
  mpz_t gcd1, gcd2;
  mpz_t tmp1, tmp2;
  mp_size_t op1_size;
  mp_size_t op2_size;
  mp_size_t alloc;
  TMP_DECL;

  op2_size = SIZ(NUM(op2));

  if (UNLIKELY (op2_size == 0))
    DIVIDE_BY_ZERO;

  if (UNLIKELY (quot == op2))
    {
      if (op1 == op2)
	{
	  PTR(NUM(quot))[0] = 1;
	  SIZ(NUM(quot)) = 1;
	  PTR(DEN(quot))[0] = 1;
	  SIZ(DEN(quot)) = 1;
	  return;
	}

      /* We checked for op1 == op2: we are not in the x=x/x case.
	 We compute x=y/x by computing x=inv(x)*y */
      MPN_PTR_SWAP (PTR(NUM(quot)), ALLOC(NUM(quot)),
		    PTR(DEN(quot)), ALLOC(DEN(quot)));
      if (op2_size > 0)
	{
	  SIZ(NUM(quot)) = SIZ(DEN(quot));
	  SIZ(DEN(quot)) = op2_size;
	}
      else
	{
	  SIZ(NUM(quot)) = - SIZ(DEN(quot));
	  SIZ(DEN(quot)) = - op2_size;
	}
      mpq_mul (quot, quot, op1);
      return;
    }

  op1_size = ABSIZ(NUM(op1));

  if (op1_size == 0)
    {
      /* We special case this to simplify allocation logic; gcd(0,x) = x
	 is a singular case for the allocations.  */
      SIZ(NUM(quot)) = 0;
      PTR(DEN(quot))[0] = 1;
      SIZ(DEN(quot)) = 1;
      return;
    }

  op2_size = ABS(op2_size);

  TMP_MARK;

  alloc = MIN (op1_size, op2_size);
  MPZ_TMP_INIT (gcd1, alloc);

  alloc = MAX (op1_size, op2_size);
  MPZ_TMP_INIT (tmp1, alloc);

  op2_size = SIZ(DEN(op2));
  op1_size = SIZ(DEN(op1));

  alloc = MIN (op1_size, op2_size);
  MPZ_TMP_INIT (gcd2, alloc);

  alloc = MAX (op1_size, op2_size);
  MPZ_TMP_INIT (tmp2, alloc);

  /* QUOT might be identical to OP1, so don't store the result there
     until we are finished with the input operand.  We can overwrite
     the numerator of QUOT when we are finished with the numerator of
     OP1. */

  mpz_gcd (gcd1, NUM(op1), NUM(op2));
  mpz_gcd (gcd2, DEN(op2), DEN(op1));

  mpz_divexact_gcd (tmp1, NUM(op1), gcd1);
  mpz_divexact_gcd (tmp2, DEN(op2), gcd2);

  mpz_mul (NUM(quot), tmp1, tmp2);

  mpz_divexact_gcd (tmp1, NUM(op2), gcd1);
  mpz_divexact_gcd (tmp2, DEN(op1), gcd2);

  mpz_mul (DEN(quot), tmp1, tmp2);

  /* Keep the denominator positive.  */
  if (SIZ(DEN(quot)) < 0)
    {
      SIZ(DEN(quot)) = -SIZ(DEN(quot));
      SIZ(NUM(quot)) = -SIZ(NUM(quot));
    }

  TMP_FREE;
}
