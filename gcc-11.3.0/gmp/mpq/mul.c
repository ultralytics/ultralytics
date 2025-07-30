/* mpq_mul -- multiply two rational numbers.

Copyright 1991, 1994-1996, 2000-2002 Free Software Foundation, Inc.

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
mpq_mul (mpq_ptr prod, mpq_srcptr op1, mpq_srcptr op2)
{
  mpz_t gcd1, gcd2;
  mpz_t tmp1, tmp2;
  mp_size_t op1_num_size;
  mp_size_t op1_den_size;
  mp_size_t op2_num_size;
  mp_size_t op2_den_size;
  mp_size_t alloc;
  TMP_DECL;

  if (op1 == op2)
    {
      /* No need for any GCDs when squaring. */
      mpz_mul (mpq_numref (prod), mpq_numref (op1), mpq_numref (op1));
      mpz_mul (mpq_denref (prod), mpq_denref (op1), mpq_denref (op1));
      return;
    }

  op1_num_size = ABSIZ(NUM(op1));
  op1_den_size =   SIZ(DEN(op1));
  op2_num_size = ABSIZ(NUM(op2));
  op2_den_size =   SIZ(DEN(op2));

  if (op1_num_size == 0 || op2_num_size == 0)
    {
      /* We special case this to simplify allocation logic; gcd(0,x) = x
	 is a singular case for the allocations.  */
      SIZ(NUM(prod)) = 0;
      PTR(DEN(prod))[0] = 1;
      SIZ(DEN(prod)) = 1;
      return;
    }

  TMP_MARK;

  alloc = MIN (op1_num_size, op2_den_size);
  MPZ_TMP_INIT (gcd1, alloc);

  alloc = MIN (op2_num_size, op1_den_size);
  MPZ_TMP_INIT (gcd2, alloc);

  alloc = MAX (op1_num_size, op2_den_size);
  MPZ_TMP_INIT (tmp1, alloc);

  alloc = MAX (op2_num_size, op1_den_size);
  MPZ_TMP_INIT (tmp2, alloc);

  /* PROD might be identical to either operand, so don't store the result there
     until we are finished with the input operands.  We can overwrite the
     numerator of PROD when we are finished with the numerators of OP1 and
     OP2.  */

  mpz_gcd (gcd1, NUM(op1), DEN(op2));
  mpz_gcd (gcd2, NUM(op2), DEN(op1));

  mpz_divexact_gcd (tmp1, NUM(op1), gcd1);
  mpz_divexact_gcd (tmp2, NUM(op2), gcd2);

  mpz_mul (NUM(prod), tmp1, tmp2);

  mpz_divexact_gcd (tmp1, DEN(op2), gcd1);
  mpz_divexact_gcd (tmp2, DEN(op1), gcd2);

  mpz_mul (DEN(prod), tmp1, tmp2);

  TMP_FREE;
}
