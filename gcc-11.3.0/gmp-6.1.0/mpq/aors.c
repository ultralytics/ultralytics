/* mpq_add, mpq_sub -- add or subtract rational numbers.

Copyright 1991, 1994-1997, 2000, 2001, 2004, 2005 Free Software Foundation,
Inc.

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


static void __gmpq_aors (REGPARM_3_1 (mpq_ptr, mpq_srcptr, mpq_srcptr, void (*) (mpz_ptr, mpz_srcptr, mpz_srcptr))) REGPARM_ATTR (1);
#define mpq_aors(w,x,y,fun)  __gmpq_aors (REGPARM_3_1 (w, x, y, fun))

REGPARM_ATTR (1) static void
mpq_aors (mpq_ptr rop, mpq_srcptr op1, mpq_srcptr op2,
          void (*fun) (mpz_ptr, mpz_srcptr, mpz_srcptr))
{
  mpz_t gcd;
  mpz_t tmp1, tmp2;
  mp_size_t op1_num_size = ABSIZ(NUM(op1));
  mp_size_t op1_den_size =   SIZ(DEN(op1));
  mp_size_t op2_num_size = ABSIZ(NUM(op2));
  mp_size_t op2_den_size =   SIZ(DEN(op2));
  TMP_DECL;

  TMP_MARK;
  MPZ_TMP_INIT (gcd, MIN (op1_den_size, op2_den_size));
  MPZ_TMP_INIT (tmp1, op1_num_size + op2_den_size);
  MPZ_TMP_INIT (tmp2, op2_num_size + op1_den_size);

  /* ROP might be identical to either operand, so don't store the
     result there until we are finished with the input operands.  We
     dare to overwrite the numerator of ROP when we are finished
     with the numerators of OP1 and OP2.  */

  mpz_gcd (gcd, DEN(op1), DEN(op2));
  if (! MPZ_EQUAL_1_P (gcd))
    {
      mpz_t t;

      MPZ_TMP_INIT (t, MAX (op1_num_size + op2_den_size,
	     op2_num_size + op1_den_size) + 2 - SIZ(gcd));

      mpz_divexact_gcd (t, DEN(op2), gcd);
      mpz_divexact_gcd (tmp2, DEN(op1), gcd);

      mpz_mul (tmp1, NUM(op1), t);
      mpz_mul (t, NUM(op2), tmp2);

      (*fun) (t, tmp1, t);

      mpz_gcd (gcd, t, gcd);
      if (MPZ_EQUAL_1_P (gcd))
        {
          mpz_set (NUM(rop), t);
          mpz_mul (DEN(rop), DEN(op2), tmp2);
        }
      else
        {
          mpz_divexact_gcd (NUM(rop), t, gcd);
          mpz_divexact_gcd (tmp1, DEN(op2), gcd);
          mpz_mul (DEN(rop), tmp1, tmp2);
        }
    }
  else
    {
      /* The common divisor is 1.  This is the case (for random input) with
	 probability 6/(pi**2), which is about 60.8%.  */
      mpz_mul (tmp1, NUM(op1), DEN(op2));
      mpz_mul (tmp2, NUM(op2), DEN(op1));
      (*fun) (NUM(rop), tmp1, tmp2);
      mpz_mul (DEN(rop), DEN(op1), DEN(op2));
    }
  TMP_FREE;
}


void
mpq_add (mpq_ptr rop, mpq_srcptr op1, mpq_srcptr op2)
{
  mpq_aors (rop, op1, op2, mpz_add);
}

void
mpq_sub (mpq_ptr rop, mpq_srcptr op1, mpq_srcptr op2)
{
  mpq_aors (rop, op1, op2, mpz_sub);
}
