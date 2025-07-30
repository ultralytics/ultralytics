/* Test mpz_mul, mpz_divexact.

Copyright 1996, 2001, 2002 Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

#include <stdio.h>
#include <stdlib.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

int
main (int argc, char **argv)
{
  mpz_t op1, op2;
  mpz_t prod, quot;
  mp_size_t size;
  int i;
  int reps = 5000;
  gmp_randstate_ptr rands;
  mpz_t bs;
  unsigned long bsi, size_range;

  tests_start ();
  TESTS_REPS (reps, argv, argc);

  rands = RANDS;

  mp_trace_base = -16;

  mpz_init (bs);

  mpz_init (op1);
  mpz_init (op2);
  mpz_init (prod);
  mpz_init (quot);

  for (i = 0; i < reps; i++)
    {
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 17 + 2; /* 0..2047 bit operands */

      mpz_urandomb (bs, rands, size_range);
      size = mpz_get_ui (bs);
      mpz_rrandomb (op1, rands, size);

      do
	{
	  mpz_urandomb (bs, rands, size_range);
	  size = mpz_get_ui (bs);
	  mpz_rrandomb (op2, rands, size);
	}
      while (mpz_sgn (op2) == 0);

      mpz_urandomb (bs, rands, 2);
      bsi = mpz_get_ui (bs);
      if ((bsi & 1) != 0)
	mpz_neg (op1, op1);
      if ((bsi & 2) != 0)
	mpz_neg (op2, op2);

      mpz_mul (prod, op1, op2);

      mpz_divexact (quot, prod, op2);
      MPZ_CHECK_FORMAT (quot);

      if (mpz_cmp (quot, op1) != 0)
        {
          printf ("Wrong results:\n");
          mpz_trace ("  got     ", quot);
          mpz_trace ("  want    ", op1);
          mpz_trace ("  dividend", prod);
          mpz_trace ("  divisor ", op2);
          abort ();
        }
    }

  mpz_clear (bs);
  mpz_clear (op1);
  mpz_clear (op2);
  mpz_clear (prod);
  mpz_clear (quot);

  tests_end ();
  exit (0);
}
