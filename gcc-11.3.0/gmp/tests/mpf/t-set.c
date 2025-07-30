/* Test mpf_set, mpf_init_set.

Copyright 2004, 2012 Free Software Foundation, Inc.

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

void
check_reuse (void)
{
  /* Try mpf_set(f,f) when f is bigger than prec.  In the past this had
     resulted in an MPN_COPY with invalid operand overlap. */
  mpf_t  f;
  mp_size_t      limbs = 20;
  unsigned long  bits = limbs * GMP_NUMB_BITS;
  mpf_init2 (f, bits);
  refmpf_fill (f, limbs, GMP_NUMB_MAX);
  mpf_set_prec_raw (f, bits / 2);
  mpf_set (f, f);
  MPF_CHECK_FORMAT (f);
  mpf_set_prec_raw (f, bits);
  mpf_clear (f);
}

void
check_random (long reps)
{
  unsigned long test;
  gmp_randstate_ptr rands;
  mpf_t a, b;
  mpz_t z;
  int precbits;

#define PRECBITS 10

  rands = RANDS;

  mpz_init (z);
  mpf_init2 (a, 1 << PRECBITS);

  for (test = 0; test < reps; test++)
    {
      mpz_urandomb (z, rands, PRECBITS + 1);
      precbits = mpz_get_ui (z) + 1;
      mpz_urandomb (z, rands, precbits);
      mpz_setbit (z, precbits  - 1);	/* make sure msb is set */
      mpf_set_z (a, z);
      if (precbits & 1)
	mpf_neg (a, a);
      mpz_urandomb (z, rands, PRECBITS);
      mpf_div_2exp (a, a, mpz_get_ui (z) + 1);
      mpz_urandomb (z, rands, PRECBITS);
      precbits -= mpz_get_ui (z);
      if (precbits <= 0)
	precbits = 1 - precbits;
      mpf_set_default_prec (precbits);

      mpf_init_set (b, a);
      MPF_CHECK_FORMAT (b);
      if (!mpf_eq (a, b, precbits))
	{
	  printf ("mpf_init_set wrong.\n");
	  abort();
	}

      mpf_set_ui (b, 0);
      mpf_set (b, a);
      MPF_CHECK_FORMAT (b);
      if (!mpf_eq (a, b, precbits))
	{
	  printf ("mpf_set wrong.\n");
	  abort();
	}

      mpf_clear (b);
    }

  mpf_clear (a);
  mpz_clear (z);
}

int
main (int argc, char *argv[])
{
  long reps = 10000;

  tests_start ();
  TESTS_REPS (reps, argv, argc);

  check_reuse ();
  check_random (reps);

  tests_end ();
  exit (0);
}
