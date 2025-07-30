/* Test mpz_abs, mpz_add, mpz_cmp, mpz_cmp_ui, mpz_tdiv_qr, mpz_tdiv_q,
   mpz_tdiv_r, mpz_mul.

Copyright 1991, 1993, 1994, 1996, 1997, 2000, 2001 Free Software Foundation, Inc.

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

void dump_abort (mpz_t, mpz_t);
void debug_mp (mpz_t, int);

int
main (int argc, char **argv)
{
  mpz_t dividend, divisor;
  mpz_t quotient, remainder;
  mpz_t quotient2, remainder2;
  mpz_t temp;
  mp_size_t dividend_size, divisor_size;
  int i;
  int reps = 1000;
  gmp_randstate_ptr rands;
  mpz_t bs;
  unsigned long bsi, size_range;

  tests_start ();
  TESTS_REPS (reps, argv, argc);

  rands = RANDS;

  mpz_init (bs);

  mpz_init (dividend);
  mpz_init (divisor);
  mpz_init (quotient);
  mpz_init (remainder);
  mpz_init (quotient2);
  mpz_init (remainder2);
  mpz_init (temp);

  for (i = 0; i < reps; i++)
    {
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 18 + 2; /* 0..524288 bit operands */

      do
	{
	  mpz_urandomb (bs, rands, size_range);
	  divisor_size = mpz_get_ui (bs);
	  mpz_rrandomb (divisor, rands, divisor_size);
	}
      while (mpz_sgn (divisor) == 0);

      mpz_urandomb (bs, rands, size_range);
      dividend_size = mpz_get_ui (bs) + divisor_size;
      mpz_rrandomb (dividend, rands, dividend_size);

      mpz_urandomb (bs, rands, 2);
      bsi = mpz_get_ui (bs);
      if ((bsi & 1) != 0)
	mpz_neg (dividend, dividend);
      if ((bsi & 2) != 0)
	mpz_neg (divisor, divisor);

      /* printf ("%ld %ld\n", SIZ (dividend), SIZ (divisor)); */

      mpz_tdiv_qr (quotient, remainder, dividend, divisor);
      mpz_tdiv_q (quotient2, dividend, divisor);
      mpz_tdiv_r (remainder2, dividend, divisor);

      /* First determine that the quotients and remainders computed
	 with different functions are equal.  */
      if (mpz_cmp (quotient, quotient2) != 0)
	dump_abort (dividend, divisor);
      if (mpz_cmp (remainder, remainder2) != 0)
	dump_abort (dividend, divisor);

      /* Check if the sign of the quotient is correct.  */
      if (mpz_cmp_ui (quotient, 0) != 0)
	if ((mpz_cmp_ui (quotient, 0) < 0)
	    != ((mpz_cmp_ui (dividend, 0) ^ mpz_cmp_ui (divisor, 0)) < 0))
	dump_abort (dividend, divisor);

      /* Check if the remainder has the same sign as the dividend
	 (quotient rounded towards 0).  */
      if (mpz_cmp_ui (remainder, 0) != 0)
	if ((mpz_cmp_ui (remainder, 0) < 0) != (mpz_cmp_ui (dividend, 0) < 0))
	  dump_abort (dividend, divisor);

      mpz_mul (temp, quotient, divisor);
      mpz_add (temp, temp, remainder);
      if (mpz_cmp (temp, dividend) != 0)
	dump_abort (dividend, divisor);

      mpz_abs (temp, divisor);
      mpz_abs (remainder, remainder);
      if (mpz_cmp (remainder, temp) >= 0)
	dump_abort (dividend, divisor);
    }

  mpz_clear (bs);
  mpz_clear (dividend);
  mpz_clear (divisor);
  mpz_clear (quotient);
  mpz_clear (remainder);
  mpz_clear (quotient2);
  mpz_clear (remainder2);
  mpz_clear (temp);

  tests_end ();
  exit (0);
}

void
dump_abort (mpz_t dividend, mpz_t divisor)
{
  fprintf (stderr, "ERROR\n");
  fprintf (stderr, "dividend = "); debug_mp (dividend, -16);
  fprintf (stderr, "divisor  = "); debug_mp (divisor, -16);
  abort();
}

void
debug_mp (mpz_t x, int base)
{
  mpz_out_str (stderr, base, x); fputc ('\n', stderr);
}
