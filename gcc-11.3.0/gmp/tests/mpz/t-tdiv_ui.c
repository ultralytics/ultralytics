/* Test mpz_abs, mpz_add, mpz_cmp, mpz_cmp_ui, mpz_tdiv_qr_ui, mpz_tdiv_q_ui,
   mpz_tdiv_r_ui, mpz_tdiv_ui, mpz_mul_ui.

Copyright 1993, 1994, 1996, 2000-2002 Free Software Foundation, Inc.

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

void dump_abort (const char *, mpz_t, unsigned long);
void debug_mp (mpz_t, int);

int
main (int argc, char **argv)
{
  mpz_t dividend;
  mpz_t quotient, remainder;
  mpz_t quotient2, remainder2;
  mpz_t temp;
  mp_size_t dividend_size;
  unsigned long divisor;
  int i;
  int reps = 200000;
  gmp_randstate_ptr rands;
  mpz_t bs;
  unsigned long bsi, size_range;
  unsigned long r_rq, r_q, r_r, r;

  tests_start ();
  rands = RANDS;

  mpz_init (bs);

  if (argc == 2)
     reps = atoi (argv[1]);

  mpz_init (dividend);
  mpz_init (quotient);
  mpz_init (remainder);
  mpz_init (quotient2);
  mpz_init (remainder2);
  mpz_init (temp);

  for (i = 0; i < reps; i++)
    {
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 10 + 2; /* 0..2047 bit operands */

      do
	{
	  mpz_rrandomb (bs, rands, 64);
	  divisor = mpz_get_ui (bs);
	}
      while (divisor == 0);

      mpz_urandomb (bs, rands, size_range);
      dividend_size = mpz_get_ui (bs);
      mpz_rrandomb (dividend, rands, dividend_size);

      mpz_urandomb (bs, rands, 2);
      bsi = mpz_get_ui (bs);
      if ((bsi & 1) != 0)
	mpz_neg (dividend, dividend);

      /* printf ("%ld\n", SIZ (dividend)); */

      r_rq = mpz_tdiv_qr_ui (quotient, remainder, dividend, divisor);
      r_q = mpz_tdiv_q_ui (quotient2, dividend, divisor);
      r_r = mpz_tdiv_r_ui (remainder2, dividend, divisor);
      r = mpz_tdiv_ui (dividend, divisor);

      /* First determine that the quotients and remainders computed
	 with different functions are equal.  */
      if (mpz_cmp (quotient, quotient2) != 0)
	dump_abort ("quotients from mpz_tdiv_qr_ui and mpz_tdiv_q_ui differ",
		    dividend, divisor);
      if (mpz_cmp (remainder, remainder2) != 0)
	dump_abort ("remainders from mpz_tdiv_qr_ui and mpz_tdiv_r_ui differ",
		    dividend, divisor);

      /* Check if the sign of the quotient is correct.  */
      if (mpz_cmp_ui (quotient, 0) != 0)
	if ((mpz_cmp_ui (quotient, 0) < 0)
	    != (mpz_cmp_ui (dividend, 0) < 0))
	dump_abort ("quotient sign wrong", dividend, divisor);

      /* Check if the remainder has the same sign as the dividend
	 (quotient rounded towards 0).  */
      if (mpz_cmp_ui (remainder, 0) != 0)
	if ((mpz_cmp_ui (remainder, 0) < 0) != (mpz_cmp_ui (dividend, 0) < 0))
	  dump_abort ("remainder sign wrong", dividend, divisor);

      mpz_mul_ui (temp, quotient, divisor);
      mpz_add (temp, temp, remainder);
      if (mpz_cmp (temp, dividend) != 0)
	dump_abort ("n mod d != n - [n/d]*d", dividend, divisor);

      mpz_abs (remainder, remainder);
      if (mpz_cmp_ui (remainder, divisor) >= 0)
	dump_abort ("remainder greater than divisor", dividend, divisor);

      if (mpz_cmp_ui (remainder, r_rq) != 0)
	dump_abort ("remainder returned from mpz_tdiv_qr_ui is wrong",
		    dividend, divisor);
      if (mpz_cmp_ui (remainder, r_q) != 0)
	dump_abort ("remainder returned from mpz_tdiv_q_ui is wrong",
		    dividend, divisor);
      if (mpz_cmp_ui (remainder, r_r) != 0)
	dump_abort ("remainder returned from mpz_tdiv_r_ui is wrong",
		    dividend, divisor);
      if (mpz_cmp_ui (remainder, r) != 0)
	dump_abort ("remainder returned from mpz_tdiv_ui is wrong",
		    dividend, divisor);
    }

  mpz_clear (bs);
  mpz_clear (dividend);
  mpz_clear (quotient);
  mpz_clear (remainder);
  mpz_clear (quotient2);
  mpz_clear (remainder2);
  mpz_clear (temp);

  tests_end ();
  exit (0);
}

void
dump_abort (const char *str, mpz_t dividend, unsigned long divisor)
{
  fprintf (stderr, "ERROR: %s\n", str);
  fprintf (stderr, "dividend = "); debug_mp (dividend, -16);
  fprintf (stderr, "divisor  = %lX\n", divisor);
  abort();
}

void
debug_mp (mpz_t x, int base)
{
  mpz_out_str (stderr, base, x); fputc ('\n', stderr);
}
