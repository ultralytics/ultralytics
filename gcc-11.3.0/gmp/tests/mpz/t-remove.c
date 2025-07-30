/* Test mpz_remove.

Copyright 1991, 1993, 1994, 1996, 1997, 2000, 2001, 2009, 2012, 2013 Free
Software Foundation, Inc.

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

void debug_mp (mpz_t);
unsigned long int mpz_refremove (mpz_t, const mpz_t, const mpz_t);

int
main (int argc, char **argv)
{
  unsigned long int exp;
  mpz_t t, dest, refdest, dividend, divisor;
  mp_size_t dividend_size, divisor_size;
  int i;
  int reps = 1000;
  unsigned long int pwr, refpwr;
  gmp_randstate_ptr rands;
  mpz_t bs;
  unsigned long size_range;

  tests_start ();
  rands = RANDS;

  if (argc == 2)
    reps = atoi (argv[1]);

  mpz_inits (bs, t, dest, refdest, dividend, divisor, NULL);

  for (i = 0; i < reps; i++)
    {
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 18 + 1; /* 1..524288 bit operands */

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

      mpz_urandomb (bs, rands, 32);
      exp = mpz_get_ui (bs) % (5 + 10000 / mpz_sizeinbase (divisor, 2));
      if (mpz_get_ui (bs) & 2)
	mpz_neg (divisor, divisor);
      mpz_pow_ui (t, divisor, exp);
      mpz_mul (dividend, dividend, t);

      refpwr = mpz_refremove (refdest, dividend, divisor);
      pwr = mpz_remove (dest, dividend, divisor);

      if (refpwr != pwr || mpz_cmp (refdest, dest) != 0)
	{
	  fprintf (stderr, "ERROR after %d tests\n", i);
	  fprintf (stderr, "refpower = %lu\n", refpwr);
	  fprintf (stderr, "   power = %lu\n", pwr);
	  fprintf (stderr, "    op1 = "); debug_mp (dividend);
	  fprintf (stderr, "    op2 = "); debug_mp (divisor);
	  fprintf (stderr, "refdest = "); debug_mp (refdest);
	  fprintf (stderr, "   dest = "); debug_mp (dest);
	  abort ();
	}
    }

  mpz_clears (bs, t, dest, refdest, dividend, divisor, NULL);

  tests_end ();
  exit (0);
}

unsigned long int
mpz_refremove (mpz_t dest, const mpz_t src, const mpz_t f)
{
  unsigned long int pwr;

  pwr = 0;

  mpz_set (dest, src);
  if (mpz_cmpabs_ui (f, 1) > 0)
    {
      mpz_t rem, x;

      mpz_init (x);
      mpz_init (rem);

      for (;; pwr++)
	{
	  mpz_tdiv_qr (x, rem, dest, f);
	  if (mpz_cmp_ui (rem, 0) != 0)
	    break;
	  mpz_swap (dest, x);
	}

      mpz_clear (x);
      mpz_clear (rem);
    }

  return pwr;
}

void
debug_mp (mpz_t x)
{
  size_t siz = mpz_sizeinbase (x, 16);

  if (siz > 65)
    {
      mpz_t q;
      mpz_init (q);
      mpz_tdiv_q_2exp (q, x, 4 * (mpz_sizeinbase (x, 16) - 25));
      gmp_fprintf (stderr, "%ZX...", q);
      mpz_tdiv_r_2exp (q, x, 4 * 25);
      gmp_fprintf (stderr, "%025ZX [%d]\n", q, (int) siz);
      mpz_clear (q);
    }
  else
    {
      gmp_fprintf (stderr, "%ZX\n", x);
    }
}
