/* Test mpz_powm, mpz_mul, mpz_mod, mpz_mod_ui, mpz_div_ui.

Copyright 1991, 1993, 1994, 1996, 1999-2001, 2009, 2012 Free Software
Foundation, Inc.

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
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

void debug_mp (mpz_t, int);

#define SIZEM 13

/* Check that all sizes up to just above MUL_TOOM22_THRESHOLD have been tested
   a few times.  FIXME: If SIZEM is set too low, this will never happen.  */
int
allsizes_seen (unsigned int *allsizes)
{
  mp_size_t i;

  for (i = 1; i < MUL_TOOM22_THRESHOLD + 4; i++)
    if (allsizes[i] < 4)
      return 0;
  return 1;
}

int
main (int argc, char **argv)
{
  mpz_t base, exp, mod;
  mpz_t r1, r2, t1, exp2, base2;
  mp_size_t base_size, exp_size, mod_size;
  int i;
  int reps = 1000;
  gmp_randstate_ptr rands;
  mpz_t bs;
  unsigned long bsi, size_range;
  unsigned int allsizes[1 << (SIZEM + 2 - 1)];

  tests_start ();
  TESTS_REPS (reps, argv, argc);

  rands = RANDS;

  mpz_init (bs);

  mpz_init (base);
  mpz_init (exp);
  mpz_init (mod);
  mpz_init (r1);
  mpz_init (r2);
  mpz_init (t1);
  mpz_init (exp2);
  mpz_init (base2);

  memset (allsizes, 0, (1 << (SIZEM + 2 - 1)) * sizeof (int));

  for (i = 0; i < reps || ! allsizes_seen (allsizes); i++)
    {
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % SIZEM + 2;

      do  /* Loop until mathematically well-defined.  */
	{
	  mpz_urandomb (bs, rands, size_range);
	  base_size = mpz_get_ui (bs);
	  mpz_rrandomb (base, rands, base_size);

	  mpz_urandomb (bs, rands, 7L);
	  exp_size = mpz_get_ui (bs);
	  mpz_rrandomb (exp, rands, exp_size);
	}
      while (mpz_cmp_ui (base, 0) == 0 && mpz_cmp_ui (exp, 0) == 0);

      do
        {
	  mpz_urandomb (bs, rands, size_range);
	  mod_size = mpz_get_ui (bs);
	  mpz_rrandomb (mod, rands, mod_size);
	}
      while (mpz_cmp_ui (mod, 0) == 0);

      allsizes[SIZ(mod)] += 1;

      mpz_urandomb (bs, rands, 2);
      bsi = mpz_get_ui (bs);
      if ((bsi & 1) != 0)
	mpz_neg (base, base);

      /* printf ("%ld %ld %ld\n", SIZ (base), SIZ (exp), SIZ (mod)); */

      mpz_set_ui (r2, 1);
      mpz_mod (base2, base, mod);
      mpz_set (exp2, exp);
      mpz_mod (r2, r2, mod);

      for (;;)
	{
	  if (mpz_tstbit (exp2, 0))
	    {
	      mpz_mul (r2, r2, base2);
	      mpz_mod (r2, r2, mod);
	    }
	  if  (mpz_cmp_ui (exp2, 1) <= 0)
	    break;
	  mpz_mul (base2, base2, base2);
	  mpz_mod (base2, base2, mod);
	  mpz_tdiv_q_2exp (exp2, exp2, 1);
	}

      mpz_powm (r1, base, exp, mod);
      MPZ_CHECK_FORMAT (r1);

      if (mpz_cmp (r1, r2) != 0)
	{
	  fprintf (stderr, "\nIncorrect results in test %d for operands:\n", i);
	  debug_mp (base, -16);
	  debug_mp (exp, -16);
	  debug_mp (mod, -16);
	  fprintf (stderr, "mpz_powm result:\n");
	  debug_mp (r1, -16);
	  fprintf (stderr, "reference result:\n");
	  debug_mp (r2, -16);
	  abort ();
	}

      if (mpz_tdiv_ui (mod, 2) == 0)
	continue;

      mpz_powm_sec (r1, base, exp, mod);
      MPZ_CHECK_FORMAT (r1);

      if (mpz_cmp (r1, r2) != 0)
	{
	  fprintf (stderr, "\nIncorrect results in test %d for operands:\n", i);
	  debug_mp (base, -16);
	  debug_mp (exp, -16);
	  debug_mp (mod, -16);
	  fprintf (stderr, "mpz_powm_sec result:\n");
	  debug_mp (r1, -16);
	  fprintf (stderr, "reference result:\n");
	  debug_mp (r2, -16);
	  abort ();
	}
    }

  mpz_clear (bs);
  mpz_clear (base);
  mpz_clear (exp);
  mpz_clear (mod);
  mpz_clear (r1);
  mpz_clear (r2);
  mpz_clear (t1);
  mpz_clear (exp2);
  mpz_clear (base2);

  tests_end ();
  exit (0);
}

void
debug_mp (mpz_t x, int base)
{
  mpz_out_str (stderr, base, x); fputc ('\n', stderr);
}
