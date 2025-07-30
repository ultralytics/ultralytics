/* Test mpz_root, mpz_rootrem, and mpz_perfect_power_p.

Copyright 1991, 1993, 1994, 1996, 2000, 2001, 2009, 2015 Free Software
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

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

void debug_mp (mpz_t, int);

void
check_one (mpz_t root1, mpz_t x2, unsigned long nth, int res, int i)
{
  mpz_t temp, temp2;
  mpz_t root2, rem2;

  mpz_init (root2);
  mpz_init (rem2);
  mpz_init (temp);
  mpz_init (temp2);

  MPZ_CHECK_FORMAT (root1);

  mpz_rootrem (root2, rem2, x2, nth);
  MPZ_CHECK_FORMAT (root2);
  MPZ_CHECK_FORMAT (rem2);

  mpz_pow_ui (temp, root1, nth);
  MPZ_CHECK_FORMAT (temp);

  mpz_add (temp2, temp, rem2);

  /* Is power of result > argument?  */
  if (mpz_cmp (root1, root2) != 0 || mpz_cmp (x2, temp2) != 0 || mpz_cmpabs (temp, x2) > 0 || res == mpz_cmp_ui (rem2, 0))
    {
      fprintf (stderr, "ERROR after test %d\n", i);
      debug_mp (x2, 10);
      debug_mp (root1, 10);
      debug_mp (root2, 10);
      fprintf (stderr, "nth: %lu ,res: %i\n", nth, res);
      abort ();
    }

  if (nth > 1 && mpz_cmp_ui (temp, 1L) > 0 && ! mpz_perfect_power_p (temp))
    {
      fprintf (stderr, "ERROR in mpz_perfect_power_p after test %d\n", i);
      debug_mp (temp, 10);
      debug_mp (root1, 10);
      fprintf (stderr, "nth: %lu\n", nth);
      abort ();
    }

  if (nth <= 10000 && mpz_sgn(x2) > 0)		/* skip too expensive test */
    {
      mpz_add_ui (temp2, root1, 1L);
      mpz_pow_ui (temp2, temp2, nth);
      MPZ_CHECK_FORMAT (temp2);

      /* Is square of (result + 1) <= argument?  */
      if (mpz_cmp (temp2, x2) <= 0)
	{
	  fprintf (stderr, "ERROR after test %d\n", i);
	  debug_mp (x2, 10);
	  debug_mp (root1, 10);
	  fprintf (stderr, "nth: %lu\n", nth);
	  abort ();
	}
    }

  mpz_clear (root2);
  mpz_clear (rem2);
  mpz_clear (temp);
  mpz_clear (temp2);
}

int
main (int argc, char **argv)
{
  mpz_t x2;
  mpz_t root1;
  mp_size_t x2_size;
  int i, res;
  int reps = 500;
  unsigned long nth;
  gmp_randstate_ptr rands;
  mpz_t bs;
  unsigned long bsi, size_range;

  tests_start ();
  TESTS_REPS (reps, argv, argc);

  rands = RANDS;

  mpz_init (bs);

  mpz_init (x2);
  mpz_init (root1);

  /* This triggers a gcc 4.3.2 bug */
  mpz_set_str (x2, "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff80000000000000000000000000000000000000000000000000000000000000002", 16);
  res = mpz_root (root1, x2, 2);
  check_one (root1, x2, 2, res, -1);

  for (i = 0; i < reps; i++)
    {
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 17 + 2;

      mpz_urandomb (bs, rands, size_range);
      x2_size = mpz_get_ui (bs) + 10;
      mpz_rrandomb (x2, rands, x2_size);

      mpz_urandomb (bs, rands, 15);
      nth = mpz_getlimbn (bs, 0) % mpz_sizeinbase (x2, 2) + 2;

      res = mpz_root (root1, x2, nth);

      mpz_urandomb (bs, rands, 4);
      bsi = mpz_get_ui (bs);
      if ((bsi & 1) != 0)
	{
	  /* With 50% probability, set x2 near a perfect power.  */
	  mpz_pow_ui (x2, root1, nth);
	  if ((bsi & 2) != 0)
	    {
	      mpz_sub_ui (x2, x2, bsi >> 2);
	      mpz_abs (x2, x2);
	    }
	  else
	    mpz_add_ui (x2, x2, bsi >> 2);
	  res = mpz_root (root1, x2, nth);
	}

      check_one (root1, x2, nth, res, i);

      if (((nth & 1) != 0) && ((bsi & 2) != 0))
	{
	  mpz_neg (x2, x2);
	  mpz_neg (root1, root1);
	  check_one (root1, x2, nth, res, i);
	}
    }

  mpz_clear (bs);
  mpz_clear (x2);
  mpz_clear (root1);

  tests_end ();
  exit (0);
}

void
debug_mp (mpz_t x, int base)
{
  mpz_out_str (stderr, base, x); fputc ('\n', stderr);
}
