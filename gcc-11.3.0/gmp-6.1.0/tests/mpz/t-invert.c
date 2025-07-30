/* Test mpz_invert.

Copyright 1991, 1993, 1994, 1996, 1997, 2000-2005, 2008, 2009, 2012, 2014 Free
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

int
main (int argc, char **argv)
{
  mpz_t a, m, ainv, t;
  int test, r;
  gmp_randstate_ptr rands;
  mpz_t bs;
  unsigned long bsi, size_range;
  int reps = 1000;

  tests_start ();
  TESTS_REPS (reps, argv, argc);

  rands = RANDS;

  mpz_init (bs);
  mpz_init (a);
  mpz_init (m);
  mpz_init (ainv);
  mpz_init (t);

  for (test = 0; test < reps; test++)
    {
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 16 + 2;

      mpz_urandomb (bs, rands, size_range);
      mpz_rrandomb (a, rands, mpz_get_ui (bs));
      do {
	mpz_urandomb (bs, rands, size_range);
	mpz_rrandomb (m, rands, mpz_get_ui (bs));
      } while (mpz_sgn (m) == 0);

      mpz_urandomb (bs, rands, 8);
      bsi = mpz_get_ui (bs);

      if ((bsi & 1) != 0)
	mpz_neg (a, a);
      if ((bsi & 2) != 0)
	mpz_neg (m, m);

      r = mpz_invert (ainv, a, m);
      if (r != 0)
	{
	  MPZ_CHECK_FORMAT (ainv);

	  if (mpz_cmp_ui (ainv, 0) < 0 || mpz_cmpabs (ainv, m) >= 0)
	    {
	      fprintf (stderr, "ERROR in test %d\n", test);
	      gmp_fprintf (stderr, "Inverse out of range.\n");
	      gmp_fprintf (stderr, "a = %Zx\n", a);
	      gmp_fprintf (stderr, "1/a = %Zx\n", ainv);
	      gmp_fprintf (stderr, "m = %Zx\n", m);
	      abort ();
	    }

	  mpz_mul (t, ainv, a);
	  mpz_mod (t, t, m);

	  if (mpz_cmp_ui (t, mpz_cmpabs_ui (m, 1) != 0) != 0)
	    {
	      fprintf (stderr, "ERROR in test %d\n", test);
	      gmp_fprintf (stderr, "a^(-1)*a != 1 (mod m)\n");
	      gmp_fprintf (stderr, "a = %Zx\n", a);
	      gmp_fprintf (stderr, "m = %Zx\n", m);
	      abort ();
	    }
	}
      else /* Inverse deos not exist */
	{
	  if (mpz_cmpabs_ui (m, 1) <= 0)
	    continue; /* OK */

	  mpz_gcd (t, a, m);
	  if (mpz_cmp_ui (t, 1) == 0)
	    {
	      fprintf (stderr, "ERROR in test %d\n", test);
	      gmp_fprintf (stderr, "Inverse exists, but was not found.\n");
	      gmp_fprintf (stderr, "a = %Zx\n", a);
	      gmp_fprintf (stderr, "m = %Zx\n", m);
	      abort ();
	    }
	}
    }

  mpz_clear (bs);
  mpz_clear (a);
  mpz_clear (m);
  mpz_clear (ainv);
  mpz_clear (t);

  tests_end ();
  exit (0);
}
