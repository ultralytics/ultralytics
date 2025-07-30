/* Test mpz_divexact_ui.

Copyright 1996, 2001 Free Software Foundation, Inc.

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
check_random (int argc, char *argv[])
{
  gmp_randstate_ptr rands = RANDS;
  int    reps = 500000;
  mpz_t  a, q, got;
  int    i, qneg;
  unsigned long  d;

  if (argc == 2)
    reps = atoi (argv[1]);

  mpz_init (a);
  mpz_init (q);
  mpz_init (got);

  for (i = 0; i < reps; i++)
    {
      do
	d = (unsigned long) urandom();
      while (d == 0);
      mpz_erandomb (q, rands, 512);
      mpz_mul_ui (a, q, d);

      for (qneg = 0; qneg <= 1; qneg++)
        {
          mpz_divexact_ui (got, a, d);
          MPZ_CHECK_FORMAT (got);
          if (mpz_cmp (got, q) != 0)
            {
              printf    ("mpz_divexact_ui wrong\n");
              mpz_trace ("    a", a);
              printf    ("    d=%lu\n", d);
              mpz_trace ("    q", q);
              mpz_trace ("  got", got);
              abort ();
            }

          mpz_neg (q, q);
          mpz_neg (a, a);
        }

    }

  mpz_clear (a);
  mpz_clear (q);
  mpz_clear (got);
}


int
main (int argc, char **argv)
{
  tests_start ();

  check_random (argc, argv);

  tests_end ();
  exit (0);
}
