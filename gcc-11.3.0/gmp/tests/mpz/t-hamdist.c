/* Test mpz_hamdist.

Copyright 2001, 2002 Free Software Foundation, Inc.

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
check_twobits (void)
{
  unsigned long  i, j, got, want;
  mpz_t  x, y;

  mpz_init (x);
  mpz_init (y);
  for (i = 0; i < 5 * GMP_NUMB_BITS; i++)
    {
      for (j = 0; j < 5 * GMP_NUMB_BITS; j++)
        {
          mpz_set_ui (x, 0L);
          mpz_setbit (x, i);
          mpz_set_ui (y, 0L);
          mpz_setbit (y, j);

          want = 2 * (i != j);
          got = mpz_hamdist (x, y);
          if (got != want)
            {
              printf    ("mpz_hamdist wrong on 2 bits pos/pos\n");
            wrong:
              printf    ("  i    %lu\n", i);
              printf    ("  j    %lu\n", j);
              printf    ("  got  %lu\n", got);
              printf    ("  want %lu\n", want);
              mpz_trace ("  x   ", x);
              mpz_trace ("  y   ", y);
              abort();
            }

          mpz_neg (x, x);
          mpz_neg (y, y);
          want = ABS ((long) (i-j));
          got = mpz_hamdist (x, y);
          if (got != want)
            {
              printf    ("mpz_hamdist wrong on 2 bits neg/neg\n");
              goto wrong;
            }
        }

    }
  mpz_clear (x);
  mpz_clear (y);
}


void
check_rand (void)
{
  gmp_randstate_ptr  rands = RANDS;
  unsigned long  got, want;
  int    i;
  mpz_t  x, y;

  mpz_init (x);
  mpz_init (y);

  for (i = 0; i < 2000; i++)
    {
      mpz_erandomb (x, rands, 6 * GMP_NUMB_BITS);
      mpz_negrandom (x, rands);
      mpz_mul_2exp (x, x, urandom() % (4 * GMP_NUMB_BITS));

      mpz_erandomb (y, rands, 6 * GMP_NUMB_BITS);
      mpz_negrandom (y, rands);
      mpz_mul_2exp (y, y, urandom() % (4 * GMP_NUMB_BITS));

      want = refmpz_hamdist (x, y);
      got = mpz_hamdist (x, y);
      if (got != want)
        {
          printf    ("mpz_hamdist wrong on random\n");
          printf    ("  got  %lu\n", got);
          printf    ("  want %lu\n", want);
          mpz_trace ("  x   ", x);
          mpz_trace ("  y   ", y);
          abort();
        }
    }
  mpz_clear (x);
  mpz_clear (y);
}

int
main (void)
{
  tests_start ();
  mp_trace_base = -16;

  check_twobits ();
  check_rand ();

  tests_end ();
  exit (0);
}
