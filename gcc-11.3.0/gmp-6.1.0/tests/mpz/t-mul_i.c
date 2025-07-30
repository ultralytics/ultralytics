/* Test mpz_mul_ui and mpz_mul_si.

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


mpz_t got, want, x;

void
compare_si (long y)
{
  if (mpz_cmp (got, want) != 0)
    {
      printf    ("mpz_mul_si wrong\n");
      mpz_trace ("  x", x);
      printf    ("  y=%ld (0x%lX)\n", y, y);
      mpz_trace ("  got ", got);
      mpz_trace ("  want", want);
      abort ();
    }
}

void
compare_ui (unsigned long y)
{
  if (mpz_cmp (got, want) != 0)
    {
      printf    ("mpz_mul_ui wrong\n");
      mpz_trace ("  x", x);
      printf    ("  y=%lu (0x%lX)\n", y, y);
      mpz_trace ("  got ", got);
      mpz_trace ("  want", want);
      abort ();
    }
}

void
check_samples (void)
{
  {
    long  y;

    mpz_set_ui (x, 1L);
    y = 0;
    mpz_mul_si (got, x, y);
    mpz_set_si (want, y);
    compare_si (y);

    mpz_set_ui (x, 1L);
    y = 1;
    mpz_mul_si (got, x, y);
    mpz_set_si (want, y);
    compare_si (y);

    mpz_set_ui (x, 1L);
    y = -1;
    mpz_mul_si (got, x, y);
    mpz_set_si (want, y);
    compare_si (y);

    mpz_set_ui (x, 1L);
    y = LONG_MIN;
    mpz_mul_si (got, x, y);
    mpz_set_si (want, y);
    compare_si (y);

    mpz_set_ui (x, 1L);
    y = LONG_MAX;
    mpz_mul_si (got, x, y);
    mpz_set_si (want, y);
    compare_si (y);
  }

  {
    unsigned long y;

    mpz_set_ui (x, 1L);
    y = 0;
    mpz_mul_ui (got, x, y);
    mpz_set_ui (want, y);
    compare_ui (y);

    mpz_set_ui (x, 1L);
    y = 1;
    mpz_mul_ui (got, x, y);
    mpz_set_ui (want, y);
    compare_ui (y);

    mpz_set_ui (x, 1L);
    y = ULONG_MAX;
    mpz_mul_ui (got, x, y);
    mpz_set_ui (want, y);
    compare_ui (y);
  }
}

int
main (int argc, char **argv)
{
  tests_start ();

  mpz_init (x);
  mpz_init (got);
  mpz_init (want);

  check_samples ();

  mpz_clear (x);
  mpz_clear (got);
  mpz_clear (want);

  tests_end ();
  exit (0);
}
