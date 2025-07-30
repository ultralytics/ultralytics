/* Test mpz_gcd_ui.

Copyright 2003 Free Software Foundation, Inc.

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

/* Check mpz_gcd_ui doesn't try to return a value out of range.
   This was wrong in gmp 4.1.2 with a long long limb.  */
static void
check_ui_range (void)
{
  unsigned long  got;
  mpz_t  x;
  int  i;

  mpz_init_set_ui (x, ULONG_MAX);

  for (i = 0; i < 20; i++)
    {
      mpz_mul_2exp (x, x, 1L);
      got = mpz_gcd_ui (NULL, x, 0L);
      if (got != 0)
        {
          printf ("mpz_gcd_ui (ULONG_MAX*2^%d, 0)\n", i);
          printf ("   return %#lx\n", got);
          printf ("   should be 0\n");
          abort ();
        }
    }

  mpz_clear (x);
}

int
main (void)
{
  tests_start ();

  check_ui_range ();

  tests_end ();
  exit (0);
}
