/* Test mpz_get_d.

Copyright 2002, 2012 Free Software Foundation, Inc.

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
check_onebit (void)
{
  int     i;
  mpz_t   z;
  double  got, want;
  /* FIXME: It'd be better to base this on the float format. */
#if defined (__vax) || defined (__vax__)
  int     limit = 127 - 1;  /* vax fp numbers have limited range */
#else
  int     limit = 512;
#endif

  mpz_init (z);

  mpz_set_ui (z, 1L);
  want = 1.0;

  for (i = 0; i < limit; i++)
    {
      got = mpz_get_d (z);

      if (got != want)
        {
          printf    ("mpz_get_d wrong on 2**%d\n", i);
          mpz_trace ("   z    ", z);
          printf    ("   want  %.20g\n", want);
          printf    ("   got   %.20g\n", got);
          abort();
        }

      mpz_mul_2exp (z, z, 1L);
      want *= 2.0;
    }
  mpz_clear (z);
}


int
main (void)
{
  tests_start ();

  check_onebit ();

  tests_end ();
  exit (0);
}
