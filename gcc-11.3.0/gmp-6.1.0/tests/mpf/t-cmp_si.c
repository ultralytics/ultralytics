/* Test mpf_cmp_si and mpf_cmp_z.

Copyright 2000, 2001, 2004, 2015 Free Software Foundation, Inc.

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

#define SGN(x)       ((x) < 0 ? -1 : (x) == 0 ? 0 : 1)

void
check_data (void)
{
  static const struct {
    int         a_base;
    const char  *a;
    const char  *b;
    int         want;
  } data[] = {
    { 10, "0",  "1", -1 },
    { 10, "0",  "0",  0 },
    { 10, "0", "-1",  1 },

    { 10, "1",  "1", 0 },
    { 10, "1",  "0", 1 },
    { 10, "1", "-1", 1 },

    { 10, "-1",  "1", -1 },
    { 10, "-1",  "0", -1 },
    { 10, "-1", "-1", 0 },

    { 10, "1.5", "2", -1 },
    { 10, "1.5", "1",  1 },
    { 10, "0.5", "1", -1 },

    { 10, "-1.5", "-2",  1 },
    { 10, "-1.5", "-1", -1 },
    { 10, "-0.5", "-1",  1 },

    { 16,         "0", "-0x80000000",  1 },
    { 16,  "80000000", "-0x80000000",  1 },
    { 16,  "80000001", "-0x80000000",  1 },
    { 16, "-80000000", "-0x80000000",  0 },
    { 16, "-80000001", "-0x80000000", -1 },
    { 16, "-FF0080000001", "-0x80000000", -1 },

    { 16,                 "0", "-0x8000000000000000",  1 },
    { 16,  "8000000000000000", "-0x8000000000000000",  1 },
    { 16,  "8000000000000001", "-0x8000000000000000",  1 },
    { 16, "-8000000000000000", "-0x8000000000000000",  0 },
    { 16, "-8000000000000000.1", "-0x8000000000000000", -1 },
    { 16, "-FF008000000000000001", "-0x8000000000000000", -1 },

    { 16,                 "0", "-0x876543210FEDCBA9876543210000000",  1 },
    { 16,  "876543210FEDCBA9876543210000000", "-0x876543210FEDCBA9876543210000000",  1 },
    { 16,  "876543210FEDCBA9876543210000001", "-0x876543210FEDCBA9876543210000000",  1 },
    { 16, "-876543210FEDCBA9876543210000000", "-0x876543210FEDCBA9876543210000000",  0 },
    { 16, "-876543210FEDCBA9876543210000000.1", "-0x876543210FEDCBA9876543210000000", -1 },
    { 16, "-FF00876543210FEDCBA9876543210000000", "-0x876543210FEDCBA9876543210000000", -1 },
  };

  mpf_t  a;
  mpz_t  bz;
  long   b;
  int    got;
  int    i;

  mpf_init2 (a, 128);
  mpz_init (bz);
  for (i = 0; i < numberof (data); i++)
    {
      mpf_set_str_or_abort (a, data[i].a, data[i].a_base);
      mpz_set_str_or_abort (bz, data[i].b, 0);

      if (mpz_fits_slong_p (bz))
        {
          b = mpz_get_si (bz);
          got = mpf_cmp_si (a, b);
          if (SGN (got) != data[i].want)
            {
              printf ("mpf_cmp_si wrong on data[%d]\n", i);
              printf ("  a="); mpf_out_str (stdout, 10, 0, a);
              printf (" (%s)\n", data[i].a);
              printf ("  b=%ld (%s)\n", b, data[i].b);
              printf ("  got=%d\n", got);
              printf ("  want=%d\n", data[i].want);
              abort();
            }
        }

      got = mpf_cmp_z (a, bz);
      if (SGN (got) != data[i].want)
      	{
	  b = mpz_get_si (bz);
      	  printf ("mpf_cmp_z wrong on data[%d]\n", i);
      	  printf ("  a="); mpf_out_str (stdout, 10, 0, a);
      	  printf (" (%s)\n", data[i].a);
      	  printf ("  b=%ld (%s)\n", b, data[i].b);
      	  printf ("  got=%d\n", got);
      	  printf ("  want=%d\n", data[i].want);
      	  abort();
      	}
    }

  mpf_clear (a);
  mpz_clear (bz);
}

int
main (void)
{
  tests_start ();

  check_data ();

  tests_end ();
  exit (0);
}
