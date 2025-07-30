/* test mpz_divisible_2exp_p */

/*
Copyright 2001 Free Software Foundation, Inc.

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
check_one (mpz_srcptr a, unsigned long d, int want)
{
  int   got;

  got = (mpz_divisible_2exp_p (a, d) != 0);
  if (want != got)
    {
      printf ("mpz_divisible_2exp_p wrong\n");
      printf ("   expected %d got %d\n", want, got);
      mpz_trace ("   a", a);
      printf    ("   d=%lu\n", d);
      mp_trace_base = -16;
      mpz_trace ("   a", a);
      printf    ("   d=0x%lX\n", d);
      abort ();
    }
}

void
check_data (void)
{
  static const struct {
    const char    *a;
    unsigned long d;
    int           want;

  } data[] = {

    { "0", 0, 1 },
    { "0", 1, 1 },
    { "0", 2, 1 },
    { "0", 3, 1 },

    { "1", 0, 1 },
    { "1", 1, 0 },
    { "1", 2, 0 },
    { "1", 3, 0 },
    { "1", 10000, 0 },

    { "4", 0, 1 },
    { "4", 1, 1 },
    { "4", 2, 1 },
    { "4", 3, 0 },
    { "4", 4, 0 },
    { "4", 10000, 0 },

    { "0x80000000", 31, 1 },
    { "0x80000000", 32, 0 },
    { "0x80000000", 64, 0 },

    { "0x100000000", 32, 1 },
    { "0x100000000", 33, 0 },
    { "0x100000000", 64, 0 },

    { "0x8000000000000000", 63, 1 },
    { "0x8000000000000000", 64, 0 },
    { "0x8000000000000000", 128, 0 },

    { "0x10000000000000000", 64, 1 },
    { "0x10000000000000000", 65, 0 },
    { "0x10000000000000000", 128, 0 },
    { "0x10000000000000000", 256, 0 },

    { "0x10000000000000000100000000", 32, 1 },
    { "0x10000000000000000100000000", 33, 0 },
    { "0x10000000000000000100000000", 64, 0 },

    { "0x1000000000000000010000000000000000", 64, 1 },
    { "0x1000000000000000010000000000000000", 65, 0 },
    { "0x1000000000000000010000000000000000", 128, 0 },
    { "0x1000000000000000010000000000000000", 256, 0 },
    { "0x1000000000000000010000000000000000", 1024, 0 },

  };

  mpz_t   a, d;
  int     i;

  mpz_init (a);
  mpz_init (d);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (a, data[i].a, 0);
      check_one (a, data[i].d, data[i].want);

      mpz_neg (a, a);
      check_one (a, data[i].d, data[i].want);
    }

  mpz_clear (a);
  mpz_clear (d);
}

int
main (int argc, char *argv[])
{
  tests_start ();

  check_data ();

  tests_end ();
  exit (0);
}
