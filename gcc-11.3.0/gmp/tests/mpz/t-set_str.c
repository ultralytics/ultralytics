/* Test mpz_set_str.

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
check_one (mpz_srcptr want, int fail, int base, const char *str)
{
  mpz_t   got;

  MPZ_CHECK_FORMAT (want);
  mp_trace_base = (base == 0 ? 16 : base);

  mpz_init (got);

  if (mpz_set_str (got, str, base) != fail)
    {
      printf ("mpz_set_str unexpectedly failed\n");
      printf ("  base %d\n", base);
      printf ("  str  \"%s\"\n", str);
      abort ();
    }
  MPZ_CHECK_FORMAT (got);

  if (fail == 0 && mpz_cmp (got, want) != 0)
    {
      printf ("mpz_set_str wrong\n");
      printf ("  base %d\n", base);
      printf ("  str  \"%s\"\n", str);
      mpz_trace ("got ", got);
      mpz_trace ("want", want);
      abort ();
    }

  mpz_clear (got);
}

void
check_samples (void)
{
  mpz_t  z;

  mpz_init (z);

  mpz_set_ui (z, 0L);
  check_one (z, 0, 0, "0 ");
  check_one (z, 0, 0, " 0 0 0 ");
  check_one (z, 0, 0, " -0B 0 ");
  check_one (z, 0, 0, "  0X 0 ");
  check_one (z, 0, 10, "0 ");
  check_one (z, 0, 10, "-0   ");
  check_one (z, 0, 10, " 0 000 000    ");

  mpz_set_ui (z, 123L);
  check_one (z, 0, 0, "123 ");
  check_one (z, 0, 0, "123    ");
  check_one (z, 0, 0, "0173   ");
  check_one (z, 0, 0, " 0b 1 11 10 11  ");
  check_one (z, 0, 0, " 0x 7b ");
  check_one (z, 0, 0, "0x7B");
  check_one (z, 0, 10, "123 ");
  check_one (z, 0, 10, "123    ");
  check_one (z, 0, 0, " 123 ");
  check_one (z, 0, 0, "  123    ");
  check_one (z, 0, 10, "  0000123 ");
  check_one (z, 0, 10, "  123    ");
  check_one (z,-1, 10, "1%");
  check_one (z,-1, 0, "3!");
  check_one (z,-1, 0, "0123456789");
  check_one (z,-1, 0, "13579BDF");
  check_one (z,-1, 0, "0b0102");
  check_one (z,-1, 0, "0x010G");
  check_one (z,-1, 37,"0x010G");
  check_one (z,-1, 99,"0x010G");

  mpz_clear (z);
}

int
main (void)
{
  tests_start ();

  check_samples ();

  tests_end ();
  exit (0);
}
