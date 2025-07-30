/* Test mpz_cmp_si.

Copyright 2000, 2001 Free Software Foundation, Inc.

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
    const char  *a, *b;
    int         want;
  } data[] = {
    { "0",  "1", -1 },
    { "0",  "0",  0 },
    { "0", "-1",  1 },

    { "1",  "1", 0 },
    { "1",  "0", 1 },
    { "1", "-1", 1 },

    { "-1",  "1", -1 },
    { "-1",  "0", -1 },
    { "-1", "-1", 0 },

    {           "0", "-0x80000000",  1 },
    {  "0x80000000", "-0x80000000",  1 },
    {  "0x80000001", "-0x80000000",  1 },
    { "-0x80000000", "-0x80000000",  0 },
    { "-0x80000001", "-0x80000000", -1 },

    {                   "0", "-0x8000000000000000",  1 },
    {  "0x8000000000000000", "-0x8000000000000000",  1 },
    {  "0x8000000000000001", "-0x8000000000000000",  1 },
    { "-0x8000000000000000", "-0x8000000000000000",  0 },
    { "-0x8000000000000001", "-0x8000000000000000", -1 },
  };

  mpz_t  a, bz;
  long   b;
  int    got;
  int    i;

  mpz_init (a);
  mpz_init (bz);
  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (a, data[i].a, 0);
      mpz_set_str_or_abort (bz, data[i].b, 0);

      if (mpz_fits_slong_p (bz))
	{
	  b = mpz_get_si (bz);
	  got = mpz_cmp_si (a, b);
	  if (SGN (got) != data[i].want)
	    {
	      printf ("mpz_cmp_si wrong on data[%d]\n", i);
	      printf ("  a="); mpz_out_str (stdout, 10, a); printf ("\n");
	      printf ("  b=%ld\n", b);
	      printf ("  got=%d\n", got);
	      printf ("  want=%d\n", data[i].want);
	      abort();
	    }
	}
    }

  mpz_clear (a);
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
