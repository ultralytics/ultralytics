/* Test mpz_odd_p and mpz_even_p.

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

void
check_data (void)
{
  static const struct {
    const char  *n;
    int          odd, even;
  } data[] = {
    {   "0", 0, 1 },
    {   "1", 1, 0 },
    {   "2", 0, 1 },
    {   "3", 1, 0 },
    {   "4", 0, 1 },

    {  "-4", 0, 1 },
    {  "-3", 1, 0 },
    {  "-2", 0, 1 },
    {  "-1", 1, 0 },

    {  "0x1000000000000000000000000000000000000000000000000000", 0, 1 },
    {  "0x1000000000000000000000000000000000000000000000000001", 1, 0 },
    {  "0x1000000000000000000000000000000000000000000000000002", 0, 1 },
    {  "0x1000000000000000000000000000000000000000000000000003", 1, 0 },

    { "-0x1000000000000000000000000000000000000000000000000004", 0, 1 },
    { "-0x1000000000000000000000000000000000000000000000000003", 1, 0 },
    { "-0x1000000000000000000000000000000000000000000000000002", 0, 1 },
    { "-0x1000000000000000000000000000000000000000000000000001", 1, 0 },
  };

  mpz_t  n;
  int    i;

  mpz_init (n);
  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (n, data[i].n, 0);

      if ((mpz_odd_p (n) != 0) != data[i].odd)
	{
	  printf ("mpz_odd_p wrong on data[%d]\n", i);
	  abort();
	}

      if ((mpz_even_p (n) != 0) != data[i].even)
	{
	  printf ("mpz_even_p wrong on data[%d]\n", i);
	  abort();
	}
    }

  mpz_clear (n);
}

int
main (void)
{
  tests_start ();

  check_data ();

  tests_end ();
  exit (0);
}
