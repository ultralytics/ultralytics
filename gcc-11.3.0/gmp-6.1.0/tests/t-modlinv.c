/* Test binvert_limb.

Copyright 2000-2003 Free Software Foundation, Inc.

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
#include <string.h>
#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"
#include "tests.h"


void
one (mp_limb_t n)
{
  mp_limb_t  inv, prod;

  binvert_limb (inv, n);
  prod = (inv * n) & GMP_NUMB_MASK;
  if (prod != 1)
    {
      printf ("binvert_limb wrong\n");
      mp_limb_trace ("  n       ", n);
      mp_limb_trace ("  got     ", inv);
      mp_limb_trace ("  product ", prod);
      abort ();
    }
}

void
some (void)
{
  int  i;
  for (i = 0; i < 10000; i++)
    one (refmpn_random_limb () | 1);
}

void
all (void)
{
  mp_limb_t  n;

  n = 1;
  do {
    one (n);
    n += 2;
  } while (n != 1);
}


int
main (int argc, char *argv[])
{
  tests_start ();

  if (argc >= 2 && strcmp (argv[1], "-a") == 0)
    {
      /* it's feasible to run all values on a 32-bit limb, but not a 64-bit */
      all ();
    }
  else
    {
      some ();
    }

  tests_end ();
  exit (0);
}
