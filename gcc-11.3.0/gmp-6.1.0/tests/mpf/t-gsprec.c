/* Test mpf_get_prec and mpf_set_prec.

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
check_consistency (void)
{
  mpf_t  x;
  unsigned long  i, a, b;

  mpf_init (x);

  for (i = 1; i < 2000; i++)
    {
      mpf_set_prec (x, i);
      a = mpf_get_prec (x);
      mpf_set_prec (x, a);
      b = mpf_get_prec (x);
      if (a != b)
        {
          printf ("mpf_get_prec / mpf_set_prec inconsistent\n");
          printf ("   set %lu gives %lu, but then set %lu gives %lu\n",
                  i, a,
                  a, b);
          abort ();
        }
    }

  mpf_clear (x);
}

int
main (void)
{
  tests_start ();

  check_consistency ();

  tests_end ();
  exit (0);
}
