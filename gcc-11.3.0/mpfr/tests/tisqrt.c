/* Test file for __gmpfr_isqrt and __gmpfr_cuberoot internal functions.

Copyright 2007-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include <stdio.h>
#include <stdlib.h>

#include "mpfr-test.h"

static void
tst_isqrt (unsigned long n, unsigned long r)
{
  unsigned long i;

  i = __gmpfr_isqrt (n);
  if (i != r)
    {
      printf ("Error in __gmpfr_isqrt (%lu): got %lu instead of %lu\n",
              n, i, r);
      exit (1);
    }
}

static void
tst_icbrt (unsigned long n, unsigned long r)
{
  unsigned long i;

  i = __gmpfr_cuberoot (n);
  if (i != r)
    {
      printf ("Error in __gmpfr_cuberoot (%lu): got %lu instead of %lu\n",
              n, i, r);
      exit (1);
    }
}

int
main (void)
{
  unsigned long c, i;

  tests_start_mpfr ();

  tst_isqrt (0, 0);
  tst_isqrt (1, 1);
  tst_isqrt (2, 1);
  for (i = 2; i <= 65535; i++)
    {
      tst_isqrt (i * i - 1, i - 1);
      tst_isqrt (i * i, i);
    }
  tst_isqrt (4294967295UL, 65535);

  tst_icbrt (0, 0);
  tst_icbrt (1, 1);
  tst_icbrt (2, 1);
  tst_icbrt (3, 1);
  for (i = 2; i <= 1625; i++)
    {
      c = i * i * i;
      tst_icbrt (c - 4, i - 1);
      tst_icbrt (c - 3, i - 1);
      tst_icbrt (c - 2, i - 1);
      tst_icbrt (c - 1, i - 1);
      tst_icbrt (c, i);
      tst_icbrt (c + 1, i);
      tst_icbrt (c + 2, i);
      tst_icbrt (c + 3, i);
      tst_icbrt (c + 4, i);
    }
  tst_icbrt (4294967295UL, 1625);

  tests_end_mpfr ();
  return 0;
}
