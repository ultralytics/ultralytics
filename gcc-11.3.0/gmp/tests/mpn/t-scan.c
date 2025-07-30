/* Test mpn_scan0 and mpn_scan1.

Copyright 2002 Free Software Foundation, Inc.

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


#define SIZE  ((mp_size_t) 3)
mp_limb_t  x[SIZE+1];

void
check (void)
{
  unsigned long  i, got, want;

  x[SIZE] = 1;
  for (i = 0; i < SIZE*GMP_NUMB_BITS; i++)
    {
      got = refmpn_scan1 (x, i);
      want = mpn_scan1 (x, i);
      if (got != want)
        {
          printf ("mpn_scan1\n");
          printf ("  i     %lu\n", i);
          printf ("  got   %lu\n", got);
          printf ("  want  %lu\n", want);
          mpn_trace ("  x    ", x, SIZE);
          abort ();
        }
    }

  x[SIZE] = 0;
  for (i = 0; i < SIZE*GMP_NUMB_BITS; i++)
    {
      got = refmpn_scan0 (x, i);
      want = mpn_scan0 (x, i);
      if (got != want)
        {
          printf ("mpn_scan0\n");
          printf ("  i     %lu\n", i);
          printf ("  got   %lu\n", got);
          printf ("  want  %lu\n", want);
          mpn_trace ("  x    ", x, SIZE);
          abort ();
        }
    }
}

void
check_twobits (void)
{
#define TWOBITS(a, b) \
  ((CNST_LIMB(1) << (a)) | (CNST_LIMB(1) << (b)))

  refmpn_zero (x, SIZE);
  x[0] = TWOBITS (1, 0);
  check ();

  refmpn_zero (x, SIZE);
  x[0] = TWOBITS (GMP_NUMB_BITS-1, 1);
  check ();

  refmpn_zero (x, SIZE);
  x[0] = CNST_LIMB(1);
  x[1] = CNST_LIMB(1);
  check ();

  refmpn_zero (x, SIZE);
  x[0] = CNST_LIMB(1) << (GMP_NUMB_BITS-1);
  x[1] = CNST_LIMB(1);
  check ();

  refmpn_zero (x, SIZE);
  x[1] = TWOBITS (1, 0);
  check ();

  refmpn_zero (x, SIZE);
  x[1] = CNST_LIMB(1);
  x[2] = CNST_LIMB(1);
  check ();
}

/* This is unused, it takes too long, especially on 64-bit systems. */
void
check_twobits_exhaustive (void)
{
  unsigned long  i, j;

  for (i = 0; i < GMP_NUMB_BITS * SIZE; i++)
    {
      for (j = 0; j < GMP_NUMB_BITS * SIZE; j++)
        {
          refmpn_zero (x, SIZE);
          refmpn_setbit (x, i);
          refmpn_setbit (x, j);
          check ();
        }
    }
}

void
check_rand (void)
{
  int  i;

  for (i = 0; i < 100; i++)
    {
      refmpn_random2 (x, SIZE);
      check ();
    }
}

int
main (void)
{
  mp_trace_base = -16;
  tests_start ();

  check_twobits ();
  check_rand ();

  tests_end ();
  exit (0);
}
