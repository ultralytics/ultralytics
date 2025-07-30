/* Test count_leading_zeros and count_trailing_zeros.

Copyright 2001-2003 Free Software Foundation, Inc.

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
#include "longlong.h"
#include "tests.h"

void
check_clz (int want, mp_limb_t n)
{
  int  got;
  count_leading_zeros (got, n);
  if (got != want)
    {
      printf        ("count_leading_zeros wrong\n");
      mp_limb_trace ("  n    ", n);
      printf        ("  want %d\n", want);
      printf        ("  got  %d\n", got);
      abort ();
    }
}

void
check_ctz (int want, mp_limb_t n)
{
  int  got;
  count_trailing_zeros (got, n);
  if (got != want)
    {
      printf ("count_trailing_zeros wrong\n");
      mpn_trace ("  n    ", &n, (mp_size_t) 1);
      printf    ("  want %d\n", want);
      printf    ("  got  %d\n", got);
      abort ();
    }
}

void
check_various (void)
{
  int        i;

#ifdef COUNT_LEADING_ZEROS_0
  check_clz (COUNT_LEADING_ZEROS_0, CNST_LIMB(0));
#endif

  for (i=0; i < GMP_LIMB_BITS; i++)
    {
      check_clz (i, CNST_LIMB(1) << (GMP_LIMB_BITS-1-i));
      check_ctz (i, CNST_LIMB(1) << i);

      check_ctz (i, MP_LIMB_T_MAX << i);
      check_clz (i, MP_LIMB_T_MAX >> i);
    }
}


int
main (int argc, char *argv[])
{
  tests_start ();
  mp_trace_base = 16;

  check_various ();

  tests_end ();
  exit (0);
}
