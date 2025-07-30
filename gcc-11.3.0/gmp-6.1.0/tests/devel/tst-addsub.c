/* Copyright 1996, 2001 Free Software Foundation, Inc.

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

#define ADD 1
#define SUB 2

#ifndef METHOD
#define METHOD ADD
#endif

#if METHOD == ADD
#define REFCALL refmpn_add_n
#define TESTCALL mpn_add_n
#endif

#if METHOD == SUB
#define REFCALL refmpn_sub_n
#define TESTCALL mpn_sub_n
#endif

#define SIZE 100

int
main (int argc, char **argv)
{
  mp_size_t alloc_size, max_size, size, i, cumul_size;
  mp_ptr s1, s2, dx, dy;
  int s1_align, s2_align, d_align;
  long pass, n_passes;
  mp_limb_t cx, cy;

  max_size = SIZE;
  n_passes = 1000000;

  argc--; argv++;
  if (argc)
    {
      max_size = atol (*argv);
      argc--; argv++;
    }

  alloc_size = max_size + 32;
  s1 = malloc (alloc_size * GMP_LIMB_BYTES);
  s2 = malloc (alloc_size * GMP_LIMB_BYTES);
  dx = malloc (alloc_size * GMP_LIMB_BYTES);
  dy = malloc (alloc_size * GMP_LIMB_BYTES);

  cumul_size = 0;
  for (pass = 0; pass < n_passes; pass++)
    {
      size = random () % max_size + 1;

      cumul_size += size;
      if (cumul_size >= 1000000)
	{
	  cumul_size -= 1000000;
	  printf ("\r%ld", pass); fflush (stdout);
	}
      s1_align = random () % 32;
      s2_align = random () % 32;
      d_align = random () % 32;

      mpn_random2 (s1 + s1_align, size);
      mpn_random2 (s2 + s2_align, size);

      for (i = 0; i < alloc_size; i++)
	dx[i] = dy[i] = i + 0x9876500;

      cx = TESTCALL (dx + d_align, s1 + s1_align, s2 + s2_align, size);
      cy = REFCALL (dy + d_align, s1 + s1_align, s2 + s2_align, size);

      if (cx != cy || mpn_cmp (dx, dy, alloc_size) != 0)
	abort ();
    }

  printf ("%ld passes OK\n", n_passes);
  exit (0);
}
