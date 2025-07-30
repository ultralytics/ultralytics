/* Test for mpn_invert function.

   Contributed to the GNU project by Marco Bodrato.

Copyright 2009 Free Software Foundation, Inc.

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


#include <stdlib.h>
#include <stdio.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

/* Sizes are up to 2^SIZE_LOG limbs */
#ifndef SIZE_LOG
#define SIZE_LOG 12
#endif

#ifndef COUNT
#define COUNT 1000
#endif

#define MAX_N (1L << SIZE_LOG)
#define MIN_N 1


static int
invert_valid (mp_srcptr ip, mp_srcptr dp, mp_size_t n)
{
  mp_ptr tp;
  int cy;
  TMP_DECL;

  TMP_MARK;
  tp = TMP_ALLOC_LIMBS (2*n);

  refmpn_mul (tp, ip, n, dp, n);
  cy  = refmpn_add_n (tp + n, tp + n, dp, n); /* This must not give a carry. */
  cy -= refmpn_add (tp, tp, 2*n, dp, n); /* This must give a carry. */
  TMP_FREE;

  return (cy == -1);
}

/*
  Check the result of the mpn_invert function in the library.
*/

int
main (int argc, char **argv)
{
  mp_ptr ip, dp, scratch;
  int count = COUNT;
  int test;
  gmp_randstate_ptr rands;
  TMP_DECL;
  TMP_MARK;

  if (argc > 1)
    {
      char *end;
      count = strtol (argv[1], &end, 0);
      if (*end || count <= 0)
	{
	  fprintf (stderr, "Invalid test count: %s.\n", argv[1]);
	  return 1;
	}
    }

  tests_start ();
  rands = RANDS;

  dp = TMP_ALLOC_LIMBS (MAX_N);
  ip = 1+TMP_ALLOC_LIMBS (MAX_N + 2);
  scratch
    = 1+TMP_ALLOC_LIMBS (mpn_invert_itch (MAX_N) + 2);

  for (test = 0; test < count; test++)
    {
      unsigned size_min;
      unsigned size_range;
      mp_size_t n;
      mp_size_t itch;
      mp_limb_t i_before, i_after, s_before, s_after;

      for (size_min = 1; (1L << size_min) < MIN_N; size_min++)
	;

      /* We generate an in the MIN_N <= n <= (1 << size_range). */
      size_range = size_min
	+ gmp_urandomm_ui (rands, SIZE_LOG + 1 - size_min);

      n = MIN_N
	+ gmp_urandomm_ui (rands, (1L << size_range) + 1 - MIN_N);

      mpn_random2 (dp, n);

      mpn_random2 (ip-1, n + 2);
      i_before = ip[-1];
      i_after = ip[n];

      itch = mpn_invert_itch (n);
      ASSERT_ALWAYS (itch <= mpn_invert_itch (MAX_N));
      mpn_random2 (scratch-1, itch+2);
      s_before = scratch[-1];
      s_after = scratch[itch];

      dp[n-1] |= GMP_NUMB_HIGHBIT;
      mpn_invert (ip, dp, n, scratch);
      if (ip[-1] != i_before || ip[n] != i_after
	  || scratch[-1] != s_before || scratch[itch] != s_after
	  || ! invert_valid(ip, dp, n))
	{
	  printf ("ERROR in test %d, n = %d\n",
		  test, (int) n);
	  if (ip[-1] != i_before)
	    {
	      printf ("before ip:"); mpn_dump (ip -1, 1);
	      printf ("keep:   "); mpn_dump (&i_before, 1);
	    }
	  if (ip[n] != i_after)
	    {
	      printf ("after ip:"); mpn_dump (ip + n, 1);
	      printf ("keep:   "); mpn_dump (&i_after, 1);
	    }
	  if (scratch[-1] != s_before)
	    {
	      printf ("before scratch:"); mpn_dump (scratch-1, 1);
	      printf ("keep:   "); mpn_dump (&s_before, 1);
	    }
	  if (scratch[itch] != s_after)
	    {
	      printf ("after scratch:"); mpn_dump (scratch + itch, 1);
	      printf ("keep:   "); mpn_dump (&s_after, 1);
	    }
	  mpn_dump (dp, n);
	  mpn_dump (ip, n);

	  abort();
	}
    }
  TMP_FREE;
  tests_end ();
  return 0;
}
