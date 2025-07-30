/* Test for sqrlo function.

Copyright 2009, 2015 Free Software Foundation, Inc.

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
#define SIZE_LOG 10
#endif

#ifndef COUNT
#define COUNT 10000
#endif

#define MAX_N (1L << SIZE_LOG)
#define MIN_N (1)

int
main (int argc, char **argv)
{
  mp_ptr ap, refp, pp, scratch;
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

#define mpn_sqrlo_itch(n) (0)

  ap = TMP_ALLOC_LIMBS (MAX_N);
  refp = TMP_ALLOC_LIMBS (MAX_N * 2);
  pp = 1+TMP_ALLOC_LIMBS (MAX_N + 2);
  scratch
    = 1+TMP_ALLOC_LIMBS (mpn_sqrlo_itch (MAX_N) + 2);

  for (test = 0; test < count; test++)
    {
      unsigned size_min;
      unsigned size_range;
      mp_size_t n;
      mp_size_t itch;
      mp_limb_t p_before, p_after, s_before, s_after;

      for (size_min = 1; (1L << size_min) < MIN_N; size_min++)
	;

      /* We generate an in the MIN_N <= n <= (1 << size_range). */
      size_range = size_min
	+ gmp_urandomm_ui (rands, SIZE_LOG + 1 - size_min);

      n = MIN_N
	+ gmp_urandomm_ui (rands, (1L << size_range) + 1 - MIN_N);

      mpn_random2 (ap, n);
      mpn_random2 (pp-1, n + 2);
      p_before = pp[-1];
      p_after = pp[n];

      itch = mpn_sqrlo_itch (n);
#if 0
      ASSERT_ALWAYS (itch <= mpn_sqrlo_itch (MAX_N));
      mpn_random2 (scratch-1, itch+2);
#endif
      s_before = scratch[-1];
      s_after = scratch[itch];

      mpn_sqrlo (pp, ap, n);
      mpn_sqr (refp, ap, n);
      if (pp[-1] != p_before || pp[n] != p_after
	  || scratch[-1] != s_before || scratch[itch] != s_after
	  || mpn_cmp (refp, pp, n) != 0)
	{
	  printf ("ERROR in test %d, n = %d",
		  test, (int) n);
	  if (pp[-1] != p_before)
	    {
	      printf ("before pp:"); mpn_dump (pp -1, 1);
	      printf ("keep:   "); mpn_dump (&p_before, 1);
	    }
	  if (pp[n] != p_after)
	    {
	      printf ("after pp:"); mpn_dump (pp + n, 1);
	      printf ("keep:   "); mpn_dump (&p_after, 1);
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
	  mpn_dump (ap, n);
	  mpn_dump (pp, n);
	  mpn_dump (refp, n);

	  abort();
	}
    }
  TMP_FREE;
  tests_end ();
  return 0;
}
