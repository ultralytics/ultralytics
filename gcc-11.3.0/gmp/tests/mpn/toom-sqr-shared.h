/* Test for various Toom squaring functions.

Copyright 2009, 2012 Free Software Foundation, Inc.

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

/* Main file is expected to define mpn_toomN_mul, mpn_toomN_sqr_itch,
 * MIN_AN, MAX_AN and then include this file. */

#ifndef COUNT
#define COUNT 2000
#endif

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

  if (MAX_AN > MIN_AN) {
    rands = RANDS;

    ap = TMP_ALLOC_LIMBS (MAX_AN);
    refp = TMP_ALLOC_LIMBS (MAX_AN * 2);
    pp = 1 + TMP_ALLOC_LIMBS (MAX_AN * 2 + 2);
    scratch
      = 1+TMP_ALLOC_LIMBS (mpn_toomN_sqr_itch (MAX_AN) + 2);

    for (test = 0; test < count; test++)
      {
	mp_size_t an;
	mp_size_t itch;
	mp_limb_t p_before, p_after, s_before, s_after;

	an = MIN_AN
	  + gmp_urandomm_ui (rands, MAX_AN - MIN_AN);

	mpn_random2 (ap, an);
	mpn_random2 (pp-1, an * 2 + 2);
	p_before = pp[-1];
	p_after = pp[an * 2];

	itch = mpn_toomN_sqr_itch (an);
	ASSERT_ALWAYS (itch <= mpn_toomN_sqr_itch (MAX_AN));
	mpn_random2 (scratch-1, itch+2);
	s_before = scratch[-1];
	s_after = scratch[itch];

	mpn_toomN_sqr (pp, ap, an, scratch);
	refmpn_mul (refp, ap, an, ap, an);
	if (pp[-1] != p_before || pp[an * 2] != p_after
	    || scratch[-1] != s_before || scratch[itch] != s_after
	    || mpn_cmp (refp, pp, an * 2) != 0)
	  {
	    printf ("ERROR in test %d, an = %d\n",
		    test, (int) an);
	    if (pp[-1] != p_before)
	      {
		printf ("before pp:"); mpn_dump (pp -1, 1);
		printf ("keep:   "); mpn_dump (&p_before, 1);
	      }
	    if (pp[an * 2] != p_after)
	      {
		printf ("after pp:"); mpn_dump (pp + an * 2, 1);
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
	    mpn_dump (ap, an);
	    mpn_dump (pp, an * 2);
	    mpn_dump (refp, an * 2);

	    abort();
	  }
      }
    TMP_FREE;
  }

  tests_end ();
  return 0;
}
