/* Test for various Toom functions.

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

/* Main file is expected to define mpn_toomMN_mul,
 * mpn_toomMN_mul_itch, MIN_AN, MIN_BN(an), MAX_BN(an) and then
 * include this file. */

/* Sizes are up to 2^SIZE_LOG limbs */
#ifndef SIZE_LOG
#define SIZE_LOG 10
#endif

#ifndef COUNT
#define COUNT 2000
#endif

#define MAX_AN (1L << SIZE_LOG)

#ifndef MAX_BN
#define MAX_BN(an) (an)
#endif

/* For general toomMN_mul, we need
 *
 * MIN_BN(an) = N + floor(((N-1)*an + M - N)/M)
 *
 * MAX_BN(an) = floor(N*(an-1)/(M-1)) - N + 1
 */

int
main (int argc, char **argv)
{
  mp_ptr ap, bp, refp, pp, scratch;
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

  ap = TMP_ALLOC_LIMBS (MAX_AN);
  bp = TMP_ALLOC_LIMBS (MAX_BN(MAX_AN));
  refp = TMP_ALLOC_LIMBS (MAX_AN + MAX_BN(MAX_AN));
  pp = 1+TMP_ALLOC_LIMBS (MAX_AN + MAX_BN(MAX_AN)+2);
  scratch
    = 1+TMP_ALLOC_LIMBS (mpn_toomMN_mul_itch (MAX_AN, MAX_BN(MAX_AN))
			 + 2);

  for (test = 0; test < count; test++)
    {
      unsigned size_min;
      unsigned size_range;
      mp_size_t an, bn;
      mp_size_t itch;
      mp_limb_t p_before, p_after, s_before, s_after;

      for (size_min = 1; (1L << size_min) < MIN_AN; size_min++)
	;

      /* We generate an in the MIN_AN <= an <= (1 << size_range). */
      size_range = size_min
	+ gmp_urandomm_ui (rands, SIZE_LOG + 1 - size_min);

      an = MIN_AN
	+ gmp_urandomm_ui (rands, (1L << size_range) + 1 - MIN_AN);
      bn = MIN_BN(an)
	+ gmp_urandomm_ui (rands, MAX_BN(an) + 1 - MIN_BN(an));

      mpn_random2 (ap, an);
      mpn_random2 (bp, bn);
      mpn_random2 (pp-1, an + bn + 2);
      p_before = pp[-1];
      p_after = pp[an + bn];

      itch = mpn_toomMN_mul_itch (an, bn);
      ASSERT_ALWAYS (itch <= mpn_toomMN_mul_itch (MAX_AN, MAX_BN(MAX_AN)));
      mpn_random2 (scratch-1, itch+2);
      s_before = scratch[-1];
      s_after = scratch[itch];

      mpn_toomMN_mul (pp, ap, an, bp, bn, scratch);
      refmpn_mul (refp, ap, an, bp, bn);
      if (pp[-1] != p_before || pp[an + bn] != p_after
	  || scratch[-1] != s_before || scratch[itch] != s_after
	  || mpn_cmp (refp, pp, an + bn) != 0)
	{
	  printf ("ERROR in test %d, an = %d, bn = %d\n",
		  test, (int) an, (int) bn);
	  if (pp[-1] != p_before)
	    {
	      printf ("before pp:"); mpn_dump (pp -1, 1);
	      printf ("keep:   "); mpn_dump (&p_before, 1);
	    }
	  if (pp[an + bn] != p_after)
	    {
	      printf ("after pp:"); mpn_dump (pp + an + bn, 1);
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
	  mpn_dump (bp, bn);
	  mpn_dump (pp, an + bn);
	  mpn_dump (refp, an + bn);

	  abort();
	}
    }
  TMP_FREE;

  tests_end ();
  return 0;
}
