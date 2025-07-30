/* Test for mulmod_bnm1 function.

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
#define SIZE_LOG 11
#endif

#ifndef COUNT
#define COUNT 5000
#endif

#define MAX_N (1L << SIZE_LOG)
#define MIN_N 1

/*
  Reference function for multiplication modulo B^rn-1.

  The result is expected to be ZERO if and only if one of the operand
  already is. Otherwise the class [0] Mod(B^rn-1) is represented by
  B^rn-1. This should not be a problem if mulmod_bnm1 is used to
  combine results and obtain a natural number when one knows in
  advance that the final value is less than (B^rn-1).
*/

static void
ref_mulmod_bnm1 (mp_ptr rp, mp_size_t rn, mp_srcptr ap, mp_size_t an, mp_srcptr bp, mp_size_t bn)
{
  mp_limb_t cy;

  ASSERT (0 < an && an <= rn);
  ASSERT (0 < bn && bn <= rn);

  if (an >= bn)
    refmpn_mul (rp, ap, an, bp, bn);
  else
    refmpn_mul (rp, bp, bn, ap, an);
  an += bn;
  if (an > rn) {
    cy = mpn_add (rp, rp, rn, rp + rn, an - rn);
    /* If cy == 1, then the value of rp is at most B^rn - 2, so there can
     * be no overflow when adding in the carry. */
    MPN_INCR_U (rp, rn, cy);
  }
}

/*
  Compare the result of the mpn_mulmod_bnm1 function in the library
  with the reference function above.
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

  ASSERT_ALWAYS (mpn_mulmod_bnm1_next_size (MAX_N) == MAX_N);

  ap = TMP_ALLOC_LIMBS (MAX_N);
  bp = TMP_ALLOC_LIMBS (MAX_N);
  refp = TMP_ALLOC_LIMBS (MAX_N * 4);
  pp = 1+TMP_ALLOC_LIMBS (MAX_N + 2);
  scratch
    = 1+TMP_ALLOC_LIMBS (mpn_mulmod_bnm1_itch (MAX_N, MAX_N, MAX_N) + 2);

  for (test = 0; test < count; test++)
    {
      unsigned size_min;
      unsigned size_range;
      mp_size_t an,bn,rn,n;
      mp_size_t itch;
      mp_limb_t p_before, p_after, s_before, s_after;

      for (size_min = 1; (1L << size_min) < MIN_N; size_min++)
	;

      /* We generate an in the MIN_N <= n <= (1 << size_range). */
      size_range = size_min
	+ gmp_urandomm_ui (rands, SIZE_LOG + 1 - size_min);

      n = MIN_N
	+ gmp_urandomm_ui (rands, (1L << size_range) + 1 - MIN_N);
      n = mpn_mulmod_bnm1_next_size (n);

      if ( (test & 1) || n == 1) {
	/* Half of the tests are done with the main scenario in mind:
	   both an and bn >= rn/2 */
	an = ((n+1) >> 1) + gmp_urandomm_ui (rands, (n+1) >> 1);
	bn = ((n+1) >> 1) + gmp_urandomm_ui (rands, (n+1) >> 1);
      } else {
	/* Second half of the tests are done using mulmod to compute a
	   full product with n/2 < an+bn <= n. */
	an = 1 + gmp_urandomm_ui (rands, n - 1);
	if (an >= n/2)
	  bn = 1 + gmp_urandomm_ui (rands, n - an);
	else
	  bn = n/2 + 1 - an + gmp_urandomm_ui (rands, (n+1)/2);
      }

      /* Make sure an >= bn */
      if (an < bn)
	MP_SIZE_T_SWAP (an, bn);

      mpn_random2 (ap, an);
      mpn_random2 (bp, bn);

      /* Sometime trigger the borderline conditions
	 A = -1,0,+1 or B = -1,0,+1 or A*B == -1,0,1 Mod(B^{n/2}+1).
	 This only makes sense if there is at least a split, i.e. n is even. */
      if ((test & 0x1f) == 1 && (n & 1) == 0) {
	mp_size_t x;
	MPN_COPY (ap, ap + (n >> 1), an - (n >> 1));
	MPN_ZERO (ap + an - (n >> 1) , n - an);
	MPN_COPY (bp, bp + (n >> 1), bn - (n >> 1));
	MPN_ZERO (bp + bn - (n >> 1) , n - bn);
	x = (n == an) ? 0 : gmp_urandomm_ui (rands, n - an);
	ap[x] += gmp_urandomm_ui (rands, 3) - 1;
	x = (n >> 1) - x % (n >> 1);
	bp[x] += gmp_urandomm_ui (rands, 3) - 1;
	/* We don't propagate carry, this means that the desired condition
	   is not triggered all the times. A few times are enough anyway. */
      }
      rn = MIN(n, an + bn);
      mpn_random2 (pp-1, rn + 2);
      p_before = pp[-1];
      p_after = pp[rn];

      itch = mpn_mulmod_bnm1_itch (n, an, bn);
      ASSERT_ALWAYS (itch <= mpn_mulmod_bnm1_itch (MAX_N, MAX_N, MAX_N));
      mpn_random2 (scratch-1, itch+2);
      s_before = scratch[-1];
      s_after = scratch[itch];

      mpn_mulmod_bnm1 (  pp, n, ap, an, bp, bn, scratch);
      ref_mulmod_bnm1 (refp, n, ap, an, bp, bn);
      if (pp[-1] != p_before || pp[rn] != p_after
	  || scratch[-1] != s_before || scratch[itch] != s_after
	  || mpn_cmp (refp, pp, rn) != 0)
	{
	  printf ("ERROR in test %d, an = %d, bn = %d, n = %d\n",
		  test, (int) an, (int) bn, (int) n);
	  if (pp[-1] != p_before)
	    {
	      printf ("before pp:"); mpn_dump (pp -1, 1);
	      printf ("keep:   "); mpn_dump (&p_before, 1);
	    }
	  if (pp[rn] != p_after)
	    {
	      printf ("after pp:"); mpn_dump (pp + rn, 1);
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
	  mpn_dump (pp, rn);
	  mpn_dump (refp, rn);

	  abort();
	}
    }
  TMP_FREE;
  tests_end ();
  return 0;
}
