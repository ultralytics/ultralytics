/* Test for mulmid function.

Copyright 2011 Free Software Foundation, Inc.

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
#define SIZE_LOG 9
#endif

#ifndef COUNT
#define COUNT 5000
#endif

#define MAX_N (1L << SIZE_LOG)

int
main (int argc, char **argv)
{
  mp_ptr ap, bp, rp, refp;
  gmp_randstate_ptr rands;
  int test;
  TMP_DECL;
  TMP_MARK;

  tests_start ();
  rands = RANDS;

  ap = TMP_ALLOC_LIMBS (MAX_N);
  bp = TMP_ALLOC_LIMBS (MAX_N);
  rp = TMP_ALLOC_LIMBS (MAX_N + 2);
  refp = TMP_ALLOC_LIMBS (MAX_N + 2);

  for (test = 0; test < COUNT; test++)
    {
      mp_size_t an, bn, rn;
      unsigned size_log;

      size_log = 1 + gmp_urandomm_ui (rands, SIZE_LOG);
      an = 1 + gmp_urandomm_ui(rands, 1L << size_log);

      size_log = 1 + gmp_urandomm_ui (rands, SIZE_LOG);
      bn = 1 + gmp_urandomm_ui(rands, 1L << size_log);

      /* Make sure an >= bn */
      if (an < bn)
	MP_SIZE_T_SWAP (an, bn);

      mpn_random2 (ap, an);
      mpn_random2 (bp, bn);

      refmpn_mulmid (refp, ap, an, bp, bn);
      mpn_mulmid (rp, ap, an, bp, bn);

      rn = an + 3 - bn;
      if (mpn_cmp (refp, rp, rn))
	{
	  printf ("ERROR in test %d, an = %d, bn = %d, rn = %d\n",
		  test, (int) an, (int) bn, (int) rn);
	  printf("a: "); mpn_dump (ap, an);
	  printf("b: "); mpn_dump (bp, bn);
	  printf("r:   "); mpn_dump (rp, rn);
	  printf("ref: "); mpn_dump (refp, rn);

	  abort();
	}
    }
  TMP_FREE;
  tests_end ();
  return 0;
}
