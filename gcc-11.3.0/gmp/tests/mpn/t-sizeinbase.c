/* Test for sizeinbase function.

Copyright 2014 Free Software Foundation, Inc.

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

/* Exponents up to 2^SIZE_LOG */
#ifndef SIZE_LOG
#define SIZE_LOG 13
#endif

#ifndef COUNT
#define COUNT 30
#endif

#define MAX_N (1<<SIZE_LOG)

int
main (int argc, char **argv)
{
  mp_limb_t a;
  mp_ptr pp, scratch;
  mp_limb_t max_b;
  int count = COUNT;
  int test;
  gmp_randstate_ptr rands;
  TMP_DECL;

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
  TMP_MARK;
  rands = RANDS;

  pp = TMP_ALLOC_LIMBS (MAX_N);
  scratch = TMP_ALLOC_LIMBS (MAX_N);
  max_b = numberof (mp_bases);

  ASSERT_ALWAYS (max_b > 62);
  ASSERT_ALWAYS (max_b < GMP_NUMB_MAX);

  for (a = 2; a < max_b; ++a)
    for (test = 0; test < count; ++test)
      {
	mp_size_t pn;
	mp_limb_t exp;
	mp_bitcnt_t res;

	exp = gmp_urandomm_ui (rands, MAX_N);

	pn = mpn_pow_1 (pp, &a, 1, exp, scratch);

	res = mpn_sizeinbase (pp, pn, a) - 1;

	if ((res < exp) || (res > exp + 1))
	  {
	    printf ("ERROR in test %d, base = %d, exp = %d, res = %d\n",
		    test, (int) a, (int) exp, (int) res);
	    abort();
	  }

	mpn_sub_1 (pp, pp, pn, CNST_LIMB(1));
	pn -= pp[pn-1] == 0;

	res = mpn_sizeinbase (pp, pn, a);

	if ((res < exp) || (res - 1 > exp))
	  {
	    printf ("ERROR in -1 test %d, base = %d, exp = %d, res = %d\n",
		    test, (int) a, (int) exp, (int) res);
	    abort();
	  }
      }

  TMP_FREE;
  tests_end ();
  return 0;
}
