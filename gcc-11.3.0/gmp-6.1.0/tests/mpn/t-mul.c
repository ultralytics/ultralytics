/* Test mpn_mul function for all sizes up to a selected limit.

Copyright 2011, 2012 Free Software Foundation, Inc.

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

static unsigned
isqrt (unsigned t)
{
  unsigned s, b;

  for (b = 0, s = t;  b++, s >>= 1; )
    ;

  s = 1 << (b >> 1);
  if (b & 1)
    s += s >> 1;

  do
    {
      b = t / s;
      s = (s + b) >> 1;
    }
  while (b < s);

  return s;
}

int
main (int argc, char **argv)
{
  mp_ptr ap, bp, rp, refp;
  mp_size_t max_n, an, bn, rn;
  gmp_randstate_ptr rands;
  int reps;
  TMP_DECL;
  TMP_MARK;

  reps = 1;

  tests_start ();
  TESTS_REPS (reps, argv, argc);

  rands = RANDS;

  /* Re-interpret reps argument as a size argument.  */
  max_n = isqrt (reps * 25000);

  ap = TMP_ALLOC_LIMBS (max_n + 1);
  bp = TMP_ALLOC_LIMBS (max_n + 1);
  rp = TMP_ALLOC_LIMBS (2 * max_n);
  refp = TMP_ALLOC_LIMBS (2 * max_n);

  for (an = 1; an <= max_n; an += 1)
    {
      for (bn = 1; bn <= an; bn += 1)
	{
	  mpn_random2 (ap, an + 1);
	  mpn_random2 (bp, bn + 1);

	  refmpn_mul (refp, ap, an, bp, bn);
	  mpn_mul (rp, ap, an, bp, bn);

	  rn = an + bn;
	  if (mpn_cmp (refp, rp, rn))
	    {
	      printf ("ERROR, an = %d, bn = %d, rn = %d\n",
		      (int) an, (int) bn, (int) rn);
	      printf ("a: "); mpn_dump (ap, an);
	      printf ("b: "); mpn_dump (bp, bn);
	      printf ("r:   "); mpn_dump (rp, rn);
	      printf ("ref: "); mpn_dump (refp, rn);
	      abort();
	    }
	}
    }
  TMP_FREE;
  tests_end ();
  return 0;
}
