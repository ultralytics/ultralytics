/* Copyright 2012, 2015 Free Software Foundation, Inc.

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


#include <stdlib.h>		/* for strtol */
#include <stdio.h>		/* for printf */

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"
#include "tests/tests.h"

#define MAX_LIMBS 150
#define COUNT 500

int
main (int argc, char **argv)
{
  gmp_randstate_ptr rands;

  mp_ptr ap, rp, pp, app, scratch;
  int count = COUNT;
  unsigned i;
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

  ap = TMP_ALLOC_LIMBS (MAX_LIMBS);
  rp = TMP_ALLOC_LIMBS (MAX_LIMBS);
  pp = TMP_ALLOC_LIMBS (MAX_LIMBS);
  app = TMP_ALLOC_LIMBS (MAX_LIMBS);
  scratch = TMP_ALLOC_LIMBS (5*MAX_LIMBS);

  for (i = 0; i < count; i++)
    {
      mp_size_t n;
      mp_limb_t k;

      n = 1 + gmp_urandomm_ui (rands, MAX_LIMBS);

      if (i & 1)
	mpn_random2 (ap, n);
      else
	mpn_random (ap, n);

      ap[0] |= 1;

      if (i < 100)
	k = 3 + 2*i;
      else
	{
	  mpn_random (&k, 1);
	  if (k < 3)
	    k = 3;
	  else
	    k |= 1;
	}
      mpn_brootinv (rp, ap, n, k, scratch);
      mpn_powlo (pp, rp, &k, 1, n, scratch);
      mpn_mullo_n (app, ap, pp, n);

      if (app[0] != 1 || !(n == 1 || mpn_zero_p (app+1, n-1)))
	{
	  gmp_fprintf (stderr,
		       "mpn_brootinv returned bad result: %u limbs\n",
		       (unsigned) n);
	  gmp_fprintf (stderr, "k     = %Mx\n", k);
	  gmp_fprintf (stderr, "a     = %Nx\n", ap, n);
	  gmp_fprintf (stderr, "r     = %Nx\n", rp, n);
	  gmp_fprintf (stderr, "r^n   = %Nx\n", pp, n);
	  gmp_fprintf (stderr, "a r^n = %Nx\n", app, n);
	  abort ();
	}
    }
  TMP_FREE;
  tests_end ();
  return 0;
}
