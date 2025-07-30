/*

Copyright 2012, 2014, Free Software Foundation, Inc.

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

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "testutils.h"

#define MAXBITS 400
#define COUNT 10000

#define GMP_LIMB_BITS (sizeof(mp_limb_t) * CHAR_BIT)
#define MAXLIMBS ((MAXBITS + GMP_LIMB_BITS - 1) / GMP_LIMB_BITS)

void
testmain (int argc, char **argv)
{
  unsigned i;
  mpz_t a, b, res, ref;

  mpz_init (a);
  mpz_init (b);
  mpz_init_set_ui (res, 5);
  mpz_init (ref);

  for (i = 0; i < COUNT; i++)
    {
      mini_random_op3 (OP_MUL, MAXBITS, a, b, ref);
      if (i & 1) {
	mpz_add (ref, ref, res);
	if (mpz_fits_ulong_p (b))
	  mpz_addmul_ui (res, a, mpz_get_ui (b));
	else
	  mpz_addmul (res, a, b);
      } else {
	mpz_sub (ref, res, ref);
	if (mpz_fits_ulong_p (b))
	  mpz_submul_ui (res, a, mpz_get_ui (b));
	else
	  mpz_submul (res, a, b);
      }
      if (mpz_cmp (res, ref))
	{
	  if (i & 1)
	    fprintf (stderr, "mpz_addmul failed:\n");
	  else
	    fprintf (stderr, "mpz_submul failed:\n");
	  dump ("a", a);
	  dump ("b", b);
	  dump ("r", res);
	  dump ("ref", ref);
	  abort ();
	}
    }
  mpz_clear (a);
  mpz_clear (b);
  mpz_clear (res);
  mpz_clear (ref);
}
