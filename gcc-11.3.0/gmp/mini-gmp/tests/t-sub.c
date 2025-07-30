/*

Copyright 2012, 2013 Free Software Foundation, Inc.

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

#include "testutils.h"

#define MAXBITS 400
#define COUNT 10000

void
testmain (int argc, char **argv)
{
  unsigned i;
  mpz_t a, b, res, res_ui, ref;

  mpz_init (a);
  mpz_init (b);
  mpz_init (res);
  mpz_init (res_ui);
  mpz_init (ref);

  for (i = 0; i < COUNT; i++)
    {
      mini_random_op3 (OP_SUB, MAXBITS, a, b, ref);
      mpz_sub (res, a, b);
      if (mpz_cmp (res, ref))
	{
	  fprintf (stderr, "mpz_sub failed:\n");
	  dump ("a", a);
	  dump ("b", b);
	  dump ("r", res);
	  dump ("ref", ref);
	  abort ();
	}
      if (mpz_fits_ulong_p (a)) {
	mpz_ui_sub (res_ui, mpz_get_ui (a), b);
	if (mpz_cmp (res_ui, ref))
	  {
	    fprintf (stderr, "mpz_ui_sub failed:\n");
	    dump ("a", a);
	    dump ("b", b);
	    dump ("r", res_ui);
	    dump ("ref", ref);
	    abort ();
	  }
      }
    }
  mpz_clear (a);
  mpz_clear (b);
  mpz_clear (res);
  mpz_clear (res_ui);
  mpz_clear (ref);
}
