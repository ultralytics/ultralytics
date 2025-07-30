/*

Copyright 2012, Free Software Foundation, Inc.

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

#include "testutils.h"

#define MAXBITS 400
#define COUNT 1000

void
testmain (int argc, char **argv)
{
  unsigned i;
  mpz_t b, e, m, res, ref;

  mpz_init (b);
  mpz_init (e);
  mpz_init (m);
  mpz_init (res);
  mpz_init (ref);

  for (i = 0; i < COUNT; i++)
    {
      mini_random_op4 (OP_POWM, MAXBITS, b, e, m, ref);
      mpz_powm (res, b, e, m);
      if (mpz_cmp (res, ref))
	{
	  fprintf (stderr, "mpz_powm failed:\n");
	  dump ("b", b);
	  dump ("e", e);
	  dump ("m", m);
	  dump ("r", res);
	  dump ("ref", ref);
	  abort ();
	}
    }
  mpz_clear (b);
  mpz_clear (e);
  mpz_clear (m);
  mpz_clear (res);
  mpz_clear (ref);
}
