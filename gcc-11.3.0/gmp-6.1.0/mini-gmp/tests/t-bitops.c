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
#define COUNT 10000

void
testmain (int argc, char **argv)
{
  unsigned i;
  mpz_t a, res, ref;
  mp_bitcnt_t b;

  mpz_init (a);
  mpz_init (res);
  mpz_init (ref);

  for (i = 0; i < COUNT; i++)
    {
      mini_random_bit_op (OP_SETBIT, MAXBITS, a, &b, ref);
      mpz_set (res, a);
      mpz_setbit (res, b);
      if (mpz_cmp (res, ref))
	{
	  fprintf (stderr, "mpz_setbit failed:\n");
	  dump ("a", a);
	  fprintf (stderr, "b: %lu\n", b);
	  dump ("r", res);
	  dump ("ref", ref);
	  abort ();
	}
      if (!mpz_tstbit (res, b))
	{
	  fprintf (stderr, "mpz_tstbit failed (after mpz_setbit):\n");
	  dump ("res", a);
	  fprintf (stderr, "b: %lu\n", b);
	  abort ();
	}
      mini_random_bit_op (OP_CLRBIT, MAXBITS, a, &b, ref);
      mpz_set (res, a);
      mpz_clrbit (res, b);
      if (mpz_cmp (res, ref))
	{
	  fprintf (stderr, "mpz_clrbit failed:\n");
	  dump ("a", a);
	  fprintf (stderr, "b: %lu\n", b);
	  dump ("r", res);
	  dump ("ref", ref);
	  abort ();
	}
      if (mpz_tstbit (res, b))
	{
	  fprintf (stderr, "mpz_tstbit failed (after mpz_clrbit):\n");
	  dump ("res", a);
	  fprintf (stderr, "b: %lu\n", b);
	  abort ();
	}
      mini_random_bit_op (OP_COMBIT, MAXBITS, a, &b, ref);
      mpz_set (res, a);
      mpz_combit (res, b);
      if (mpz_cmp (res, ref))
	{
	  fprintf (stderr, "mpz_combit failed:\n");
	  dump ("a", a);
	  fprintf (stderr, "b: %lu\n", b);
	  dump ("r", res);
	  dump ("ref", ref);
	  abort ();
	}
      if (mpz_tstbit (res, b) == mpz_tstbit (a, b))
	{
	  fprintf (stderr, "mpz_tstbit failed (after mpz_combit):\n");
	  dump ("res", a);
	  fprintf (stderr, "b: %lu\n", b);
	  abort ();
	}
    }
  mpz_clear (a);
  mpz_clear (res);
  mpz_clear (ref);
}
