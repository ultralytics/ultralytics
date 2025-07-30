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
  mpz_t a;
  mp_bitcnt_t b, res, ref;

  mpz_init (a);

  for (i = 0; i < COUNT; i++)
    {
      mini_random_scan_op (OP_SCAN0, MAXBITS, a, &b, &ref);
      res = mpz_scan0 (a, b);
      if (res != ref)
	{
	  fprintf (stderr, "mpz_scan0 failed:\n");
	  dump ("a", a);
	  fprintf (stderr, "b: %lu\n", b);
	  fprintf (stderr, "r: %lu\n", res);
	  fprintf (stderr, "ref: %lu\n", ref);
	  abort ();
	}
      if (mpz_sgn (a) > 0 && ref < mpz_sizeinbase (a, 2))
	{
	  res = mpn_scan0 (a->_mp_d, b);
	  if (res != ref)
	    {
	      fprintf (stderr, "mpn_scan0 failed:\n");
	      dump ("a", a);
	      fprintf (stderr, "b: %lu\n", b);
	      fprintf (stderr, "r: %lu\n", res);
	      fprintf (stderr, "ref: %lu\n", ref);
	      abort ();
	    }
	}
      mini_random_scan_op (OP_SCAN1, MAXBITS, a, &b, &ref);
      res = mpz_scan1 (a, b);
      if (res != ref)
	{
	  fprintf (stderr, "mpz_scan1 failed:\n");
	  dump ("a", a);
	  fprintf (stderr, "b: %lu\n", b);
	  fprintf (stderr, "r: %lu\n", res);
	  fprintf (stderr, "ref: %lu\n", ref);
	  abort ();
	}
      if (mpz_sgn (a) > 0 && ref != ~ (mp_bitcnt_t) 0)
	{
	  res = mpn_scan1 (a->_mp_d, b);
	  if (res != ref)
	    {
	      fprintf (stderr, "mpn_scan1 failed:\n");
	      dump ("a", a);
	      fprintf (stderr, "b: %lu\n", b);
	      fprintf (stderr, "r: %lu\n", res);
	      fprintf (stderr, "ref: %lu\n", ref);
	      abort ();
	    }
	}
    }
  mpz_clear (a);
}
