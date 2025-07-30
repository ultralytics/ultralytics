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

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

#include "testutils.h"

#define MAXBITS 400
#define COUNT 10000

void
testlogops (int count)
{
  unsigned i;
  mpz_t a, b, res, ref;
  mp_bitcnt_t c;

  mpz_init (a);
  mpz_init (b);
  mpz_init (res);
  mpz_init (ref);

  for (i = 0; i < count; i++)
    {
      mini_random_op3 (OP_AND, MAXBITS, a, b, ref);
      mpz_and (res, a, b);
      if (mpz_cmp (res, ref))
	{
	  fprintf (stderr, "mpz_and failed:\n");
	  dump ("a", a);
	  dump ("b", b);
	  dump ("r", res);
	  dump ("ref", ref);
	  abort ();
	}

      mini_random_op3 (OP_IOR, MAXBITS, a, b, ref);
      mpz_ior (res, a, b);
      if (mpz_cmp (res, ref))
	{
	  fprintf (stderr, "mpz_ior failed:\n");
	  dump ("a", a);
	  dump ("b", b);
	  dump ("r", res);
	  dump ("ref", ref);
	  abort ();
	}

      mini_random_op3 (OP_XOR, MAXBITS, a, b, ref);
      mpz_xor (res, a, b);
      if (mpz_cmp (res, ref))
	{
	  fprintf (stderr, "mpz_xor failed:\n");
	  dump ("a", a);
	  dump ("b", b);
	  dump ("r", res);
	  dump ("ref", ref);
	  abort ();
	}

      if (i % 8) {
	c = 0;
	mpz_mul_2exp (res, res, i % 8);
      } else if (mpz_sgn (res) >= 0) {
	c = mpz_odd_p (res) != 0;
	mpz_tdiv_q_2exp (res, res, 1);
      } else {
	c = (~ (mp_bitcnt_t) 0) - 3;
	mpz_set_ui (res, 11 << ((i >> 3)%4)); /* set 3 bits */
      }

      if (mpz_popcount (res) + c != mpz_hamdist (a, b))
	{
	  fprintf (stderr, "mpz_popcount(r) + %lu and mpz_hamdist(a,b) differ:\n", c);
	  dump ("a", a);
	  dump ("b", b);
	  dump ("r", res);
	  fprintf (stderr, "mpz_popcount(r) = %lu:\n", mpz_popcount (res));
	  fprintf (stderr, "mpz_hamdist(a,b) = %lu:\n", mpz_hamdist (a, b));
	  abort ();
	}
    }
  mpz_clear (a);
  mpz_clear (b);
  mpz_clear (res);
  mpz_clear (ref);
}

void
testmain (int argc, char **argv)
{
  testhalves (COUNT*2/3, testlogops);
  testlogops (COUNT/3);
}
