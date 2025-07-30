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

/* Called when s is supposed to be floor(root(u,z)), and r = u - s^z */
static int
rootrem_valid_p (const mpz_t u, const mpz_t s, const mpz_t r, unsigned long z)
{
  mpz_t t;

  mpz_init (t);
  if (mpz_fits_ulong_p (s))
    mpz_ui_pow_ui (t, mpz_get_ui (s), z);
  else
    mpz_pow_ui (t, s, z);
  mpz_sub (t, u, t);
  if ((mpz_sgn (t) != mpz_sgn(u) && mpz_sgn (t) != 0) || mpz_cmp (t, r) != 0)
    {
      mpz_clear (t);
      return 0;
    }
  if (mpz_sgn (s) > 0)
    mpz_add_ui (t, s, 1);
  else
    mpz_sub_ui (t, s, 1);
  mpz_pow_ui (t, t, z);
  if (mpz_cmpabs (t, u) <= 0)
    {
      mpz_clear (t);
      return 0;
    }

  mpz_clear (t);
  return 1;
}

void
testmain (int argc, char **argv)
{
  unsigned i;
  unsigned long e;
  mpz_t u, s, r, bs;

  mpz_init (u);
  mpz_init (s);
  mpz_init (r);
  mpz_init (bs);

  for (i = 0; i < COUNT; i++)
    {
      mini_rrandomb (u, MAXBITS);
      mini_rrandomb (bs, 12);
      e = mpz_getlimbn (bs, 0) % mpz_sizeinbase (u, 2) + 1;
      if ((e & 1) && (mpz_getlimbn (bs, 0) & (1L<<10)))
	mpz_neg (u, u);
      mpz_rootrem (s, r, u, e);

      if (!rootrem_valid_p (u, s, r, e))
	{
	  fprintf (stderr, "mpz_rootrem(%lu-th) failed:\n", e);
	  dump ("u", u);
	  dump ("root", s);
	  dump ("rem", r);
	  abort ();
	}
    }
  mpz_clear (bs);
  mpz_clear (u);
  mpz_clear (s);
  mpz_clear (r);
}
