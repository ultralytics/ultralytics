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
#define COUNT 100

void
my_mpz_mul (mpz_t r, mpz_srcptr a,  mpz_srcptr b)
{
  mp_limb_t *tp;
  mp_size_t tn, an, bn;

  an = mpz_size (a);
  bn = mpz_size (b);
  tn = an + bn;

  tp = mpz_limbs_write (r, tn);
  if (mpz_sgn (a) * mpz_sgn(b) == 0)
    mpn_zero (tp, tn);
  else if (an > bn)
    mpn_mul (tp, mpz_limbs_read (a), an, mpz_limbs_read (b), bn);
  else
    mpn_mul (tp, mpz_limbs_read (b), bn, mpz_limbs_read (a), an);

  if (mpz_sgn (a) != mpz_sgn(b))
    tn = - tn;

  mpz_limbs_finish (r, tn);
}

void
testmain (int argc, char **argv)
{
  unsigned i;
  mpz_t a, b, res, ref;

  mpz_init (a);
  mpz_init (b);
  mpz_init (res);
  mpz_init (ref);

  for (i = 0; i < COUNT; i++)
    {
      mini_random_op3 (OP_MUL, MAXBITS, a, b, ref);
      my_mpz_mul (res, a, b);
      if (mpz_cmp (res, ref))
	{
	  fprintf (stderr, "my_mpz_mul failed:\n");
	  dump ("a", a);
	  dump ("b", b);
	  dump ("r", res);
	  dump ("ref", ref);
	  abort ();
	}
      /* The following test exploits a side-effect of my_mpz_mul: res
	 points to a buffer with at least an+bn limbs, and the limbs
	 above the result are zeroed. */
      if (mpz_size (b) > 0 && mpz_getlimbn (res, mpz_size(a)) != mpz_limbs_read (res) [mpz_size(a)])
	{
	  fprintf (stderr, "getlimbn - limbs_read differ.\n");
	  abort ();
	}
      if ((i % 4 == 0) && mpz_size (res) > 1)
	{
	  mpz_realloc2 (res, 1);
	  if (mpz_cmp_ui (res, 0))
	    {
	      fprintf (stderr, "mpz_realloc2 did not clear res.\n");
	      abort ();
	    }
	  mpz_limbs_finish (ref, 0);
	  if (mpz_cmp_d (ref, 0))
	    {
	      fprintf (stderr, "mpz_limbs_finish did not clear res.\n");
	      abort ();
	    }
	}
    }
  mpz_clear (a);
  mpz_clear (b);
  mpz_clear (res);
  mpz_clear (ref);
}
