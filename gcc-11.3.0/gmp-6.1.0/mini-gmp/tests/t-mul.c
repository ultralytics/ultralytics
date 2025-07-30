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
  mpz_t a, b, res, res_ui, ref, tz;
  mp_limb_t t[2*MAXLIMBS];
  mp_size_t an;

  mpz_init (a);
  mpz_init (b);
  mpz_init (res);
  mpz_init (res_ui);
  mpz_init (ref);

  for (i = 0; i < COUNT; i++)
    {
      mini_random_op3 (OP_MUL, MAXBITS, a, b, ref);
      mpz_mul (res, a, b);
      if (mpz_cmp (res, ref))
	{
	  fprintf (stderr, "mpz_mul failed:\n");
	  dump ("a", a);
	  dump ("b", b);
	  dump ("r", res);
	  dump ("ref", ref);
	  abort ();
	}
      if (mpz_size (a) == mpz_size (b))
	{
	  memset (t, 0x55, sizeof(t));
	  an = mpz_size (a);
	  if (an > 0)
	    {
	      mpn_mul_n (t, a->_mp_d, b->_mp_d, an);

	      mpz_roinit_n (tz, t, 2*an);
	      if (mpz_cmpabs (tz, ref))
		{
		  fprintf (stderr, "mpn_mul_n failed:\n");
		  dump ("a", a);
		  dump ("b", b);
		  dump ("ref", ref);
		  abort ();
		}
	    }
	}
      if (mpz_fits_slong_p (b)) {
	mpz_mul_si (res_ui, a, mpz_get_si (b));
	if (mpz_cmp (res_ui, ref))
	  {
	    fprintf (stderr, "mpz_mul_si failed:\n");
	    dump ("a", a);
	    dump ("b", b);
	    dump ("r", res_ui);
	    dump ("ref", ref);
	    abort ();
	  }
      }
      mini_random_op2 (OP_SQR, MAXBITS, a, ref);
      an = mpz_size (a);
      if (an > 0)
	{
	  memset (t, 0x33, sizeof(t));
	  mpn_sqr (t, mpz_limbs_read (a), an);

	  mpz_roinit_n (tz, t, 2*an);
	  if (mpz_cmp (tz, ref))
	    {
	      fprintf (stderr, "mpn (squaring) failed:\n");
	      dump ("a", a);
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
