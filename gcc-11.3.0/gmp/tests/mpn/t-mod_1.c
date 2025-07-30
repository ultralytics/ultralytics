/* Test mpn_mod_1 variants.

Copyright 2010, 2013 Free Software Foundation, Inc.

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

#include <stdio.h>
#include <stdlib.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

static void
check_one (mp_srcptr ap, mp_size_t n, mp_limb_t b)
{
  mp_limb_t r_ref = refmpn_mod_1 (ap, n, b);
  mp_limb_t r;

  if (n >= 2)
    {
      mp_limb_t pre[4];
      mpn_mod_1_1p_cps (pre, b);
      r = mpn_mod_1_1p (ap, n, b << pre[1], pre);
      if (r != r_ref)
	{
	  printf ("mpn_mod_1_1p failed\n");
	  goto fail;
	}
    }
  if ((b & GMP_NUMB_HIGHBIT) == 0)
    {
      mp_limb_t pre[5];
      mpn_mod_1s_2p_cps (pre, b);
      r = mpn_mod_1s_2p (ap, n, b << pre[1], pre);
      if (r != r_ref)
	{
	  printf ("mpn_mod_1s_2p failed\n");
	  goto fail;
	}
    }
  if (b <= GMP_NUMB_MASK / 3)
    {
      mp_limb_t pre[6];
      mpn_mod_1s_3p_cps (pre, b);
      r = mpn_mod_1s_3p (ap, n, b << pre[1], pre);
      if (r != r_ref)
	{
	  printf ("mpn_mod_1s_3p failed\n");
	  goto fail;
	}
    }
  if (b <= GMP_NUMB_MASK / 4)
    {
      mp_limb_t pre[7];
      mpn_mod_1s_4p_cps (pre, b);
      r = mpn_mod_1s_4p (ap, n, b << pre[1], pre);
      if (r != r_ref)
	{
	  printf ("mpn_mod_1s_4p failed\n");
	  goto fail;
	}
    }
  r = mpn_mod_1 (ap, n, b);
  if (r != r_ref)
    {
      printf ("mpn_mod_1 failed\n");
    fail:
      printf ("an = %d, a: ", (int) n); mpn_dump (ap, n);
      printf ("b           : "); mpn_dump (&b, 1);
      printf ("r (expected): "); mpn_dump (&r_ref, 1);
      printf ("r (bad)     : "); mpn_dump (&r, 1);
      abort();
    }
}

int
main (int argc, char **argv)
{
  gmp_randstate_ptr rands;
  int i;
  unsigned a_bits;
  unsigned b_bits;
  mpz_t a;
  mpz_t b;

  tests_start ();
  rands = RANDS;
  mpz_init (a);
  mpz_init (b);

  for (i = 0; i < 300; i++)
    {
      mp_size_t asize;
      a_bits = 1 + gmp_urandomm_ui (rands, 1000);
      b_bits = 1 + gmp_urandomm_ui (rands, GMP_NUMB_BITS);

      mpz_rrandomb (a, rands, a_bits);
      mpz_rrandomb (b, rands, b_bits);

      asize = SIZ(a);
      if (!asize)
	asize = 1;
      if (mpz_sgn (b) == 0)
	mpz_set_ui (b, 1);

      check_one (PTR(a), asize, PTR(b)[0]);
    }

  mpz_clear (a);
  mpz_clear (b);

  tests_end ();
  return 0;
}

