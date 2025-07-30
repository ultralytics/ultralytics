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

#include "testutils.h"

#define MAXBITS 400
#define COUNT 9000

/* Called when s is supposed to be floor(sqrt(u)), and r = u - s^2 */
static int
sqrtrem_valid_p (const mpz_t u, const mpz_t s, const mpz_t r)
{
  mpz_t t;

  mpz_init (t);
  mpz_mul (t, s, s);
  mpz_sub (t, u, t);
  if (mpz_sgn (t) < 0 || mpz_cmp (t, r) != 0)
    {
      mpz_clear (t);
      return 0;
    }
  mpz_add_ui (t, s, 1);
  mpz_mul (t, t, t);
  if (mpz_cmp (t, u) <= 0)
    {
      mpz_clear (t);
      return 0;
    }

  mpz_clear (t);
  return 1;
}

void
mpz_mpn_sqrtrem (mpz_t s, mpz_t r, const mpz_t u)
{
  mp_limb_t *sp, *rp;
  mp_size_t un, sn, ret;

  un = mpz_size (u);

  mpz_xor (s, s, u);
  sn = (un + 1) / 2;
  sp = mpz_limbs_write (s, sn + 1);
  sp [sn] = 11;

  if (un & 1)
    rp = NULL; /* Exploits the fact that r already is correct. */
  else {
    mpz_add (r, u, s);
    rp = mpz_limbs_write (r, un + 1);
    rp [un] = 19;
  }

  ret = mpn_sqrtrem (sp, rp, mpz_limbs_read (u), un);

  if (sp [sn] != 11)
    {
      fprintf (stderr, "mpn_sqrtrem buffer overrun on sp.\n");
      abort ();
    }
  if (un & 1) {
    if ((ret != 0) != (mpz_size (r) != 0)) {
      fprintf (stderr, "mpn_sqrtrem wrong return value with NULL.\n");
      abort ();
    }
  } else {
    mpz_limbs_finish (r, ret);
    if (ret != mpz_size (r)) {
      fprintf (stderr, "mpn_sqrtrem wrong return value.\n");
      abort ();
    }
    if (rp [un] != 19)
      {
	fprintf (stderr, "mpn_sqrtrem buffer overrun on rp.\n");
	abort ();
      }
  }

  mpz_limbs_finish (s, (un + 1) / 2);
}

void
testmain (int argc, char **argv)
{
  unsigned i;
  mpz_t u, s, r;

  mpz_init (s);
  mpz_init (r);

  mpz_init_set_si (u, -1);
  if (mpz_perfect_square_p (u))
    {
      fprintf (stderr, "mpz_perfect_square_p failed on -1.\n");
      abort ();
    }

  if (!mpz_perfect_square_p (s))
    {
      fprintf (stderr, "mpz_perfect_square_p failed on 0.\n");
      abort ();
    }

  for (i = 0; i < COUNT; i++)
    {
      mini_rrandomb (u, MAXBITS - (i & 0xFF));
      mpz_sqrtrem (s, r, u);

      if (!sqrtrem_valid_p (u, s, r))
	{
	  fprintf (stderr, "mpz_sqrtrem failed:\n");
	  dump ("u", u);
	  dump ("sqrt", s);
	  dump ("rem", r);
	  abort ();
	}

      mpz_mpn_sqrtrem (s, r, u);

      if (!sqrtrem_valid_p (u, s, r))
	{
	  fprintf (stderr, "mpn_sqrtrem failed:\n");
	  dump ("u", u);
	  dump ("sqrt", s);
	  dump ("rem", r);
	  abort ();
	}

      if (mpz_sgn (r) == 0) {
	mpz_neg (u, u);
	mpz_sub_ui (u, u, 1);
      }

      if ((mpz_sgn (u) <= 0 || (i & 1)) ?
	  mpz_perfect_square_p (u) :
	  mpn_perfect_square_p (mpz_limbs_read (u), mpz_size (u)))
	{
	  fprintf (stderr, "mp%s_perfect_square_p failed on non square:\n",
		   (mpz_sgn (u) <= 0 || (i & 1)) ? "z" : "n");
	  dump ("u", u);
	  abort ();
	}

      mpz_mul (u, s, s);
      if (!((mpz_sgn (u) <= 0 || (i & 1)) ?
	    mpz_perfect_square_p (u) :
	    mpn_perfect_square_p (mpz_limbs_read (u), mpz_size (u))))
	{
	  fprintf (stderr, "mp%s_perfect_square_p failed on square:\n",
		   (mpz_sgn (u) <= 0 || (i & 1)) ? "z" : "n");
	  dump ("u", u);
	  abort ();
	}

    }
  mpz_clear (u);
  mpz_clear (s);
  mpz_clear (r);
}
