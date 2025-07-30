/* test mpz_congruent_2exp_p */

/*
Copyright 2001, 2013 Free Software Foundation, Inc.

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


void
check_one (mpz_srcptr a, mpz_srcptr c, unsigned long d, int want)
{
  mpz_t  diff, d2exp;
  int    got;
  int    swap;

  for (swap = 0; swap <= 1; swap++)
    {
      got = (mpz_congruent_2exp_p (a, c, d) != 0);
      if (want != got)
        {
          mpz_init (diff);
          mpz_init (d2exp);

          mpz_sub (diff, a, c);
          mpz_set_ui (d2exp, 1L);
          mpz_mul_2exp (d2exp, d2exp, d);

          printf ("mpz_congruent_2exp_p wrong\n");
          printf ("   expected %d got %d\n", want, got);
          mpz_trace ("   a", a);
          mpz_trace ("   c", c);
          mpz_trace (" a-c", diff);
          mpz_trace (" 2^d", d2exp);
          printf    ("   d=%lu\n", d);

          mp_trace_base = -16;
          mpz_trace ("   a", a);
          mpz_trace ("   c", c);
          mpz_trace (" a-c", diff);
          mpz_trace (" 2^d", d2exp);
          printf    ("   d=0x%lX\n", d);
          abort ();
        }

      MPZ_SRCPTR_SWAP (a, c);
    }
}


void
check_data (void)
{
  static const struct {
    const char     *a;
    const char     *c;
    unsigned long  d;
    int            want;

  } data[] = {

    /* anything is congruent mod 1 */
    { "0", "0", 0, 1 },
    { "1", "0", 0, 1 },
    { "0", "1", 0, 1 },
    { "123", "-456", 0, 1 },
    { "0x123456789123456789", "0x987654321987654321", 0, 1 },
    { "0xfffffffffffffffffffffffffffffff7", "-0x9", 129, 0 },
    { "0xfffffffffffffffffffffffffffffff6", "-0xa", 128, 1 },

  };

  mpz_t   a, c;
  int     i;

  mpz_init (a);
  mpz_init (c);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (a, data[i].a, 0);
      mpz_set_str_or_abort (c, data[i].c, 0);
      check_one (a, c, data[i].d, data[i].want);
    }

  mpz_clear (a);
  mpz_clear (c);
}


void
check_random (int reps)
{
  gmp_randstate_ptr rands = RANDS;
  unsigned long  d;
  mpz_t  a, c, ra, rc;
  int    i;

  mpz_init (a);
  mpz_init (c);
  mpz_init (ra);
  mpz_init (rc);

  for (i = 0; i < reps; i++)
    {
      mpz_errandomb (a, rands, 8*GMP_LIMB_BITS);
      mpz_errandomb (c, rands, 8*GMP_LIMB_BITS);
      d = urandom() % (8*GMP_LIMB_BITS);

      mpz_mul_2exp (a, a, urandom() % (2*GMP_LIMB_BITS));
      mpz_mul_2exp (c, c, urandom() % (2*GMP_LIMB_BITS));

      mpz_negrandom (a, rands);
      mpz_negrandom (c, rands);

      mpz_fdiv_r_2exp (ra, a, d);
      mpz_fdiv_r_2exp (rc, c, d);

      mpz_sub (ra, ra, rc);
      if (mpz_cmp_ui (ra, 0) != 0)
	{
	  check_one (a, c, d, 0);
	  mpz_sub (a, a, ra);
	}
      check_one (a, c, d, 1);
      if (d != 0)
	{
	  mpz_combit (a, urandom() % d);
	  check_one (a, c, d, 0);
	}
    }

  mpz_clear (a);
  mpz_clear (c);
  mpz_clear (ra);
  mpz_clear (rc);
}

void
check_random_bits (int reps)
{
  mp_bitcnt_t ea, ec, en, d;
  mp_bitcnt_t m = 10 * GMP_LIMB_BITS;
  mpz_t  a, c;
  int    i;

  mpz_init2 (a, m + 1);
  mpz_init2 (c, m);

  for (i = 0; i < reps; i++)
    {
      d  = urandom() % m;
      ea = urandom() % m;
      ec = urandom() % m;
      en = urandom() % m;

      mpz_set_ui (c, 0);
      mpz_setbit (c, en);

      mpz_set_ui (a, 0);
      mpz_setbit (a, ec);
      mpz_sub (c , a, c);

      mpz_set_ui (a, 0);
      mpz_setbit (a, ea);
      mpz_add (a , a, c);

      check_one (a, c, d, ea >= d);
    }

  mpz_clear (a);
  mpz_clear (c);
}


int
main (int argc, char *argv[])
{
  int    reps = 5000;

  tests_start ();
  TESTS_REPS (reps, argv, argc);

  check_data ();
  check_random (reps);
  check_random_bits (reps);

  tests_end ();
  exit (0);
}
