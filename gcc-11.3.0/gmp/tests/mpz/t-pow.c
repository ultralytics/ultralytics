/* Test mpz_pow_ui and mpz_ui_pow_ui.

Copyright 1997, 1999-2001 Free Software Foundation, Inc.

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
check_one (mpz_srcptr want, mpz_srcptr base, unsigned long exp)
{
  mpz_t  got;

  mpz_init (got);

  MPZ_CHECK_FORMAT (want);

  mpz_pow_ui (got, base, exp);
  if (mpz_cmp (got, want))
    {
      printf ("mpz_pow_ui wrong\n");
      mpz_trace ("  base", base);
      printf    ("  exp = %lu (0x%lX)\n", exp, exp);
      mpz_trace ("  got ", got);
      mpz_trace ("  want", want);
      abort ();
    }

  mpz_set (got, base);
  mpz_pow_ui (got, got, exp);
  if (mpz_cmp (got, want))
    {
      printf ("mpz_pow_ui wrong\n");
      mpz_trace ("  base", base);
      printf    ("  exp = %lu (0x%lX)\n", exp, exp);
      mpz_trace ("  got ", got);
      mpz_trace ("  want", want);
      abort ();
    }

  if (mpz_fits_ulong_p (base))
    {
      unsigned long  base_u = mpz_get_ui (base);
      mpz_ui_pow_ui (got, base_u, exp);
      if (mpz_cmp (got, want))
	{
	  printf    ("mpz_ui_pow_ui wrong\n");
	  printf    ("  base=%lu (0x%lX)\n", base_u, base_u);
	  printf    ("  exp = %lu (0x%lX)\n", exp, exp);
	  mpz_trace ("  got ", got);
	  mpz_trace ("  want", want);
	  abort ();
	}
    }

  mpz_clear (got);
}

void
check_base (mpz_srcptr base)
{
  unsigned long  exp;
  mpz_t          want;

  mpz_init (want);
  mpz_set_ui (want, 1L);

  for (exp = 0; exp < 20; exp++)
    {
      check_one (want, base, exp);
      mpz_mul (want, want, base);
    }

  mpz_clear (want);
}

void
check_various (void)
{
  static const struct {
    const char *base;
  } data[] = {
    { "0" },
    { "1" },
    { "2" },
    { "3" },
    { "4" },
    { "5" },
    { "6" },
    { "10" },
    { "15" },
    { "16" },

    { "0x1F" },
    { "0xFF" },
    { "0x1001" },
    { "0xFFFF" },
    { "0x10000001" },
    { "0x1000000000000001" },

    /* actual size closest to estimate */
    { "0xFFFFFFFF" },
    { "0xFFFFFFFFFFFFFFFF" },

    /* same after rshift */
    { "0xFFFFFFFF0" },
    { "0xFFFFFFFF00" },
    { "0xFFFFFFFFFFFFFFFF0" },
    { "0xFFFFFFFFFFFFFFFF00" },

    /* change from 2 limbs to 1 after rshift */
    { "0x180000000" },
    { "0x18000000000000000" },

    /* change from 3 limbs to 2 after rshift */
    { "0x18000000100000000" },
    { "0x180000000000000010000000000000000" },

    /* handling of absolute value */
    { "-0x80000000" },
    { "-0x8000000000000000" },

    /* low zero limb, and size>2, checking argument overlap detection */
    { "0x3000000000000000300000000000000030000000000000000" },
  };

  mpz_t  base;
  int    i;

  mpz_init (base);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (base, data[i].base, 0);
      check_base (base);
    }

  mpz_clear (base);
}

void
check_random (int reps)
{
  mpz_t              base, want;
  mp_size_t          base_size;
  int                i;
  unsigned long      size_range, exp;
  gmp_randstate_ptr  rands = RANDS;

  mpz_init (base);
  mpz_init (want);

  for (i = 0; i < reps; i++)
    {
      /* exponentially random 0 to 2^13 bits for base */
      mpz_urandomb (want, rands, 32);
      size_range = mpz_get_ui (want) % 12 + 2;
      mpz_urandomb (want, rands, size_range);
      base_size = mpz_get_ui (want);
      mpz_rrandomb (base, rands, base_size);

      /* randomly signed base */
      mpz_urandomb (want, rands, 2);
      if ((mpz_get_ui (want) & 1) != 0)
	mpz_neg (base, base);

      /* random 5 bits for exponent */
      mpz_urandomb (want, rands, 5L);
      exp = mpz_get_ui (want);

      refmpz_pow_ui (want, base, exp);
      check_one (want, base, exp);
    }

  mpz_clear (base);
  mpz_clear (want);
}

int
main (int argc, char **argv)
{
  int reps = 5000;

  /* dummy call to drag in refmpn.o for testing mpz/n_pow_ui.c with
     refmpn_mul_2 */
  refmpn_zero_p (NULL, (mp_size_t) 0);

  tests_start ();
  mp_trace_base = -16;

  if (argc == 2)
     reps = atoi (argv[1]);

  check_various ();
  check_random (reps);

  tests_end ();
  exit (0);
}
