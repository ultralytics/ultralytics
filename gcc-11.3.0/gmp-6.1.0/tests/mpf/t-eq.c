/* Test mpf_eq.

Copyright 2009, 2012 Free Software Foundation, Inc.

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

#define SZ (2 * sizeof(mp_limb_t))

void insert_random_low_zero_limbs (mpf_t, gmp_randstate_ptr);
void dump_abort (mpf_t, mpf_t, int, int, int, int, int, long);
void hexdump (mpf_t);

void
check_data (void)
{
  static const struct
  {
    struct {
      int        exp, size;
      mp_limb_t  d[10];
    } x, y;
    mp_bitcnt_t bits;
    int want;

  } data[] = {
    { { 0, 0, { 0 } },             { 0, 0, { 0 } },    0, 1 },

    { { 0, 1, { 7 } },             { 0, 1, { 7 } },    0, 1 },
    { { 0, 1, { 7 } },             { 0, 1, { 7 } },   17, 1 },
    { { 0, 1, { 7 } },             { 0, 1, { 7 } }, 4711, 1 },

    { { 0, 1, { 7 } },             { 0, 1, { 6 } },    0, 1 },
    { { 0, 1, { 7 } },             { 0, 1, { 6 } },    2, 1 },
    { { 0, 1, { 7 } },             { 0, 1, { 6 } },    3, 0 },

    { { 0, 0, { 0 } },             { 0, 1, { 1 } },    0, 0 },
    { { 0, 1, { 1 } },             { 0,-1 ,{ 1 } },    0, 0 },
    { { 1, 1, { 1 } },             { 0, 1, { 1 } },    0, 0 },

    { { 0, 1, { 8 } },             { 0, 1, { 4 } },    0, 0 },

    { { 0, 2, { 0, 3 } },          { 0, 1, { 3 } }, 1000, 1 },
  };

  mpf_t  x, y;
  int got, got_swapped;
  int i;
  mp_trace_base = 16;

  for (i = 0; i < numberof (data); i++)
    {
      PTR(x) = (mp_ptr) data[i].x.d;
      SIZ(x) = data[i].x.size;
      EXP(x) = data[i].x.exp;
      PREC(x) = numberof (data[i].x.d);
      MPF_CHECK_FORMAT (x);

      PTR(y) = (mp_ptr) data[i].y.d;
      SIZ(y) = data[i].y.size;
      EXP(y) = data[i].y.exp;
      PREC(y) = numberof (data[i].y.d);
      MPF_CHECK_FORMAT (y);

      got         = mpf_eq (x, y, data[i].bits);
      got_swapped = mpf_eq (y, x, data[i].bits);

      if (got != got_swapped || got != data[i].want)
	{
	  printf ("check_data() wrong result at data[%d]\n", i);
	  mpf_trace ("x   ", x);
	  mpf_trace ("y   ", y);
	  printf ("got         %d\n", got);
	  printf ("got_swapped %d\n", got_swapped);
	  printf ("want        %d\n", data[i].want);
	  abort ();
        }
    }
}

void
check_random (long reps)
{
  unsigned long test;
  gmp_randstate_ptr rands = RANDS;
  mpf_t a, b, x;
  mpz_t ds;
  int hibits, lshift1, lshift2;
  int xtra;

#define HIBITS 10
#define LSHIFT1 10
#define LSHIFT2 10

  mpf_set_default_prec ((1 << HIBITS) + (1 << LSHIFT1) + (1 << LSHIFT2));

  mpz_init (ds);
  mpf_inits (a, b, x, NULL);

  for (test = 0; test < reps; test++)
    {
      mpz_urandomb (ds, rands, HIBITS);
      hibits = mpz_get_ui (ds) + 1;
      mpz_urandomb (ds, rands, hibits);
      mpz_setbit (ds, hibits  - 1);	/* make sure msb is set */
      mpf_set_z (a, ds);
      mpf_set_z (b, ds);

      mpz_urandomb (ds, rands, LSHIFT1);
      lshift1 = mpz_get_ui (ds);
      mpf_mul_2exp (a, a, lshift1 + 1);
      mpf_mul_2exp (b, b, lshift1 + 1);
      mpf_add_ui (a, a, 1);	/* make a one-bit difference */

      mpz_urandomb (ds, rands, LSHIFT2);
      lshift2 = mpz_get_ui (ds);
      mpf_mul_2exp (a, a, lshift2);
      mpf_mul_2exp (b, b, lshift2);
      mpz_urandomb (ds, rands, lshift2);
      mpf_set_z (x, ds);
      mpf_add (a, a, x);
      mpf_add (b, b, x);

      insert_random_low_zero_limbs (a, rands);
      insert_random_low_zero_limbs (b, rands);

      if (mpf_eq (a, b, lshift1 + hibits) == 0 ||
	  mpf_eq (b, a, lshift1 + hibits) == 0)
	{
	  dump_abort (a, b, lshift1 + hibits, lshift1, lshift2, hibits, 1, test);
	}
      for (xtra = 1; xtra < 100; xtra++)
	if (mpf_eq (a, b, lshift1 + hibits + xtra) != 0 ||
	    mpf_eq (b, a, lshift1 + hibits + xtra) != 0)
	  {
	    dump_abort (a, b, lshift1 + hibits + xtra, lshift1, lshift2, hibits, 0, test);
	  }
    }

  mpf_clears (a, b, x, NULL);
  mpz_clear (ds);
}

void
insert_random_low_zero_limbs (mpf_t x, gmp_randstate_ptr rands)
{
  mp_size_t max = PREC(x) - SIZ(x);
  mp_size_t s;
  mpz_t ds; mpz_init (ds);
  mpz_urandomb (ds, rands, 32);
  s = mpz_get_ui (ds) % (max + 1);
  MPN_COPY_DECR (PTR(x) + s, PTR(x), SIZ(x));
  MPN_ZERO (PTR(x), s);
  SIZ(x) += s;
  mpz_clear (ds);
}

void
dump_abort (mpf_t a, mpf_t b, int cmp_prec, int lshift1, int lshift2, int hibits, int want, long test)
{
  printf ("ERROR in test %ld\n", test);
  printf ("want %d got %d from mpf_eq\n", want, 1-want);
  printf ("cmp_prec = %d\n", cmp_prec);
  printf ("lshift1 = %d\n", lshift1);
  printf ("lshift2 = %d\n", lshift2);
  printf ("hibits = %d\n", hibits);
  hexdump (a); puts ("");
  hexdump (b); puts ("");
  abort ();
}

void
hexdump (mpf_t x)
{
  mp_size_t i;
  for (i = ABSIZ(x) - 1; i >= 0; i--)
    {
      gmp_printf ("%0*MX", SZ, PTR(x)[i]);
      if (i != 0)
	printf (" ");
    }
}

int
main (int argc, char *argv[])
{
  long reps = 10000;

  if (argc == 2)
    reps = strtol (argv[1], 0, 0);

  tests_start ();

  check_data ();
  check_random (reps);

  tests_end ();
  exit (0);
}
