/* test mpz_divisible_p and mpz_divisible_ui_p

Copyright 2001, 2009 Free Software Foundation, Inc.

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
check_one (mpz_srcptr a, mpz_srcptr d, int want)
{
  int   got;

  if (mpz_fits_ulong_p (d))
    {
      unsigned long  u = mpz_get_ui (d);
      got = (mpz_divisible_ui_p (a, u) != 0);
      if (want != got)
        {
          printf ("mpz_divisible_ui_p wrong\n");
          printf ("   expected %d got %d\n", want, got);
          mpz_trace ("   a", a);
          printf ("   d=%lu\n", u);
          mp_trace_base = -16;
          mpz_trace ("   a", a);
          printf ("   d=0x%lX\n", u);
          abort ();
        }
    }

  got = (mpz_divisible_p (a, d) != 0);
  if (want != got)
    {
      printf ("mpz_divisible_p wrong\n");
      printf ("   expected %d got %d\n", want, got);
      mpz_trace ("   a", a);
      mpz_trace ("   d", d);
      mp_trace_base = -16;
      mpz_trace ("   a", a);
      mpz_trace ("   d", d);
      abort ();
    }
}

void
check_data (void)
{
  static const struct {
    const char *a;
    const char *d;
    int        want;

  } data[] = {

    { "0",    "0", 1 },
    { "17",   "0", 0 },
    { "0",    "1", 1 },
    { "123",  "1", 1 },
    { "-123", "1", 1 },

    { "0",  "2", 1 },
    { "1",  "2", 0 },
    { "2",  "2", 1 },
    { "-2", "2", 1 },
    { "0x100000000000000000000000000000000", "2", 1 },
    { "0x100000000000000000000000000000001", "2", 0 },

    { "0x3333333333333333", "3", 1 },
    { "0x3333333333333332", "3", 0 },
    { "0x33333333333333333333333333333333", "3", 1 },
    { "0x33333333333333333333333333333332", "3", 0 },

    /* divisor changes from 2 to 1 limb after stripping 2s */
    {          "0x3333333300000000",         "0x180000000",         1 },
    {  "0x33333333333333330000000000000000", "0x18000000000000000", 1 },
    { "0x133333333333333330000000000000000", "0x18000000000000000", 0 },
  };

  mpz_t   a, d;
  int     i;

  mpz_init (a);
  mpz_init (d);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (a, data[i].a, 0);
      mpz_set_str_or_abort (d, data[i].d, 0);
      check_one (a, d, data[i].want);
    }

  mpz_clear (a);
  mpz_clear (d);
}

void
check_random (int reps)
{
  gmp_randstate_ptr rands = RANDS;
  mpz_t   a, d, r;
  int     i;
  int     want;

  mpz_init (a);
  mpz_init (d);
  mpz_init (r);

  for (i = 0; i < reps; i++)
    {
      mpz_erandomb (a, rands, 1 << 19);
      mpz_erandomb_nonzero (d, rands, 1 << 18);

      mpz_fdiv_r (r, a, d);

      want = (mpz_sgn (r) == 0);
      check_one (a, d, want);

      mpz_sub (a, a, r);
      check_one (a, d, 1);

      if (mpz_cmpabs_ui (d, 1L) == 0)
        continue;

      mpz_add_ui (a, a, 1L);
      check_one (a, d, 0);
    }

  mpz_clear (a);
  mpz_clear (d);
  mpz_clear (r);
}


int
main (int argc, char *argv[])
{
  int  reps = 100;

  tests_start ();

  if (argc == 2)
    reps = atoi (argv[1]);

  check_data ();
  check_random (reps);

  tests_end ();
  exit (0);
}
