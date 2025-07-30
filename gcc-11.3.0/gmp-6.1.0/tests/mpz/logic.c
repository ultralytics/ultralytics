/* Test mpz_com, mpz_and, mpz_ior, and mpz_xor.

Copyright 1993, 1994, 1996, 1997, 2001, 2013 Free Software Foundation, Inc.

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

void dump_abort (void);
void debug_mp (mpz_t, int);

int
main (int argc, char **argv)
{
  mpz_t x, y, r1, r2;
  mpz_t t1, t2, t3;
  mp_size_t xsize, ysize;
  int i;
  int reps = 100000;
  gmp_randstate_ptr rands;
  mpz_t bs;
  unsigned long bsi, size_range;

  tests_start ();
  rands = RANDS;

  mpz_init (bs);

  if (argc == 2)
     reps = atoi (argv[1]);

  mpz_init (x);
  mpz_init (y);
  mpz_init (r1);
  mpz_init (r2);
  mpz_init (t1);
  mpz_init (t2);
  mpz_init (t3);

  mpz_set_si (x, -1);
  mpz_set_ui (y, 0);
  for (i = 0; i < 300; i++)
    {
      mpz_mul_2exp (x, x, 1);

      mpz_and (r1, x, x);
      MPZ_CHECK_FORMAT (r1);
      if (mpz_cmp (r1, x) != 0)
	dump_abort ();

      mpz_ior (r2, x, x);
      MPZ_CHECK_FORMAT (r2);
      if (mpz_cmp (r2, x) != 0)
	dump_abort ();

      mpz_xor (t1, x, x);
      MPZ_CHECK_FORMAT (t1);
      if (mpz_cmp_si (t1, 0) != 0)
	dump_abort ();

      mpz_ior (t1, x, y);
      MPZ_CHECK_FORMAT (t1);
      if (mpz_cmp (t1, x) != 0)
	dump_abort ();

      mpz_xor (t2, x, y);
      MPZ_CHECK_FORMAT (t2);
      if (mpz_cmp (t2, x) != 0)
	dump_abort ();

      mpz_com (t2, x);
      MPZ_CHECK_FORMAT (t2);
      mpz_xor (t3, t2, x);
      MPZ_CHECK_FORMAT (t3);
      if (mpz_cmp_si (t3, -1) != 0)
	dump_abort ();
    }

  for (i = 0; i < reps; i++)
    {
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 8 + 2;

      mpz_urandomb (bs, rands, size_range);
      xsize = mpz_get_ui (bs);
      mpz_rrandomb (x, rands, xsize);
      mpz_urandomb (bs, rands, 1);
      bsi = mpz_get_ui (bs);
      if ((bsi & 1) != 0)
	mpz_neg (x, x);

      mpz_urandomb (bs, rands, size_range);
      ysize = mpz_get_ui (bs);
      mpz_rrandomb (y, rands, ysize);
      mpz_urandomb (bs, rands, 1);
      bsi = mpz_get_ui (bs);
      if ((bsi & 1) != 0)
	mpz_neg (y, y);

      mpz_com (r1, x);
      MPZ_CHECK_FORMAT (r1);
      mpz_com (r1, r1);
      MPZ_CHECK_FORMAT (r1);
      if (mpz_cmp (r1, x) != 0)
	dump_abort ();

      mpz_com (r1, y);
      MPZ_CHECK_FORMAT (r1);
      mpz_com (r2, r1);
      MPZ_CHECK_FORMAT (r2);
      if (mpz_cmp (r2, y) != 0)
	dump_abort ();

      mpz_com (t1, x);
      MPZ_CHECK_FORMAT (t1);
      mpz_com (t2, y);
      MPZ_CHECK_FORMAT (t2);
      mpz_and (t3, t1, t2);
      MPZ_CHECK_FORMAT (t3);
      mpz_com (r1, t3);
      MPZ_CHECK_FORMAT (r1);
      mpz_ior (r2, x, y);
      MPZ_CHECK_FORMAT (r2);
      if (mpz_cmp (r1, r2) != 0)
	dump_abort ();

      mpz_com (t1, x);
      MPZ_CHECK_FORMAT (t1);
      mpz_com (t2, y);
      MPZ_CHECK_FORMAT (t2);
      mpz_ior (t3, t1, t2);
      MPZ_CHECK_FORMAT (t3);
      mpz_com (r1, t3);
      MPZ_CHECK_FORMAT (r1);
      mpz_and (r2, x, y);
      MPZ_CHECK_FORMAT (r2);
      if (mpz_cmp (r1, r2) != 0)
	dump_abort ();

      mpz_ior (t1, x, y);
      MPZ_CHECK_FORMAT (t1);
      mpz_and (t2, x, y);
      MPZ_CHECK_FORMAT (t2);
      mpz_com (t3, t2);
      MPZ_CHECK_FORMAT (t3);
      mpz_and (r1, t1, t3);
      MPZ_CHECK_FORMAT (r1);
      mpz_xor (r2, x, y);
      MPZ_CHECK_FORMAT (r2);
      if (mpz_cmp (r1, r2) != 0)
	dump_abort ();
    }

  mpz_clear (bs);
  mpz_clear (x);
  mpz_clear (y);
  mpz_clear (r1);
  mpz_clear (r2);
  mpz_clear (t1);
  mpz_clear (t2);
  mpz_clear (t3);

  tests_end ();
  exit (0);
}

void
dump_abort ()
{
  abort();
}

void
debug_mp (mpz_t x, int base)
{
  mpz_out_str (stderr, base, x); fputc ('\n', stderr);
}
