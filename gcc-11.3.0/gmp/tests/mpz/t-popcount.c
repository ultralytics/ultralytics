/* Test mpz_popcount.

Copyright 2001, 2005 Free Software Foundation, Inc.

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
check_onebit (void)
{
  mpz_t          n;
  unsigned long  i, got;

  mpz_init (n);
  for (i = 0; i < 5 * GMP_LIMB_BITS; i++)
    {
      mpz_setbit (n, i);
      got = mpz_popcount (n);
      if (got != 1)
	{
	  printf ("mpz_popcount wrong on single bit at %lu\n", i);
	  printf ("   got %lu, want 1\n", got);
	  abort();
	}
      mpz_clrbit (n, i);
    }
  mpz_clear (n);
}


void
check_data (void)
{
  static const struct {
    const char     *n;
    unsigned long  want;
  } data[] = {
    { "-1", ~ (unsigned long) 0 },
    { "-12345678", ~ (unsigned long) 0 },
    { "0", 0 },
    { "1", 1 },
    { "3", 2 },
    { "5", 2 },
    { "0xFFFF", 16 },
    { "0xFFFFFFFF", 32 },
    { "0xFFFFFFFFFFFFFFFF", 64 },
    { "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 128 },
  };

  unsigned long   got;
  int    i;
  mpz_t  n;

  mpz_init (n);
  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (n, data[i].n, 0);
      got = mpz_popcount (n);
      if (got != data[i].want)
	{
	  printf ("mpz_popcount wrong at data[%d]\n", i);
	  printf ("   n     \"%s\"\n", data[i].n);
	  printf ("         ");   mpz_out_str (stdout, 10, n); printf ("\n");
	  printf ("         0x"); mpz_out_str (stdout, 16, n); printf ("\n");
	  printf ("   got   %lu\n", got);
	  printf ("   want  %lu\n", data[i].want);
	  abort ();
	}
    }
  mpz_clear (n);
}

unsigned long
refmpz_popcount (mpz_t arg)
{
  mp_size_t n, i;
  unsigned long cnt;
  mp_limb_t x;

  n = SIZ(arg);
  if (n < 0)
    return ~(unsigned long) 0;

  cnt = 0;
  for (i = 0; i < n; i++)
    {
      x = PTR(arg)[i];
      while (x != 0)
	{
	  cnt += (x & 1);
	  x >>= 1;
	}
    }
  return cnt;
}

void
check_random (void)
{
  gmp_randstate_ptr rands;
  mpz_t bs;
  mpz_t arg;
  unsigned long arg_size, size_range;
  unsigned long got, ref;
  int i;

  rands = RANDS;

  mpz_init (bs);
  mpz_init (arg);

  for (i = 0; i < 10000; i++)
    {
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 11 + 2; /* 0..4096 bit operands */

      mpz_urandomb (bs, rands, size_range);
      arg_size = mpz_get_ui (bs);
      mpz_rrandomb (arg, rands, arg_size);

      got = mpz_popcount (arg);
      ref = refmpz_popcount (arg);
      if (got != ref)
	{
	  printf ("mpz_popcount wrong on random\n");
	  printf ("         ");   mpz_out_str (stdout, 10, arg); printf ("\n");
	  printf ("         0x"); mpz_out_str (stdout, 16, arg); printf ("\n");
	  printf ("   got   %lu\n", got);
	  printf ("   want  %lu\n", ref);
	  abort ();
	}
    }
  mpz_clear (arg);
  mpz_clear (bs);
}

int
main (void)
{
  tests_start ();

  check_onebit ();
  check_data ();
  check_random ();

  tests_end ();
  exit (0);
}
