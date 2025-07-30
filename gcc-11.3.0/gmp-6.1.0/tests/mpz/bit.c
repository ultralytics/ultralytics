/* Test mpz_setbit, mpz_clrbit, mpz_tstbit.

Copyright 1997, 2000-2003, 2012, 2013 Free Software Foundation, Inc.

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

#ifndef SIZE
#define SIZE 4
#endif


void
debug_mp (mpz_srcptr x, int base)
{
  mpz_out_str (stdout, base, x); fputc ('\n', stdout);
}


/* exercise the case where mpz_clrbit or mpz_combit ends up extending a
   value like -2^(k*GMP_NUMB_BITS-1) when clearing bit k*GMP_NUMB_BITS-1.  */
/* And vice-versa. */
void
check_clr_extend (void)
{
  mpz_t          got, want;
  unsigned long  i;
  int            f;

  mpz_init (got);
  mpz_init (want);

  for (i = 1; i < 5; i++)
    {
      for (f = 0; f <= 1; f++)
	{
	  /* lots of 1 bits in _mp_d */
	  mpz_set_si (got, 1L);
	  mpz_mul_2exp (got, got, 10*GMP_NUMB_BITS);
	  mpz_sub_ui (got, got, 1L);

	  /* value -2^(n-1) representing ..11100..00 */
	  mpz_set_si (got, -1L);
	  mpz_mul_2exp (got, got, i*GMP_NUMB_BITS-1);

	  /* complement bit n, giving ..11000..00 which is -2^n */
	  if (f == 0)
	    mpz_clrbit (got, i*GMP_NUMB_BITS-1);
	  else
	    mpz_combit (got, i*GMP_NUMB_BITS-1);
	  MPZ_CHECK_FORMAT (got);

	  mpz_set_si (want, -1L);
	  mpz_mul_2exp (want, want, i*GMP_NUMB_BITS);

	  if (mpz_cmp (got, want) != 0)
	    {
	      if (f == 0)
		printf ("mpz_clrbit: ");
	      else
		printf ("mpz_combit: ");
	      printf ("wrong after extension\n");
	      mpz_trace ("got ", got);
	      mpz_trace ("want", want);
	      abort ();
	    }

	  /* complement bit n, going back to ..11100..00 which is -2^(n-1) */
	  if (f == 0)
	    mpz_setbit (got, i*GMP_NUMB_BITS-1);
	  else
	    mpz_combit (got, i*GMP_NUMB_BITS-1);
	  MPZ_CHECK_FORMAT (got);

	  mpz_set_si (want, -1L);
	  mpz_mul_2exp (want, want, i*GMP_NUMB_BITS - 1);

	  if (mpz_cmp (got, want) != 0)
	    {
	      if (f == 0)
		printf ("mpz_setbit: ");
	      else
		printf ("mpz_combit: ");
	      printf ("wrong after shrinking\n");
	      mpz_trace ("got ", got);
	      mpz_trace ("want", want);
	      abort ();
	    }
	}
    }

  mpz_clear (got);
  mpz_clear (want);
}

void
check_com_negs (void)
{
  static const struct {
    unsigned long  bit;
    mp_size_t      inp_size;
    mp_limb_t      inp_n[5];
    mp_size_t      want_size;
    mp_limb_t      want_n[5];
  } data[] = {
    { GMP_NUMB_BITS,   2, { 1, 1 },  1, { 1 } },
    { GMP_NUMB_BITS+1, 2, { 1, 1 },  2, { 1, 3 } },

    { GMP_NUMB_BITS,   2, { 0, 1 },  2, { 0, 2 } },
    { GMP_NUMB_BITS+1, 2, { 0, 1 },  2, { 0, 3 } },
  };
  mpz_t  inp, got, want;
  int    i;

  mpz_init (got);
  mpz_init (want);
  mpz_init (inp);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_n (inp, data[i].inp_n, data[i].inp_size);
      mpz_neg (inp, inp);

      mpz_set_n (want, data[i].want_n, data[i].want_size);
      mpz_neg (want, want);

      mpz_set (got, inp);
      mpz_combit (got, data[i].bit);

      if (mpz_cmp (got, want) != 0)
	{
	  printf ("mpz_combit: wrong on neg data[%d]\n", i);
	  mpz_trace ("inp ", inp);
	  printf    ("bit %lu\n", data[i].bit);
	  mpz_trace ("got ", got);
	  mpz_trace ("want", want);
	  abort ();
	}
    }

  mpz_clear (inp);
  mpz_clear (got);
  mpz_clear (want);
}

/* See that mpz_tstbit matches a twos complement calculated explicitly, for
   various low zeros.  */
void
check_tstbit (void)
{
#define MAX_ZEROS  3
#define NUM_LIMBS  3

  mp_limb_t      pos[1+NUM_LIMBS+MAX_ZEROS];
  mp_limb_t      neg[1+NUM_LIMBS+MAX_ZEROS];
  mpz_t          z;
  unsigned long  i;
  int            zeros, low1;
  int            got, want;

  mpz_init (z);
  for (zeros = 0; zeros <= MAX_ZEROS; zeros++)
    {
      MPN_ZERO (pos, numberof(pos));
      mpn_random2 (pos+zeros, (mp_size_t) NUM_LIMBS);

      for (low1 = 0; low1 <= 1; low1++)
	{
	  if (low1)
	    pos[0] |= 1;

	  refmpn_neg (neg, pos, (mp_size_t) numberof(neg));
	  mpz_set_n (z, neg, (mp_size_t) numberof(neg));
	  mpz_neg (z, z);

	  for (i = 0; i < numberof(pos)*GMP_NUMB_BITS; i++)
	    {
	      got = mpz_tstbit (z, i);
	      want = refmpn_tstbit (pos, i);
	      if (got != want)
		{
		  printf ("wrong at bit %lu, with %d zeros\n", i, zeros);
		  printf ("z neg "); debug_mp (z, -16);
		  mpz_set_n (z, pos, (mp_size_t) numberof(pos));
		  printf ("pos   "); debug_mp (z, -16);
		  mpz_set_n (z, neg, (mp_size_t) numberof(neg));
		  printf ("neg   "); debug_mp (z, -16);
		  exit (1);
		}
	    }
	}
    }
  mpz_clear (z);
}


void
check_single (void)
{
  mpz_t  x;
  int    limb, offset, initial;
  unsigned long  bit;

  mpz_init (x);

  for (limb = 0; limb < 4; limb++)
    {
      for (offset = (limb==0 ? 0 : -2); offset <= 2; offset++)
	{
	  for (initial = 1; initial >= -1; initial--)
	    {
	      mpz_set_si (x, (long) initial);

	      bit = (unsigned long) limb*GMP_LIMB_BITS + offset;

	      mpz_clrbit (x, bit);
	      MPZ_CHECK_FORMAT (x);
	      if (mpz_tstbit (x, bit) != 0)
		{
		  printf ("check_single(): expected 0\n");
		  abort ();
		}

	      mpz_setbit (x, bit);
	      MPZ_CHECK_FORMAT (x);
	      if (mpz_tstbit (x, bit) != 1)
		{
		  printf ("check_single(): expected 1\n");
		  abort ();
		}

	      mpz_clrbit (x, bit);
	      MPZ_CHECK_FORMAT (x);
	      if (mpz_tstbit (x, bit) != 0)
		{
		  printf ("check_single(): expected 0\n");
		  abort ();
		}

	      mpz_combit (x, bit);
	      MPZ_CHECK_FORMAT (x);
	      if (mpz_tstbit (x, bit) != 1)
		{
		  printf ("check_single(): expected 1\n");
		  abort ();
		}

	      mpz_combit (x, bit);
	      MPZ_CHECK_FORMAT (x);
	      if (mpz_tstbit (x, bit) != 0)
		{
		  printf ("check_single(): expected 0\n");
		  abort ();
		}
	    }
	}
    }

  mpz_clear (x);
}


void
check_random (int argc, char *argv[])
{
  mpz_t x, s0, s1, s2, s3, m;
  mp_size_t xsize;
  int i;
  int reps = 100000;
  int bit0, bit1, bit2, bit3;
  unsigned long int bitindex;
  const char  *s = "";

  if (argc == 2)
    reps = atoi (argv[1]);

  mpz_init (x);
  mpz_init (s0);
  mpz_init (s1);
  mpz_init (s2);
  mpz_init (s3);
  mpz_init (m);

  for (i = 0; i < reps; i++)
    {
      xsize = urandom () % (2 * SIZE) - SIZE;
      mpz_random2 (x, xsize);
      bitindex = urandom () % SIZE;

      mpz_set (s0, x);
      bit0 = mpz_tstbit (x, bitindex);
      mpz_setbit (x, bitindex);
      MPZ_CHECK_FORMAT (x);

      mpz_set (s1, x);
      bit1 = mpz_tstbit (x, bitindex);
      mpz_clrbit (x, bitindex);
      MPZ_CHECK_FORMAT (x);

      mpz_set (s2, x);
      bit2 = mpz_tstbit (x, bitindex);
      mpz_combit (x, bitindex);
      MPZ_CHECK_FORMAT (x);

      mpz_set (s3, x);
      bit3 = mpz_tstbit (x, bitindex);

#define FAIL(str) do { s = str; goto fail; } while (0)

      if (bit1 != 1)  FAIL ("bit1 != 1");
      if (bit2 != 0)  FAIL ("bit2 != 0");
      if (bit3 != 1)  FAIL ("bit3 != 1");

      if (bit0 == 0)
	{
	  if (mpz_cmp (s0, s1) == 0 || mpz_cmp (s0, s2) != 0 || mpz_cmp (s0, s3) == 0)
	    abort ();
	}
      else
	{
	  if (mpz_cmp (s0, s1) != 0 || mpz_cmp (s0, s2) == 0 || mpz_cmp (s0, s3) != 0)
	    abort ();
	}

      if (mpz_cmp (s1, s2) == 0 || mpz_cmp (s1, s3) != 0)
	abort ();
      if (mpz_cmp (s2, s3) == 0)
	abort ();

      mpz_combit (x, bitindex);
      MPZ_CHECK_FORMAT (x);
      if (mpz_cmp (s2, x) != 0)
	abort ();

      mpz_clrbit (x, bitindex);
      MPZ_CHECK_FORMAT (x);
      if (mpz_cmp (s2, x) != 0)
	abort ();

      mpz_ui_pow_ui (m, 2L, bitindex);
      MPZ_CHECK_FORMAT (m);
      mpz_ior (x, s0, m);
      MPZ_CHECK_FORMAT (x);
      if (mpz_cmp (x, s3) != 0)
	abort ();

      mpz_com (m, m);
      MPZ_CHECK_FORMAT (m);
      mpz_and (x, s0, m);
      MPZ_CHECK_FORMAT (x);
      if (mpz_cmp (x, s2) != 0)
	abort ();
    }

  mpz_clear (x);
  mpz_clear (s0);
  mpz_clear (s1);
  mpz_clear (s2);
  mpz_clear (s3);
  mpz_clear (m);
  return;


 fail:
  printf ("%s\n", s);
  printf ("bitindex = %lu\n", bitindex);
  printf ("x = "); mpz_out_str (stdout, -16, x); printf (" hex\n");
  exit (1);
}



int
main (int argc, char *argv[])
{
  tests_start ();
  mp_trace_base = -16;

  check_clr_extend ();
  check_com_negs ();
  check_tstbit ();
  check_random (argc, argv);
  check_single ();

  tests_end ();
  exit (0);
}
