/* Test conversion using mpz_get_str and mpz_set_str.

Copyright 1993, 1994, 1996, 1999-2002, 2006, 2007 Free Software Foundation,
Inc.

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
#include <string.h> /* for strlen */

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

void debug_mp (mpz_t, int);


void
string_urandomb (char *bp, size_t len, int base, gmp_randstate_ptr rands)
{
  mpz_t bs;
  unsigned long bsi;
  int d, l;
  const char *collseq = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  mpz_init (bs);

  mpz_urandomb (bs, rands, 32);
  bsi = mpz_get_ui (bs);
  d = bsi % base;
  while (len != 0)
    {
      l = (bsi >> 16) % 20;
      l = MIN (l, len);

      memset (bp, collseq[d], l);

      len -= l;
      bp += l;

      mpz_urandomb (bs, rands, 32);
      bsi = mpz_get_ui (bs);
      d = bsi & 0xfff;
      if (d >= base)
	d = 0;
    }

  bp[0] = '\0';
  mpz_clear (bs);
}

int
main (int argc, char **argv)
{
  mpz_t op1, op2;
  mp_size_t size;
  int i;
  int reps = 2000;
  char *str, *buf, *bp;
  int base;
  gmp_randstate_ptr rands;
  mpz_t bs;
  unsigned long bsi, size_range;
  size_t len;

  tests_start ();
  TESTS_REPS (reps, argv, argc);

  rands = RANDS;

  mpz_init (bs);

  mpz_init (op1);
  mpz_init (op2);

  for (i = 0; i < reps; i++)
    {
      /* 1. Generate random mpz_t and convert to a string and back to mpz_t
	 again.  */
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 17 + 2;	/* 2..18 */
      mpz_urandomb (bs, rands, size_range);	/* 3..262144 bits */
      size = mpz_get_ui (bs);
      mpz_rrandomb (op1, rands, size);

      mpz_urandomb (bs, rands, 1);
      bsi = mpz_get_ui (bs);
      if ((bsi & 1) != 0)
	mpz_neg (op1, op1);

      mpz_urandomb (bs, rands, 32);
      bsi = mpz_get_ui (bs);
      base = bsi % 62 + 1;
      if (base == 1)
	base = 0;

      str = mpz_get_str ((char *) 0, base, op1);
      mpz_set_str_or_abort (op2, str, base);

      if (mpz_cmp (op1, op2))
	{
	  fprintf (stderr, "ERROR, op1 and op2 different in test %d\n", i);
	  fprintf (stderr, "str  = %s\n", str);
	  fprintf (stderr, "base = %d\n", base);
	  fprintf (stderr, "op1  = "); debug_mp (op1, -16);
	  fprintf (stderr, "op2  = "); debug_mp (op2, -16);
	  abort ();
	}

      (*__gmp_free_func) (str, strlen (str) + 1);

      /* 2. Generate random string and convert to mpz_t and back to a string
	 again.  */
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 16 + 1;	/* 1..16 */
      mpz_urandomb (bs, rands, size_range);	/* 1..65536 digits */
      len = mpz_get_ui (bs) + 1;
      buf = (char *) (*__gmp_allocate_func) (len + 1);
      if (base == 0)
	base = 10;
      string_urandomb (buf, len, base, rands);

      mpz_set_str_or_abort (op1, buf, base);
      str = mpz_get_str ((char *) 0, base, op1);

      /* Skip over leading zeros, but don't leave the string at zero length. */
      for (bp = buf; bp[0] == '0' && bp[1] != '\0'; bp++)
	;

      if (strcasecmp (str, bp) != 0)
	{
	  fprintf (stderr, "ERROR, str and buf different in test %d\n", i);
	  fprintf (stderr, "str  = %s\n", str);
	  fprintf (stderr, "buf  = %s\n", buf);
	  fprintf (stderr, "base = %d\n", base);
	  fprintf (stderr, "op1  = "); debug_mp (op1, -16);
	  abort ();
	}

      (*__gmp_free_func) (buf, len + 1);
      (*__gmp_free_func) (str, strlen (str) + 1);
    }

  mpz_clear (bs);
  mpz_clear (op1);
  mpz_clear (op2);

  tests_end ();
  exit (0);
}

void
debug_mp (mpz_t x, int base)
{
  mpz_out_str (stderr, base, x); fputc ('\n', stderr);
}
