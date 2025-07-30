/*

Copyright 2011, 2013, Free Software Foundation, Inc.

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

#include "mini-random.h"

static void
set_str (mpz_t r, const char *s)
{
  if (mpz_set_str (r, s, 16) != 0)
    {
      fprintf (stderr, "mpz_set_str failed on input %s\n", s);
      abort ();
    }
}

void
mini_urandomb (mpz_t r, unsigned long bits)
{
  char *s;
  s = hex_urandomb (bits);
  set_str (r, s);
  free (s);
}

void
mini_rrandomb (mpz_t r, unsigned long bits)
{
  char *s;
  s = hex_rrandomb (bits);
  set_str (r, s);
  free (s);
}

void
mini_rrandomb_export (mpz_t r, void *dst, size_t *countp,
		      int order, size_t size, int endian, unsigned long bits)
{
  char *s;
  s = hex_rrandomb_export (dst, countp, order, size, endian, bits);
  set_str (r, s);
  free (s);
}

void
mini_random_op2 (enum hex_random_op op, unsigned long maxbits,
		 mpz_t a, mpz_t r)
{
  char *ap;
  char *rp;

  hex_random_op2 (op, maxbits, &ap, &rp);
  set_str (a, ap);
  set_str (r, rp);

  free (ap);
  free (rp);
}

void
mini_random_op3 (enum hex_random_op op, unsigned long maxbits,
		 mpz_t a, mpz_t b, mpz_t r)
{
  char *ap;
  char *bp;
  char *rp;

  hex_random_op3 (op, maxbits, &ap, &bp, &rp);
  set_str (a, ap);
  set_str (b, bp);
  set_str (r, rp);

  free (ap);
  free (bp);
  free (rp);
}

void
mini_random_op4 (enum hex_random_op op, unsigned long maxbits,
		 mpz_t a, mpz_t b, mpz_t c, mpz_t d)
{
  char *ap;
  char *bp;
  char *cp;
  char *dp;

  hex_random_op4 (op, maxbits, &ap, &bp, &cp, &dp);
  set_str (a, ap);
  set_str (b, bp);
  set_str (c, cp);
  set_str (d, dp);

  free (ap);
  free (bp);
  free (cp);
  free (dp);
}

void
mini_random_bit_op (enum hex_random_op op, unsigned long maxbits,
			 mpz_t a, mp_bitcnt_t *b, mpz_t r)
{
  char *ap;
  char *rp;

  hex_random_bit_op (op, maxbits, &ap, b, &rp);
  set_str (a, ap);
  set_str (r, rp);

  free (ap);
  free (rp);
}

void
mini_random_scan_op (enum hex_random_op op, unsigned long maxbits,
		     mpz_t a, mp_bitcnt_t *b, mp_bitcnt_t *r)
{
  char *ap;

  hex_random_scan_op (op, maxbits, &ap, b, r);
  set_str (a, ap);

  free (ap);
}
