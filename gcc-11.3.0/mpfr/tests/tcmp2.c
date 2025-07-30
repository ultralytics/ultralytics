/* Test file for mpfr_cmp2.

Copyright 1999-2003, 2006-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include <stdio.h>
#include <stdlib.h>

#include "mpfr-test.h"

/* set bit n of x to b, where bit 0 is the most significant one */
static void
set_bit (mpfr_t x, unsigned int n, int b)
{
  unsigned l;
  mp_size_t xn;

  xn = (MPFR_PREC(x) - 1) / mp_bits_per_limb;
  l = n / mp_bits_per_limb;
  n %= mp_bits_per_limb;
  n = mp_bits_per_limb - 1 - n;
  if (b)
    MPFR_MANT(x)[xn - l] |= (mp_limb_t) 1 << n;
  else
    MPFR_MANT(x)[xn - l] &= ~((mp_limb_t) 1 << n);
}

/* check that for x = 1.u 1 v 0^k low(x)
                  y = 1.u 0 v 1^k low(y)
   mpfr_cmp2 (x, y) returns 1 + |u| + |v| + k for low(x) >= low(y),
                        and 1 + |u| + |v| + k + 1 otherwise */
static void
worst_cases (void)
{
  mpfr_t x, y;
  unsigned int i, j, k, b, expected;
  mpfr_prec_t l;

  mpfr_init2 (x, 200);
  mpfr_init2 (y, 200);

  mpfr_set_ui (y, 1, MPFR_RNDN);
  for (i = 1; i < MPFR_PREC(x); i++)
    {
      mpfr_set_ui (x, 1, MPFR_RNDN);
      mpfr_div_2exp (y, y, 1, MPFR_RNDN); /* y = 1/2^i */

      l = 0;
      if (mpfr_cmp2 (x, y, &l) <= 0 || l != 1)
        {
          printf ("Error in mpfr_cmp2:\nx=");
          mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
          printf ("\ny=");
          mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
          printf ("\ngot %lu instead of 1\n", (unsigned long) l);
          exit (1);
        }

      mpfr_add (x, x, y, MPFR_RNDN); /* x = 1 + 1/2^i */
      l = 0;
      if (mpfr_cmp2 (x, y, &l) <= 0 || l != 0)
        {
          printf ("Error in mpfr_cmp2:\nx=");
          mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
          printf ("\ny=");
          mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
          printf ("\ngot %lu instead of 0\n", (unsigned long) l);
          exit (1);
        }
    }

  for (i = 0; i < 64; i++) /* |u| = i */
    {
      mpfr_urandomb (x, RANDS);
      mpfr_set (y, x, MPFR_RNDN);
      set_bit (x, i + 1, 1);
      set_bit (y, i + 1, 0);
      for (j = 0; j < 64; j++) /* |v| = j */
        {
          b = randlimb () % 2;
          set_bit (x, i + j + 2, b);
          set_bit (y, i + j + 2, b);
          for (k = 0; k < 64; k++)
            {
              if (k)
                set_bit (x, i + j + k + 1, 0);
              if (k)
                set_bit (y, i + j + k + 1, 1);
              set_bit (x, i + j + k + 2, 1);
              set_bit (y, i + j + k + 2, 0);
              l = 0; mpfr_cmp2 (x, y, &l);
              expected = i + j + k + 1;
              if (l != expected)
                {
                  printf ("Error in mpfr_cmp2:\nx=");
                  mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
                  printf ("\ny=");
                  mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
                  printf ("\ngot %lu instead of %u\n",
                          (unsigned long) l, expected);
                  exit (1);
                }
              set_bit (x, i + j + k + 2, 0);
              set_bit (x, i + j + k + 3, 0);
              set_bit (y, i + j + k + 3, 1);
              l = 0; mpfr_cmp2 (x, y, &l);
              expected = i + j + k + 2;
              if (l != expected)
                {
                  printf ("Error in mpfr_cmp2:\nx=");
                  mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
                  printf ("\ny=");
                  mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
                  printf ("\ngot %lu instead of %u\n",
                          (unsigned long) l, expected);
                  exit (1);
                }
            }
        }
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
tcmp2 (double x, double y, int i)
{
  mpfr_t xx, yy;
  mpfr_prec_t j;

  if (i == -1)
    {
      if (x == y)
        i = 53;
      else
        i = (int) (__gmpfr_floor_log2 (x) - __gmpfr_floor_log2 (x - y));
    }
  mpfr_init2(xx, 53); mpfr_init2(yy, 53);
  mpfr_set_d (xx, x, MPFR_RNDN);
  mpfr_set_d (yy, y, MPFR_RNDN);
  j = 0;
  if (mpfr_cmp2 (xx, yy, &j) == 0)
    {
      if (x != y)
        {
          printf ("Error in mpfr_cmp2 for\nx=");
          mpfr_out_str (stdout, 2, 0, xx, MPFR_RNDN);
          printf ("\ny=");
          mpfr_out_str (stdout, 2, 0, yy, MPFR_RNDN);
          printf ("\ngot sign 0 for x != y\n");
          exit (1);
        }
    }
  else if (j != (unsigned) i)
    {
      printf ("Error in mpfr_cmp2 for\nx=");
      mpfr_out_str (stdout, 2, 0, xx, MPFR_RNDN);
      printf ("\ny=");
      mpfr_out_str (stdout, 2, 0, yy, MPFR_RNDN);
      printf ("\ngot %lu instead of %d\n", (unsigned long) j, i);
      exit (1);
    }
  mpfr_clear(xx); mpfr_clear(yy);
}

static void
special (void)
{
  mpfr_t x, y;
  mpfr_prec_t j;

  mpfr_init (x); mpfr_init (y);

  /* bug found by Nathalie Revol, 21 March 2001 */
  mpfr_set_prec (x, 65);
  mpfr_set_prec (y, 65);
  mpfr_set_str_binary (x, "0.10000000000000000000000000000000000001110010010110100110011110000E1");
  mpfr_set_str_binary (y, "0.11100100101101001100111011111111110001101001000011101001001010010E-35");
  j = 0;
  if (mpfr_cmp2 (x, y, &j) <= 0 || j != 1)
    {
      printf ("Error in mpfr_cmp2:\n");
      printf ("x=");
      mpfr_print_binary (x);
      puts ("");
      printf ("y=");
      mpfr_print_binary (y);
      puts ("");
      printf ("got %lu, expected 1\n", (unsigned long) j);
      exit (1);
    }

  mpfr_set_prec(x, 127); mpfr_set_prec(y, 127);
  mpfr_set_str_binary(x, "0.1011010000110111111000000101011110110001000101101011011110010010011110010000101101000010011001100110010000000010110000101000101E6");
  mpfr_set_str_binary(y, "0.1011010000110111111000000101011011111100011101000011001111000010100010100110110100110010011001100110010000110010010110000010110E6");
  j = 0;
  if (mpfr_cmp2(x, y, &j) <= 0 || j != 32)
    {
      printf ("Error in mpfr_cmp2:\n");
      printf ("x="); mpfr_print_binary(x); puts ("");
      printf ("y="); mpfr_print_binary(y); puts ("");
      printf ("got %lu, expected 32\n", (unsigned long) j);
      exit (1);
    }

  mpfr_set_prec (x, 128); mpfr_set_prec (y, 239);
  mpfr_set_str_binary (x, "0.10001000110110000111011000101011111100110010010011001101000011111010010110001000000010100110100111111011011010101100100000000000E167");
  mpfr_set_str_binary (y, "0.10001000110110000111011000101011111100110010010011001101000011111010010110001000000010100110100111111011011010101100011111111111111111111111111111111111111111111111011111100101011100011001101000100111000010000000000101100110000111111000101E167");
  j = 0;
  if (mpfr_cmp2(x, y, &j) <= 0 || j != 164)
    {
      printf ("Error in mpfr_cmp2:\n");
      printf ("x="); mpfr_print_binary(x); puts ("");
      printf ("y="); mpfr_print_binary(y); puts ("");
      printf ("got %lu, expected 164\n", (unsigned long) j);
      exit (1);
    }

  /* bug found by Nathalie Revol, 29 March 2001 */
  mpfr_set_prec (x, 130); mpfr_set_prec (y, 130);
  mpfr_set_str_binary (x, "0.1100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000E2");
  mpfr_set_str_binary (y, "0.1011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100E2");
  j = 0;
  if (mpfr_cmp2(x, y, &j) <= 0 || j != 127)
    {
      printf ("Error in mpfr_cmp2:\n");
      printf ("x="); mpfr_print_binary(x); puts ("");
      printf ("y="); mpfr_print_binary(y); puts ("");
      printf ("got %lu, expected 127\n", (unsigned long) j);
      exit (1);
    }

  /* bug found by Nathalie Revol, 2 Apr 2001 */
  mpfr_set_prec (x, 65); mpfr_set_prec (y, 65);
  mpfr_set_ui (x, 5, MPFR_RNDN);
  mpfr_set_str_binary (y, "0.10011111111111111111111111111111111111111111111111111111111111101E3");
  j = 0;
  if (mpfr_cmp2(x, y, &j) <= 0 || j != 63)
    {
      printf ("Error in mpfr_cmp2:\n");
      printf ("x="); mpfr_print_binary(x); puts ("");
      printf ("y="); mpfr_print_binary(y); puts ("");
      printf ("got %lu, expected 63\n", (unsigned long) j);
      exit (1);
    }

  /* bug found by Nathalie Revol, 2 Apr 2001 */
  mpfr_set_prec (x, 65); mpfr_set_prec (y, 65);
  mpfr_set_str_binary (x, "0.10011011111000101001110000000000000000000000000000000000000000000E-69");
  mpfr_set_str_binary (y, "0.10011011111000101001101111111111111111111111111111111111111111101E-69");
  j = 0;
  if (mpfr_cmp2(x, y, &j) <= 0 || j != 63)
    {
      printf ("Error in mpfr_cmp2:\n");
      printf ("x="); mpfr_print_binary(x); puts ("");
      printf ("y="); mpfr_print_binary(y); puts ("");
      printf ("got %lu, expected 63\n", (unsigned long) j);
      exit (1);
    }

  mpfr_set_prec (x, 65);
  mpfr_set_prec (y, 32);
  mpfr_set_str_binary (x, "1.1110111011110001110111011111111111101000011001011100101100101101");
  mpfr_set_str_binary (y, "0.11101110111100011101110111111111");
  if (mpfr_cmp2 (x, y, &j) <= 0 || j != 0)
    {
      printf ("Error in mpfr_cmp2 (1)\n");
      exit (1);
    }

  mpfr_set_prec (x, 2 * GMP_NUMB_BITS);
  mpfr_set_prec (y, GMP_NUMB_BITS);
  mpfr_set_ui (x, 1, MPFR_RNDN); /* x = 1 */
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_nextbelow (y);            /* y = 1 - 2^(-GMP_NUMB_BITS) */
  mpfr_cmp2 (x, y, &j);
  if (mpfr_cmp2 (x, y, &j) <= 0 || j != GMP_NUMB_BITS)
    {
      printf ("Error in mpfr_cmp2 (2)\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

int
main (void)
{
  int i;
  long j;
  double x, y, z;

  tests_start_mpfr ();
  mpfr_test_init ();

  worst_cases ();
  special ();
  tcmp2 (5.43885304644369510000e+185, -1.87427265794105340000e-57, 1);
  tcmp2 (1.06022698059744327881e+71, 1.05824655795525779205e+71, -1);
  tcmp2 (1.0, 1.0, 53);
  /* warning: cmp2 does not allow 0 as input */
  for (x = 0.5, i = 1; i < 54; i++)
    {
      tcmp2 (1.0, 1.0-x, i);
      x /= 2.0;
    }
  for (x = 0.5, i = 1; i < 100; i++)
    {
      tcmp2 (1.0, x, 1);
      x /= 2.0;
    }
  for (j = 0; j < 100000; j++)
    {
      x = DBL_RAND ();
      y = DBL_RAND ();
      if (x < y)
        {
          z = x;
          x = y;
          y = z;
        }
      if (y != 0.0)
        tcmp2 (x, y, -1);
    }

  tests_end_mpfr ();

  return 0;
}
