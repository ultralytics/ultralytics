/* mpfr_tlngamma -- test file for lngamma function

Copyright 2005-2017 Free Software Foundation, Inc.
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

#define TEST_FUNCTION mpfr_lngamma
#define TEST_RANDOM_POS 16
#include "tgeneric.c"

static void
special (void)
{
  mpfr_t x, y;
  int i, inex;

  mpfr_init (x);
  mpfr_init (y);

  mpfr_set_nan (x);
  mpfr_lngamma (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error for lngamma(NaN)\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_clear_flags ();
  mpfr_lngamma (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0 || __gmpfr_flags != 0)
    {
      printf ("Error for lngamma(+Inf)\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  mpfr_clear_flags ();
  mpfr_lngamma (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0 || __gmpfr_flags != 0)
    {
      printf ("Error for lngamma(-Inf)\n");
      exit (1);
    }

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_clear_flags ();
  mpfr_lngamma (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0 ||
      __gmpfr_flags != MPFR_FLAGS_DIVBY0)
    {
      printf ("Error for lngamma(+0)\n");
      exit (1);
    }

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_clear_flags ();
  mpfr_lngamma (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0 ||
      __gmpfr_flags != MPFR_FLAGS_DIVBY0)
    {
      printf ("Error for lngamma(-0)\n");
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_clear_flags ();
  mpfr_lngamma (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui0 (y, 0) || MPFR_IS_NEG (y))
    {
      printf ("Error for lngamma(1)\n");
      exit (1);
    }

  for (i = 1; i <= 5; i++)
    {
      int c;

      mpfr_set_si (x, -i, MPFR_RNDN);
      mpfr_clear_flags ();
      mpfr_lngamma (y, x, MPFR_RNDN);
      if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0 ||
          __gmpfr_flags != MPFR_FLAGS_DIVBY0)
        {
          printf ("Error for lngamma(-%d)\n", i);
          exit (1);
        }
      if (i & 1)
        {
          mpfr_nextabove (x);
          c = '+';
        }
      else
        {
          mpfr_nextbelow (x);
          c = '-';
        }
      mpfr_lngamma (y, x, MPFR_RNDN);
      if (!mpfr_nan_p (y))
        {
          printf ("Error for lngamma(-%d%cepsilon)\n", i, c);
          exit (1);
        }
    }

  mpfr_set_ui (x, 2, MPFR_RNDN);
  mpfr_lngamma (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui0 (y, 0) || MPFR_IS_NEG (y))
    {
      printf ("Error for lngamma(2)\n");
      exit (1);
    }

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);

#define CHECK_X1 "1.0762904832837976166"
#define CHECK_Y1 "-0.039418362817587634939"

  mpfr_set_str (x, CHECK_X1, 10, MPFR_RNDN);
  mpfr_lngamma (y, x, MPFR_RNDN);
  mpfr_set_str (x, CHECK_Y1, 10, MPFR_RNDN);
  if (MPFR_IS_NAN (y) || mpfr_cmp (y, x))
    {
      printf ("mpfr_lngamma("CHECK_X1") is wrong:\n"
              "expected ");
      mpfr_print_binary (x); putchar ('\n');
      printf ("got      ");
      mpfr_print_binary (y); putchar ('\n');
      exit (1);
    }

#define CHECK_X2 "9.23709516716202383435e-01"
#define CHECK_Y2 "0.049010669407893718563"
  mpfr_set_str (x, CHECK_X2, 10, MPFR_RNDN);
  mpfr_lngamma (y, x, MPFR_RNDN);
  mpfr_set_str (x, CHECK_Y2, 10, MPFR_RNDN);
  if (mpfr_cmp0 (y, x))
    {
      printf ("mpfr_lngamma("CHECK_X2") is wrong:\n"
              "expected ");
      mpfr_print_binary (x); putchar ('\n');
      printf ("got      ");
      mpfr_print_binary (y); putchar ('\n');
      exit (1);
    }

  mpfr_set_prec (x, 8);
  mpfr_set_prec (y, 175);
  mpfr_set_ui (x, 33, MPFR_RNDN);
  mpfr_lngamma (y, x, MPFR_RNDU);
  mpfr_set_prec (x, 175);
  mpfr_set_str_binary (x, "0.1010001100011101101011001101110010100001000001000001110011000001101100001111001001000101011011100100010101011110100111110101010100010011010010000101010111001100011000101111E7");
  if (mpfr_cmp0 (x, y))
    {
      printf ("Error in mpfr_lngamma (1)\n");
      exit (1);
    }

  mpfr_set_prec (x, 21);
  mpfr_set_prec (y, 8);
  mpfr_set_ui (y, 120, MPFR_RNDN);
  mpfr_lngamma (x, y, MPFR_RNDZ);
  mpfr_set_prec (y, 21);
  mpfr_set_str_binary (y, "0.111000101000001100101E9");
  if (mpfr_cmp0 (x, y))
    {
      printf ("Error in mpfr_lngamma (120)\n");
      printf ("Expected "); mpfr_print_binary (y); puts ("");
      printf ("Got      "); mpfr_print_binary (x); puts ("");
      exit (1);
    }

  mpfr_set_prec (x, 3);
  mpfr_set_prec (y, 206);
  mpfr_set_str_binary (x, "0.110e10");
  inex = mpfr_lngamma (y, x, MPFR_RNDN);
  mpfr_set_prec (x, 206);
  mpfr_set_str_binary (x, "0.10000111011000000011100010101001100110001110000111100011000100100110110010001011011110101001111011110110000001010100111011010000000011100110110101100111000111010011110010000100010111101010001101000110101001E13");
  if (mpfr_cmp0 (x, y))
    {
      printf ("Error in mpfr_lngamma (768)\n");
      exit (1);
    }
  if (inex >= 0)
    {
      printf ("Wrong flag for mpfr_lngamma (768)\n");
      exit (1);
    }

  mpfr_set_prec (x, 4);
  mpfr_set_prec (y, 4);
  mpfr_set_str_binary (x, "0.1100E-66");
  mpfr_lngamma (y, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.1100E6");
  if (mpfr_cmp0 (x, y))
    {
      printf ("Error for lngamma(0.1100E-66)\n");
      exit (1);
    }

  mpfr_set_prec (x, 256);
  mpfr_set_prec (y, 32);
  mpfr_set_si_2exp (x, -1, 200, MPFR_RNDN);
  mpfr_add_ui (x, x, 1, MPFR_RNDN);
  mpfr_div_2ui (x, x, 1, MPFR_RNDN);
  mpfr_lngamma (y, x, MPFR_RNDN);
  mpfr_set_prec (x, 32);
  mpfr_set_str_binary (x, "-0.10001000111011111011000010100010E207");
  if (mpfr_cmp0 (x, y))
    {
      printf ("Error for lngamma(-2^199+0.5)\n");
      printf ("Got        ");
      mpfr_dump (y);
      printf ("instead of ");
      mpfr_dump (x);
      exit (1);
    }

  mpfr_set_prec (x, 256);
  mpfr_set_prec (y, 32);
  mpfr_set_si_2exp (x, -1, 200, MPFR_RNDN);
  mpfr_sub_ui (x, x, 1, MPFR_RNDN);
  mpfr_div_2ui (x, x, 1, MPFR_RNDN);
  mpfr_lngamma (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error for lngamma(-2^199-0.5)\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

int
main (void)
{
  tests_start_mpfr ();

  special ();
  test_generic (2, 100, 2);

  tests_end_mpfr ();
  return 0;
}
