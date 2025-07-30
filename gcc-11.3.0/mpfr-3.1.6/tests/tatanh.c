/* Test file for mpfr_atanh.

Copyright 2001-2017 Free Software Foundation, Inc.
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

#define TEST_FUNCTION mpfr_atanh
#define TEST_RANDOM_EMAX 7
#include "tgeneric.c"

static void
special (void)
{
  mpfr_t x, y, z;
  int i;

  mpfr_init (x);
  mpfr_init (y);

  MPFR_SET_INF(x);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_atanh (x, y, MPFR_RNDN);
  if (MPFR_IS_INF(x) || MPFR_IS_NAN(x) )
    {
      printf ("Inf flag not clears in atanh!\n");
      exit (1);
    }

  MPFR_SET_NAN(x);
  mpfr_atanh (x, y, MPFR_RNDN);
  if (MPFR_IS_NAN(x) || MPFR_IS_INF(x))
    {
      printf ("NAN flag not clears in atanh!\n");
      exit (1);
    }

  /* atanh(+/-x) = NaN if x > 1 */
  for (i = 3; i <= 6; i++)
    {
      mpfr_set_si (x, i, MPFR_RNDN);
      mpfr_div_2ui (x, x, 1, MPFR_RNDN);
      mpfr_atanh (y, x, MPFR_RNDN);
      if (!mpfr_nan_p (y))
        {
          printf ("Error: mpfr_atanh(%d/2) <> NaN\n", i);
          exit (1);
        }
      mpfr_neg (x, x, MPFR_RNDN);
      mpfr_atanh (y, x, MPFR_RNDN);
      if (!mpfr_nan_p (y))
        {
          printf ("Error: mpfr_atanh(-%d/2) <> NaN\n", i);
          exit (1);
        }
    }

  /* atanh(+0) = +0, atanh(-0) = -0 */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_atanh (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 0) || mpfr_sgn (y) < 0)
    {
      printf ("Error: mpfr_atanh(+0) <> +0\n");
      exit (1);
    }
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_atanh (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 0) || mpfr_sgn (y) > 0)
    {
      printf ("Error: mpfr_atanh(-0) <> -0\n");
      exit (1);
    }

  MPFR_SET_NAN(x);
  mpfr_atanh (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_atanh(NaN) <> NaN\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_atanh (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_atanh(+Inf) <> NaN\n");
      mpfr_print_binary (y); printf ("\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  mpfr_atanh (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_atanh(-Inf) <> NaN\n");
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_atanh (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0)
    {
      printf ("Error: mpfr_atanh(1) <> +Inf\n");
      exit (1);
    }

  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_atanh (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) > 0)
    {
      printf ("Error: mpfr_atanh(-1) <> -Inf\n");
      exit (1);
    }

  mpfr_set_prec (x, 32);
  mpfr_set_prec (y, 32);

  mpfr_set_str_binary (x, "0.10001000001001011000100001E-6");
  mpfr_atanh (x, x, MPFR_RNDN);
  mpfr_set_str_binary (y, "0.10001000001001100101010110100001E-6");
  if (mpfr_cmp (x, y))
    {
      printf ("Error: mpfr_atanh (1)\n");
      exit (1);
    }

  mpfr_set_str_binary (x, "-0.1101011110111100111010011001011E-1");
  mpfr_atanh (x, x, MPFR_RNDN);
  mpfr_set_str_binary (y, "-0.11100110000100001111101100010111E-1");
  if (mpfr_cmp (x, y))
    {
      printf ("Error: mpfr_atanh (2)\n");
      exit (1);
    }

  mpfr_set_prec (x, 33);
  mpfr_set_prec (y, 43);
  mpfr_set_str_binary (x, "0.111001101100000110011001010000101");
  mpfr_atanh (y, x, MPFR_RNDZ);
  mpfr_init2 (z, 43);
  mpfr_set_str_binary (z, "1.01111010110001101001000000101101011110101");
  if (mpfr_cmp (y, z))
    {
      printf ("Error: mpfr_atanh (3)\n");
      mpfr_print_binary (y); printf ("\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  special ();

  test_generic (2, 100, 25);

  data_check ("data/atanh", mpfr_atanh, "mpfr_atanh");
  bad_cases (mpfr_atanh, mpfr_tanh, "mpfr_atanh", 256, -128, 9,
             4, 128, 800, 100);

  tests_end_mpfr ();
  return 0;
}
