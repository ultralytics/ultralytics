/* Test file for mpfr_asin.

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

#define TEST_FUNCTION mpfr_asin
#define TEST_RANDOM_EMAX 7
#include "tgeneric.c"

static void
special (void)
{
  mpfr_t x, y;
  int r;

  mpfr_init (x);
  mpfr_init (y);

  /* asin(NaN) = NaN */
  mpfr_set_nan (x);
  mpfr_asin (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_asin (NaN) <> NaN\n");
      exit (1);
    }

  /* asin(+/-Inf) = NaN */
  mpfr_set_inf (x, 1);
  mpfr_asin (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_asin (+Inf) <> NaN\n");
      exit (1);
    }
  mpfr_set_inf (x, -1);
  mpfr_asin (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_asin (-Inf) <> NaN\n");
      exit (1);
    }

  /* asin(+/-2) = NaN */
  mpfr_set_ui (x, 2, MPFR_RNDN);
  mpfr_asin (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_asin (+2) <> NaN\n");
      exit (1);
    }
  mpfr_set_si (x, -2, MPFR_RNDN);
  mpfr_asin (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_asin (-2) <> NaN\n");
      exit (1);
    }

  /* asin(+/-0) = +/-0 */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_asin (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 0) || mpfr_sgn (y) < 0)
    {
      printf ("Error: mpfr_asin (+0) <> +0\n");
      exit (1);
    }
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_asin (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 0) || mpfr_sgn (y) > 0)
    {
      printf ("Error: mpfr_asin (-0) <> -0\n");
      exit (1);
    }

  /* asin(1) = Pi/2 */
  for (r = 0; r < MPFR_RND_MAX; r++)
    {
      mpfr_set_ui (x, 1, MPFR_RNDN); /* exact */
      mpfr_asin (y, x, (mpfr_rnd_t) r);
      mpfr_const_pi (x, (mpfr_rnd_t) r);
      mpfr_div_2exp (x, x, 1, MPFR_RNDN); /* exact */
      if (mpfr_cmp (x, y))
        {
          printf ("Error: asin(1) != Pi/2 for rnd=%s\n",
                  mpfr_print_rnd_mode ((mpfr_rnd_t) r));
          exit (1);
        }
    }

  /* asin(-1) = -Pi/2 */
  for (r = 0; r < MPFR_RND_MAX; r++)
    {
      mpfr_set_si (x, -1, MPFR_RNDN); /* exact */
      mpfr_asin (y, x, (mpfr_rnd_t) r);
      mpfr_const_pi (x, MPFR_INVERT_RND((mpfr_rnd_t) r));
      mpfr_neg (x, x, MPFR_RNDN); /* exact */
      mpfr_div_2exp (x, x, 1, MPFR_RNDN); /* exact */
      if (mpfr_cmp (x, y))
        {
          printf ("Error: asin(-1) != -Pi/2 for rnd=%s\n",
                  mpfr_print_rnd_mode ((mpfr_rnd_t) r));
          exit (1);
        }
    }

  mpfr_set_prec (x, 32);
  mpfr_set_prec (y, 32);

  mpfr_set_str_binary (x, "0.1101110111111111001011101000101");
  mpfr_asin (x, x, MPFR_RNDN);
  mpfr_set_str_binary (y, "1.00001100101011000001111100111");
  if (mpfr_cmp (x, y))
    {
      printf ("Error: mpfr_asin (1)\n");
      exit (1);
    }

  mpfr_set_str_binary (x, "-0.01110111000011101010111100000101");
  mpfr_asin (x, x, MPFR_RNDN);
  mpfr_set_str_binary (y, "-0.0111101111010100011111110101");
  if (mpfr_cmp (x, y))
    {
      printf ("Error: mpfr_asin (2)\n");
      mpfr_print_binary (x); printf ("\n");
      mpfr_print_binary (y); printf ("\n");
      exit (1);
    }

  mpfr_set_prec (x, 9);
  mpfr_set_prec (y, 19);
  mpfr_set_str_binary (x, "0.110000000E-6");
  mpfr_asin (y, x, MPFR_RNDD);
  mpfr_set_prec (x, 19);
  mpfr_set_str_binary (x, "0.1100000000000001001E-6");
  if (mpfr_cmp (x, y))
    {
      printf ("Error: mpfr_asin (3)\n");
      mpfr_dump (x);
      mpfr_dump (y);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
special_overflow (void)
{
  mpfr_t x, y;
  mpfr_exp_t emin, emax;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  set_emin (-125);
  set_emax (128);
  mpfr_init2 (x, 24);
  mpfr_init2 (y, 48);
  mpfr_set_str_binary (x, "0.101100100000000000110100E0");
  mpfr_asin (y, x, MPFR_RNDN);
  if (mpfr_cmp_str (y, "0.110001001101001111110000010110001000111011001000E0",
                    2, MPFR_RNDN))
    {
      printf("Special Overflow error.\n");
      mpfr_dump (y);
      exit (1);
    }
  mpfr_clear (y);
  mpfr_clear (x);
  set_emin (emin);
  set_emax (emax);
}

/* bug reported by Kevin Rauch on 15 December 2007 */
static void
test20071215 (void)
{
  mpfr_t x, y;

  mpfr_init (x);
  mpfr_init (y);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_asin (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_zero_p (y) && MPFR_IS_NEG(y));

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_set_si (y, -1, MPFR_RNDN);
  mpfr_asin (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_zero_p (y) && MPFR_IS_POS(y));

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
reduced_expo_range (void)
{
  mpfr_exp_t emin, emax;
  mpfr_t x, y, ex_y;
  int inex, ex_inex;
  unsigned int flags, ex_flags;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  mpfr_inits2 (4, x, y, ex_y, (mpfr_ptr) 0);
  mpfr_set_str (x, "-0.1e1", 2, MPFR_RNDN);

  mpfr_set_emin (1);
  mpfr_set_emax (1);
  mpfr_clear_flags ();
  inex = mpfr_asin (y, x, MPFR_RNDA);
  flags = __gmpfr_flags;
  mpfr_set_emin (emin);
  mpfr_set_emax (emax);

  mpfr_set_str (ex_y, "-0.1101e1", 2, MPFR_RNDN);
  ex_inex = -1;
  ex_flags = MPFR_FLAGS_INEXACT;

  if (SIGN (inex) != ex_inex || flags != ex_flags ||
      ! mpfr_equal_p (y, ex_y))
    {
      printf ("Error in reduced_expo_range\non x = ");
      mpfr_dump (x);
      printf ("Expected y = ");
      mpfr_out_str (stdout, 2, 0, ex_y, MPFR_RNDN);
      printf ("\n         inex = %d, flags = %u\n", ex_inex, ex_flags);
      printf ("Got      y = ");
      mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
      printf ("\n         inex = %d, flags = %u\n", SIGN (inex), flags);
      exit (1);
    }

  mpfr_clears (x, y, ex_y, (mpfr_ptr) 0);
}

int
main (void)
{
  tests_start_mpfr ();

  special ();
  special_overflow ();
  reduced_expo_range ();

  test_generic (2, 100, 15);

  tests_end_mpfr ();

  data_check ("data/asin", mpfr_asin, "mpfr_asin");
  bad_cases (mpfr_asin, mpfr_sin, "mpfr_asin", 256, -40, 1, 4, 128, 800, 30);

  test20071215 ();

  tests_end_mpfr ();
  return 0;
}
