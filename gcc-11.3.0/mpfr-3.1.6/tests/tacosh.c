/* Test file for mpfr_acosh.

Copyright 2001-2004, 2006-2017 Free Software Foundation, Inc.
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

#define TEST_FUNCTION mpfr_acosh
#define TEST_RANDOM_POS 4
#define TEST_RANDOM_EMIN 1
#include "tgeneric.c"

#define TEST_FUNCTION mpfr_acosh
#define TEST_RANDOM_POS 1
#define TEST_RANDOM_EMIN MPFR_EMAX_MAX
#define TEST_RANDOM_EMAX MPFR_EMAX_MAX
#define test_generic test_generic_huge
#include "tgeneric.c"

static void
special (void)
{
  mpfr_t x, y;

  mpfr_init (x);
  mpfr_init (y);

  MPFR_SET_INF(x);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_acosh (x, y, MPFR_RNDN);
  if (MPFR_IS_INF(x) || MPFR_IS_NAN(x) )
    {
      printf ("Inf flag not clears in acosh!\n");
      exit (1);
    }
  if (mpfr_cmp_ui (x, 0))
    {
      printf ("Error: mpfr_acosh(1) <> 0\n");
      exit (1);
    }

  MPFR_SET_NAN(x);
  mpfr_acosh (x, y, MPFR_RNDN);
  if (MPFR_IS_NAN(x) || MPFR_IS_INF(x) )
    {
      printf ("NAN flag not clears in acosh!\n");
      exit (1);
    }

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_acosh (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_acosh(0) <> NaN\n");
      exit (1);
    }

  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_acosh (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_acosh(-1) <> NaN\n");
      exit (1);
    }

  MPFR_SET_NAN(x);
  mpfr_acosh (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_acosh(NaN) <> NaN\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_acosh (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0)
    {
      printf ("Error: mpfr_acosh(+Inf) <> +Inf\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  mpfr_acosh (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_acosh(-Inf) <> NaN\n");
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_div_2exp (x, x, 1, MPFR_RNDN);
  mpfr_acosh (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_acosh(1/2) <> NaN\n");
      exit (1);
    }

  mpfr_set_prec (x, 32);
  mpfr_set_prec (y, 32);
  mpfr_set_str_binary (x, "1.000001101011101111001011");
  mpfr_acosh (y, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.111010100101101001010001101001E-2");
  if (mpfr_cmp (x, y))
    {
      printf ("Error: mpfr_acosh (1)\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

/* With MPFR 2.3.0, this yields an assertion failure in mpfr_acosh. */
static void
bug20070831 (void)
{
  mpfr_t x, y, z;
  int inex;

  mpfr_init2 (x, 256);
  mpfr_init2 (y, 32);
  mpfr_init2 (z, 32);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_nextabove (x);
  inex = mpfr_acosh (y, x, MPFR_RNDZ);
  mpfr_set_ui_2exp (z, 1, -127, MPFR_RNDN);
  mpfr_nextbelow (z);
  if (!mpfr_equal_p (y, z))
    {
      printf ("Error in bug20070831 (1):\nexpected ");
      mpfr_dump (z);
      printf ("got      ");
      mpfr_dump (y);
      exit (1);
    }
  MPFR_ASSERTN (inex < 0);

  mpfr_nextabove (x);
  mpfr_set_prec (y, 29);
  inex = mpfr_acosh (y, x, MPFR_RNDN);
  mpfr_set_str_binary (z, "1.011010100000100111100110011E-127");
  if (!mpfr_equal_p (y, z))
    {
      printf ("Error in bug20070831 (2):\nexpected ");
      mpfr_dump (z);
      printf ("got      ");
      mpfr_dump (y);
      exit (1);
    }
  MPFR_ASSERTN (inex < 0);

  mpfr_clears (x, y, z, (mpfr_ptr) 0);
}

static void
huge (void)
{
  mpfr_t x, y, z;
  int inex;

  /* TODO: extend the exponent range and use mpfr_get_emax (). */
  mpfr_inits2 (32, x, y, z, (mpfr_ptr) 0);
  mpfr_set_ui_2exp (x, 1, 1073741822, MPFR_RNDN);
  inex = mpfr_acosh (y, x, MPFR_RNDN);
  mpfr_set_str_binary (z, "0.10110001011100100001011111110101E30");
  if (!mpfr_equal_p (y, z))
    {
      printf ("Error in huge:\nexpected ");
      mpfr_dump (z);
      printf ("got      ");
      mpfr_dump (y);
      exit (1);
    }
  MPFR_ASSERTN (inex < 0);

  mpfr_clears (x, y, z, (mpfr_ptr) 0);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  special ();
  bug20070831 ();
  huge ();

  test_generic (2, 100, 25);
  test_generic_huge (2, 100, 5);

  data_check ("data/acosh", mpfr_acosh, "mpfr_acosh");
  bad_cases (mpfr_acosh, mpfr_cosh, "mpfr_acosh", 0, -128, 29,
             4, 128, 800, 40);

  tests_end_mpfr ();
  return 0;
}
