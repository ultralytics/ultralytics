/* Test file for mpfr_cosh.

Copyright 2001-2002, 2004-2017 Free Software Foundation, Inc.
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

#define TEST_FUNCTION mpfr_cosh
#define TEST_RANDOM_EMIN -36
#define TEST_RANDOM_EMAX 36
#include "tgeneric.c"

static void
special (void)
{
  mpfr_t  x, y;
  int i;

  mpfr_init (x);
  mpfr_init (y);

  mpfr_set_nan (x);
  mpfr_cosh (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: cosh(NaN) != NaN\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_cosh (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0)
    {
      printf ("Error: cosh(+Inf) != +Inf\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  mpfr_cosh (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0)
    {
      printf ("Error: cosh(-Inf) != +Inf\n");
      exit (1);
    }

  /* cosh(+/-0) = 1 */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_cosh (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 1))
    {
      printf ("Error: cosh(+0) != 1\n");
      exit (1);
    }
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_cosh (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 1))
    {
      printf ("Error: cosh(-0) != 1\n");
      exit (1);
    }

  mpfr_set_prec (x, 32);
  mpfr_set_prec (y, 32);

  mpfr_set_str_binary (x, "0.1101110111111111001011101000101");
  mpfr_set_str_binary (y, "1.0110011001110000101100011001001");
  mpfr_cosh (x, x, MPFR_RNDN);
  if (mpfr_cmp (x, y))
    {
      printf ("Error: mpfr_cosh for prec=32 (1)\n");
      exit (1);
    }

  mpfr_set_str_binary (x, "-0.1110111000011101010111100000101E-1");
  mpfr_set_str_binary (y, "1.0001110000101111111111100110101");
  mpfr_cosh (x, x, MPFR_RNDN);
  if (mpfr_cmp (x, y))
    {
      printf ("Error: mpfr_cosh for prec=32 (2)\n");
      exit (1);
    }

  mpfr_set_prec (x, 2);
  mpfr_clear_flags ();
  mpfr_set_str_binary (x, "1E1000000000");
  i = mpfr_cosh (x, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_INF (x) && MPFR_SIGN (x) > 0);
  MPFR_ASSERTN (mpfr_overflow_p ());
  MPFR_ASSERTN (i == 1);

  mpfr_clear_flags ();
  mpfr_set_str_binary (x, "-1E1000000000");
  i = mpfr_cosh (x, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_INF (x) && MPFR_SIGN (x) > 0);
  MPFR_ASSERTN (mpfr_overflow_p () && !mpfr_underflow_p ());
  MPFR_ASSERTN (i == 1);

  mpfr_clear_flags ();
  mpfr_set_str_binary (x, "-1E1000000000");
  i = mpfr_cosh (x, x, MPFR_RNDD);
  MPFR_ASSERTN (!MPFR_IS_INF (x) && MPFR_SIGN (x) > 0);
  MPFR_ASSERTN (mpfr_overflow_p () && !mpfr_underflow_p ());
  MPFR_ASSERTN (i == -1);

  mpfr_clear_flags ();
  mpfr_set_str_binary (x, "-1E1000000000");
  i = mpfr_cosh (x, x, MPFR_RNDU);
  MPFR_ASSERTN (MPFR_IS_INF (x) && MPFR_SIGN (x) > 0);
  MPFR_ASSERTN (mpfr_overflow_p () && !mpfr_underflow_p ());
  MPFR_ASSERTN (i == 1);

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
special_overflow (void)
{
  /* Check for overflow in 3 cases:
     1. cosh(x) is representable, but not exp(x)
     2. cosh(x) is not representable in the selected range of exp.
     3. cosh(x) exp overflow even with the largest range of exp */
  mpfr_t x, y;
  mpfr_exp_t emin, emax;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  set_emin (-125);
  set_emax (128);

  mpfr_init2 (x, 24);
  mpfr_init2 (y, 24);

  mpfr_set_str_binary (x, "0.101100100000000000110100E7");
  mpfr_cosh (y, x, MPFR_RNDN);
  if (mpfr_cmp_str (y, "0.101010001111001010001110E128", 2, MPFR_RNDN))
    {
      printf("Special overflow error 1.\n");
      mpfr_dump (y);
      exit (1);
    }

  mpfr_set_str_binary (x, "0.101100100000000000110100E8");
  mpfr_cosh (y, x, MPFR_RNDN);
  if (!mpfr_inf_p(y))
    {
      printf("Special overflow error 2.\n");
      mpfr_dump (y);
      exit (1);
    }

  set_emin (emin);
  set_emax (emax);

  mpfr_set_str_binary (x, "0.101100100000000000110100E1000000");
  mpfr_cosh (y, x, MPFR_RNDN);
  if (!mpfr_inf_p(y))
    {
      printf("Special overflow error 3.\n");
      mpfr_dump (y);
      exit (1);
    }

  mpfr_clear (y);
  mpfr_clear (x);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  special_overflow ();
  special ();

  test_generic (2, 100, 100);

  data_check ("data/cosh", mpfr_cosh, "mpfr_cosh");
  bad_cases (mpfr_cosh, mpfr_acosh, "mpfr_cosh", 0, 1, 255, 4, 128, 800, 100);

  tests_end_mpfr ();
  return 0;
}
