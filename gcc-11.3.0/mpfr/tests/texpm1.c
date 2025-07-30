/* Test file for mpfr_expm1.

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

#ifdef CHECK_EXTERNAL
static int
test_expm1 (mpfr_ptr a, mpfr_srcptr b, mpfr_rnd_t rnd_mode)
{
  int res;
  int ok = rnd_mode == MPFR_RNDN && mpfr_number_p (b) && mpfr_get_prec (a)>=53;
  if (ok)
    {
      mpfr_print_raw (b);
    }
  res = mpfr_expm1 (a, b, rnd_mode);
  if (ok)
    {
      printf (" ");
      mpfr_print_raw (a);
      printf ("\n");
    }
  return res;
}
#else
#define test_expm1 mpfr_expm1
#endif

#define TEST_FUNCTION test_expm1
#define TEST_RANDOM_EMIN -36
#define TEST_RANDOM_EMAX 36
#include "tgeneric.c"

static void
special (void)
{
  mpfr_t x, y;
  int i;

  mpfr_init (x);
  mpfr_init (y);

  mpfr_set_nan (x);
  test_expm1 (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error for expm1(NaN)\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  test_expm1 (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0)
    {
      printf ("Error for expm1(+Inf)\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  test_expm1 (y, x, MPFR_RNDN);
  if (mpfr_cmp_si (y, -1))
    {
      printf ("Error for expm1(-Inf)\n");
      exit (1);
    }

  mpfr_set_ui (x, 0, MPFR_RNDN);
  test_expm1 (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 0) || mpfr_sgn (y) < 0)
    {
      printf ("Error for expm1(+0)\n");
      exit (1);
    }

  mpfr_neg (x, x, MPFR_RNDN);
  test_expm1 (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 0) || mpfr_sgn (y) > 0)
    {
      printf ("Error for expm1(-0)\n");
      exit (1);
    }

  /* Check overflow of expm1(x) */
  mpfr_clear_flags ();
  mpfr_set_str_binary (x, "1.1E1000000000");
  i = test_expm1 (x, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_INF (x) && MPFR_SIGN (x) > 0);
  MPFR_ASSERTN (mpfr_overflow_p ());
  MPFR_ASSERTN (i == 1);

  mpfr_clear_flags ();
  mpfr_set_str_binary (x, "1.1E1000000000");
  i = test_expm1 (x, x, MPFR_RNDU);
  MPFR_ASSERTN (MPFR_IS_INF (x) && MPFR_SIGN (x) > 0);
  MPFR_ASSERTN (mpfr_overflow_p ());
  MPFR_ASSERTN (i == 1);

  mpfr_clear_flags ();
  mpfr_set_str_binary (x, "1.1E1000000000");
  i = test_expm1 (x, x, MPFR_RNDD);
  MPFR_ASSERTN (!MPFR_IS_INF (x) && MPFR_SIGN (x) > 0);
  MPFR_ASSERTN (mpfr_overflow_p ());
  MPFR_ASSERTN (i == -1);

  /* Check internal underflow of expm1 (x) */
  mpfr_set_prec (x, 2);
  mpfr_clear_flags ();
  mpfr_set_str_binary (x, "-1.1E1000000000");
  i = test_expm1 (x, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_si (x, -1) == 0);
  MPFR_ASSERTN (!mpfr_overflow_p () && !mpfr_underflow_p ());
  MPFR_ASSERTN (i == -1);

  mpfr_set_str_binary (x, "-1.1E1000000000");
  i = test_expm1 (x, x, MPFR_RNDD);
  MPFR_ASSERTN (mpfr_cmp_si (x, -1) == 0);
  MPFR_ASSERTN (!mpfr_overflow_p () && !mpfr_underflow_p ());
  MPFR_ASSERTN (i == -1);

  mpfr_set_str_binary (x, "-1.1E1000000000");
  i = test_expm1 (x, x, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_cmp_str (x, "-0.11", 2, MPFR_RNDN) == 0);
  MPFR_ASSERTN (!mpfr_overflow_p () && !mpfr_underflow_p ());
  MPFR_ASSERTN (i == 1);

  mpfr_set_str_binary (x, "-1.1E1000000000");
  i = test_expm1 (x, x, MPFR_RNDU);
  MPFR_ASSERTN (mpfr_cmp_str (x, "-0.11", 2, MPFR_RNDN) == 0);
  MPFR_ASSERTN (!mpfr_overflow_p () && !mpfr_underflow_p ());
  MPFR_ASSERTN (i == 1);

  mpfr_clear (x);
  mpfr_clear (y);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  special ();

  test_generic (2, 100, 100);

  data_check ("data/expm1", mpfr_expm1, "mpfr_expm1");
  bad_cases (mpfr_expm1, mpfr_log1p, "mpfr_expm1", 256, -256, 255,
             4, 128, 800, 40);

  tests_end_mpfr ();
  return 0;
}
