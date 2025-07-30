/* Test file for mpfr_log10.

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
test_log10 (mpfr_ptr a, mpfr_srcptr b, mpfr_rnd_t rnd_mode)
{
  int res;
  int ok = rnd_mode == MPFR_RNDN && mpfr_number_p (b) && mpfr_get_prec (a)>=53;
  if (ok)
    {
      mpfr_print_raw (b);
    }
  res = mpfr_log10 (a, b, rnd_mode);
  if (ok)
    {
      printf (" ");
      mpfr_print_raw (a);
      printf ("\n");
    }
  return res;
}
#else
#define test_log10 mpfr_log10
#endif

#define TEST_FUNCTION test_log10
#define TEST_RANDOM_POS 8
#include "tgeneric.c"

int
main (int argc, char *argv[])
{
  mpfr_t x, y;
  unsigned int n;
  int inex;

  tests_start_mpfr ();

  test_generic (2, 100, 20);

  mpfr_init2 (x, 53);
  mpfr_init2 (y, 53);

  /* check NaN */
  mpfr_set_nan (x);
  inex = test_log10 (y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (y) && inex == 0);

  /* check Inf */
  mpfr_set_inf (x, -1);
  inex = test_log10 (y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (y) && inex == 0);

  mpfr_set_inf (x, 1);
  inex = test_log10 (y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (y) && mpfr_sgn (y) > 0 && inex == 0);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  inex = test_log10 (x, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (x) && mpfr_sgn (x) < 0 && inex == 0);
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  inex = test_log10 (x, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (x) && mpfr_sgn (x) < 0 && inex == 0);

  /* check negative argument */
  mpfr_set_si (x, -1, MPFR_RNDN);
  inex = test_log10 (y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (y) && inex == 0);

  /* check log10(1) = 0 */
  mpfr_set_ui (x, 1, MPFR_RNDN);
  inex = test_log10 (y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_ui (y, 0) == 0 && MPFR_IS_POS (y) && inex == 0);

  /* check log10(10^n)=n */
  mpfr_set_ui (x, 1, MPFR_RNDN);
  for (n = 1; n <= 15; n++)
    {
      mpfr_mul_ui (x, x, 10, MPFR_RNDN); /* x = 10^n */
      inex = test_log10 (y, x, MPFR_RNDN);
      if (mpfr_cmp_ui (y, n))
        {
          printf ("log10(10^n) <> n for n=%u\n", n);
          exit (1);
        }
      MPFR_ASSERTN (inex == 0);
    }

  mpfr_clear (x);
  mpfr_clear (y);

  data_check ("data/log10", mpfr_log10, "mpfr_log10");

  tests_end_mpfr ();
  return 0;
}
