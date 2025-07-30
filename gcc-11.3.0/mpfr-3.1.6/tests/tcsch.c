/* Test file for mpfr_csch.

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

#define TEST_FUNCTION mpfr_csch
#define TEST_RANDOM_EMAX 63
#include "tgeneric.c"

static void
check_specials (void)
{
  mpfr_t  x, y;

  mpfr_init2 (x, 123L);
  mpfr_init2 (y, 123L);

  mpfr_set_nan (x);
  mpfr_csch (y, x, MPFR_RNDN);
  if (! mpfr_nan_p (y))
    {
      printf ("Error: csch(NaN) != NaN\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_csch (y, x, MPFR_RNDN);
  if (! (mpfr_zero_p (y) && MPFR_SIGN (y) >0))
    {
      printf ("Error: csch(+Inf) != +0\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  mpfr_csch (y, x, MPFR_RNDN);
  if (! (mpfr_zero_p (y) && MPFR_SIGN (y) <0))
    {
      printf ("Error: csch(-0) != -0\n");
      exit (1);
    }

  /* csc(+/-0) = +/-Inf */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_csch (y, x, MPFR_RNDN);
  if (! (mpfr_inf_p (y) && mpfr_sgn (y) > 0))
    {
      printf ("Error: csch(+0) != +Inf\n");
      exit (1);
    }
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_csch (y, x, MPFR_RNDN);
  if (! (mpfr_inf_p (y) && mpfr_sgn (y) < 0))
    {
      printf ("Error: csch(-0) != -Inf\n");
      exit (1);
    }

  /* check huge x */
  mpfr_set_str (x, "8e8", 10, MPFR_RNDN);
  mpfr_csch (y, x, MPFR_RNDN);
  if (! (mpfr_zero_p (y) && MPFR_SIGN (y) > 0))
    {
      printf ("Error: csch(8e8) != +0\n");
      exit (1);
    }
  mpfr_set_str (x, "-8e8", 10, MPFR_RNDN);
  mpfr_csch (y, x, MPFR_RNDN);
  if (! (mpfr_zero_p (y) && MPFR_SIGN (y) < 0))
    {
      printf ("Error: csch(-8e8) != -0\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  check_specials ();
  test_generic (2, 200, 10);

  tests_end_mpfr ();
  return 0;
}
