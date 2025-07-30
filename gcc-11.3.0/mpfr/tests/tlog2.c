/* Test file for mpfr_log2.

Copyright 2001-2002, 2004, 2006-2017 Free Software Foundation, Inc.
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

#define TEST_FUNCTION mpfr_log2
#define TEST_RANDOM_POS 8
#include "tgeneric.c"

static void
special (void)
{
  mpfr_t x;
  int inex;

  mpfr_init (x);

  mpfr_set_nan (x);
  inex = mpfr_log2 (x, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (x) && inex == 0);

  mpfr_set_inf (x, -1);
  inex = mpfr_log2 (x, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (x) && inex == 0);

  mpfr_set_inf (x, 1);
  inex = mpfr_log2 (x, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (x) && mpfr_sgn (x) > 0 && inex == 0);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  inex = mpfr_log2 (x, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (x) && mpfr_sgn (x) < 0 && inex == 0);
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  inex = mpfr_log2 (x, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (x) && mpfr_sgn (x) < 0 && inex == 0);

  mpfr_set_si (x, -1, MPFR_RNDN);
  inex = mpfr_log2 (x, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (x) && inex == 0);

  mpfr_set_si (x, 1, MPFR_RNDN);
  inex = mpfr_log2 (x, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_ui (x, 0) == 0 && MPFR_IS_POS(x) && inex == 0);

  mpfr_clear (x);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  special ();

  test_generic (2, 100, 30);

  data_check ("data/log2", mpfr_log2, "mpfr_log2");

  tests_end_mpfr ();
  return 0;
}
