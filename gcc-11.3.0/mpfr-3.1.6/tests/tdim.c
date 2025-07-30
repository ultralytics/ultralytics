/* Test file for mpfr_dim.

Copyright 2004-2017 Free Software Foundation, Inc.
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

#define TEST_FUNCTION mpfr_dim
#define TWO_ARGS
#define TEST_RANDOM_EMIN -20
#define TEST_RANDOM_EMAX 20
#define TGENERIC_NOWARNING 1
#include "tgeneric.c"

int
main (void)
{
  mpfr_t x, y, z;

  tests_start_mpfr ();

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (z);

  /* case x=NaN */
  mpfr_set_nan (x);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_dim (z, x, y, MPFR_RNDN);
  if (!mpfr_nan_p (z))
    {
      printf ("Error in mpfr_dim (NaN, 0)\n");
      exit (1);
    }

  /* case x=+Inf */
  mpfr_set_inf (x, 1);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_dim (z, x, y, MPFR_RNDN);
  if (!mpfr_inf_p (z) || mpfr_sgn (z) < 0)
    {
      printf ("Error in mpfr_dim (+Inf, 0)\n");
      exit (1);
    }

  /* case x=-Inf */
  mpfr_set_inf (x, -1);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_dim (z, x, y, MPFR_RNDN);
  if (mpfr_cmp_ui (z, 0) || mpfr_sgn (z) < 0)
    {
      printf ("Error in mpfr_dim (-Inf, 0)\n");
      exit (1);
    }

  /* case x=y=+Inf */
  mpfr_set_inf (x, 1);
  mpfr_set_inf (y, 1);
  mpfr_dim (z, x, y, MPFR_RNDN);
  if (mpfr_cmp_ui (z, 0) || mpfr_sgn (z) < 0)
    {
      printf ("Error in mpfr_dim (+Inf, +Inf)\n");
      exit (1);
    }

  /* case x > y */
  mpfr_set_ui (x, 2, MPFR_RNDN);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_dim (z, x, y, MPFR_RNDN);
  if (mpfr_cmp_ui (z, 1))
    {
      printf ("Error in mpfr_dim (2, 1)\n");
      exit (1);
    }

  /* case x < y */
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_set_ui (y, 2, MPFR_RNDN);
  mpfr_dim (z, x, y, MPFR_RNDN);
  if (mpfr_cmp_ui (z, 0))
    {
      printf ("Error in mpfr_dim (1, 2)\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);

  test_generic (2, 220, 42);

  tests_end_mpfr ();
  return 0;
}
