/* ty0 -- test file for the Bessel function of second kind (order 0)

Copyright 2007-2017 Free Software Foundation, Inc.
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

#define TEST_FUNCTION mpfr_y0
#define RAND_FUNCTION(x) mpfr_random2(x, MPFR_LIMB_SIZE (x), 8, RANDS)
#include "tgeneric.c"

int
main (int argc, char *argv[])
{
  mpfr_t x, y;

  tests_start_mpfr ();

  mpfr_init (x);
  mpfr_init (y);

  /* special values */
  mpfr_set_nan (x);
  mpfr_y0 (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (y));

  mpfr_set_inf (x, 1); /* +Inf */
  mpfr_y0 (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (y, 0) == 0 && MPFR_IS_POS (y));

  mpfr_set_inf (x, -1); /* -Inf */
  mpfr_y0 (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (y));

  mpfr_set_ui (x, 0, MPFR_RNDN); /* +0 */
  mpfr_y0 (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (y) && MPFR_IS_NEG (y)); /* y0(+0)=-Inf */

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN); /* -0 */
  mpfr_y0 (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (y) && MPFR_IS_NEG (y)); /* y0(-0)=-Inf */

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_y0 (y, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.00010110100110000000001000100110111100110101100011011111");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_y0 for x=1, rnd=MPFR_RNDN\n");
      printf ("Expected "); mpfr_dump (x);
      printf ("Got      "); mpfr_dump (y);
      exit (1);
    }

  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_y0 (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error in mpfr_y0 for x=-1, rnd=MPFR_RNDN\n");
      printf ("Expected NaN\n");
      printf ("Got      "); mpfr_dump (y);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);

  test_generic (2, 100, 1);

  data_check ("data/y0", mpfr_y0, "mpfr_y0");

  tests_end_mpfr ();

  return 0;
}
