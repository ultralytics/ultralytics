/* tj0 -- test file for the Bessel function of first kind (order 0)

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

#define TEST_FUNCTION mpfr_j0
#define RAND_FUNCTION(x) mpfr_random2(x, MPFR_LIMB_SIZE (x), 8, RANDS)
#define REDUCE_EMAX 262143 /* otherwise arg. reduction is too expensive */
#include "tgeneric.c"

int
main (int argc, char *argv[])
{
  mpfr_t x, y;
  int inex;

  tests_start_mpfr ();

  mpfr_init (x);
  mpfr_init (y);

  /* special values */
  mpfr_set_nan (x);
  mpfr_j0 (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (y));

  mpfr_set_inf (x, 1); /* +Inf */
  mpfr_j0 (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (y, 0) == 0 && MPFR_IS_POS (y));

  mpfr_set_inf (x, -1); /* -Inf */
  mpfr_j0 (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (y, 0) == 0 && MPFR_IS_POS (y));

  mpfr_set_ui (x, 0, MPFR_RNDN); /* +0 */
  mpfr_j0 (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (y, 1) == 0); /* j0(+0)=1 */

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN); /* -0 */
  mpfr_j0 (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (y, 1) == 0); /* j0(-0)=1 */

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_j0 (y, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.1100001111100011111111101101111010111101110001111");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_j0 for x=1, rnd=MPFR_RNDN\n");
      printf ("Expected "); mpfr_dump (x);
      printf ("Got      "); mpfr_dump (y);
      exit (1);
    }

  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_j0 (y, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.1100001111100011111111101101111010111101110001111");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_j0 for x=-1, rnd=MPFR_RNDN\n");
      printf ("Expected "); mpfr_dump (x);
      printf ("Got      "); mpfr_dump (y);
      exit (1);
    }

  /* Bug reported on 2007-07-03 by Sisyphus (assertion failed in r4619) */
  mpfr_set_si (x, 70000, MPFR_RNDN);
  mpfr_j0 (y, x, MPFR_RNDN);

  /* Bug reported by Kevin Rauch on 27 Oct 2007 */
  mpfr_set_prec (x, 7);
  mpfr_set_prec (y, 7);
  mpfr_set_si (x, -100, MPFR_RNDN);
  mpfr_j0 (y, x, MPFR_RNDN);
  MPFR_ASSERTN (! mpfr_nan_p (y) && mpfr_cmp_ui_2exp (y, 41, -11) == 0);

  /* Bug reported by Fredrik Johansson on 19 Jan 2016 */
  mpfr_set_prec (x, 53);
  mpfr_set_str (x, "0x4.3328p+0", 0, MPFR_RNDN);
  mpfr_set_prec (y, 2);
  mpfr_j0 (y, x, MPFR_RNDD);
  /* y should be -0.5 */
  MPFR_ASSERTN (! mpfr_nan_p (y) && mpfr_cmp_si_2exp (y, -1, -1) == 0);
  mpfr_set_prec (y, 3);
  mpfr_j0 (y, x, MPFR_RNDD);
  /* y should be -0.4375 */
  MPFR_ASSERTN (! mpfr_nan_p (y) && mpfr_cmp_si_2exp (y, -7, -4) == 0);

  /* Case for which s = 0 in mpfr_jn */
  mpfr_set_prec (x, 44);
  mpfr_set_prec (y, 44);
  mpfr_set_si (x, 2, MPFR_RNDN);
  mpfr_clear_flags ();
  inex = mpfr_j0 (y, x, MPFR_RNDN);
  MPFR_ASSERTN (__gmpfr_flags == MPFR_FLAGS_INEXACT);
  mpfr_set_str (x, "0x.e5439fd9267p-2", 0, MPFR_RNDN);
  if (! mpfr_equal_p (y, x))
    {
      printf ("Error on 2:\n");
      printf ("Expected ");
      mpfr_dump (x);
      printf ("Got      ");
      mpfr_dump (y);
      exit (1);
    }
  if (inex >= 0)
    {
      printf ("Bad ternary value on 2: expected negative, got %d\n", inex);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);

  test_generic (2, 100, 10);

  data_check ("data/j0", mpfr_j0, "mpfr_j0");

  tests_end_mpfr ();

  return 0;
}
