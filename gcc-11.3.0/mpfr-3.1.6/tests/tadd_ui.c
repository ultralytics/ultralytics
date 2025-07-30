/* Test file for mpfr_add_ui

Copyright 2000-2017 Free Software Foundation, Inc.
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
#include <float.h>

#include "mpfr-test.h"

/* checks that x+y gives the right results with 53 bits of precision */
static void
check3 (const char *xs, unsigned long y, mpfr_rnd_t rnd_mode, const char *zs)
{
  mpfr_t xx, zz;

  mpfr_inits2 (53, xx, zz, (mpfr_ptr) 0);
  mpfr_set_str1 (xx, xs);
  mpfr_add_ui (zz, xx, y, rnd_mode);
  if (mpfr_cmp_str1(zz, zs) )
    {
      printf ("expected sum is %s, got ",zs);
      mpfr_out_str(stdout, 10, 0, zz, MPFR_RNDN);
      printf ("\nmpfr_add_ui failed for x=%s y=%lu with rnd_mode=%s\n",
              xs, y, mpfr_print_rnd_mode(rnd_mode));
      exit (1);
  }
  mpfr_clears (xx, zz, (mpfr_ptr) 0);
}

static void
special (void)
{
  mpfr_t x, y;

  mpfr_init2 (x, 63);
  mpfr_init2 (y, 63);
  mpfr_set_str_binary (x, "0.110100000000000001110001110010111111000000000101100011100100011");
  mpfr_add_ui (y, x, 1, MPFR_RNDD);
  mpfr_clear (x);
  mpfr_clear (y);
}

static void
check_nans (void)
{
  mpfr_t  x, y;

  mpfr_init2 (x, 123L);
  mpfr_init2 (y, 123L);

  /* nan + 2394875 == nan */
  mpfr_set_nan (x);
  mpfr_clear_nanflag ();
  mpfr_add_ui (y, x, 2394875L, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nanflag_p ());
  MPFR_ASSERTN (mpfr_nan_p (y));

  /* +inf + 2394875 == +inf */
  mpfr_set_inf (x, 1);
  mpfr_add_ui (y, x, 2394875L, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (y));
  MPFR_ASSERTN (mpfr_sgn (y) > 0);

  /* -inf + 2394875 == -inf */
  mpfr_set_inf (x, -1);
  mpfr_add_ui (y, x, 2394875L, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (y));
  MPFR_ASSERTN (mpfr_sgn (y) < 0);

  mpfr_clear (x);
  mpfr_clear (y);
}

#define TEST_FUNCTION mpfr_add_ui
#define ULONG_ARG2
#define RAND_FUNCTION(x) mpfr_random2(x, MPFR_LIMB_SIZE (x), 1, RANDS)
#include "tgeneric.c"

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  check_nans ();

  special ();
  check3 ("-1.716113812768534e-140", 1271212614, MPFR_RNDZ,
          "1.27121261399999976e9");
  check3 ("1.22191250737771397120e+20", 948002822, MPFR_RNDN,
          "122191250738719408128.0");
  check3 ("-6.72658901114033715233e-165", 2000878121, MPFR_RNDZ,
          "2.0008781209999997615e9");
  check3 ("-2.0769715792901673e-5", 880524, MPFR_RNDN,
          "8.8052399997923023e5");

  test_generic (2, 1000, 100);

  tests_end_mpfr ();
  return 0;
}
