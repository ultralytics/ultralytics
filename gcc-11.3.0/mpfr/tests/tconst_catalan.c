/* Test file for mpfr_const_catalan.

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

/* Wrapper for tgeneric */
static int
my_const_catalan (mpfr_ptr x, mpfr_srcptr y, mpfr_rnd_t r)
{
  return mpfr_const_catalan (x, r);
}

#define RAND_FUNCTION(x) mpfr_set_ui(x,0,MPFR_RNDN)
#define TEST_FUNCTION my_const_catalan
#include "tgeneric.c"

int
main (int argc, char *argv[])
{
  mpfr_t x;

  tests_start_mpfr ();

  mpfr_init2 (x, 32);
  (mpfr_const_catalan) (x, MPFR_RNDN);
  mpfr_mul_2exp (x, x, 32, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 3934042271UL))
    {
      printf ("Error in const_catalan for prec=32\n");
      exit (1);
    }
  mpfr_clear (x);

  test_generic (2, 200, 1);

  tests_end_mpfr ();
  return 0;
}
