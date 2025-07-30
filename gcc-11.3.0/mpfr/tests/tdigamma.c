/* test file for digamma function

Copyright 2009-2017 Free Software Foundation, Inc.
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

#define TEST_FUNCTION mpfr_digamma
#include "tgeneric.c"

static void
special (void)
{
  mpfr_t x, y;

  mpfr_init (x);
  mpfr_init (y);

  mpfr_set_inf (y, -1);
  mpfr_set_inf (x, 1);
  mpfr_digamma (y, x, MPFR_RNDN);
  if (mpfr_inf_p (y) == 0 || mpfr_sgn (y) < 0)
    {
      printf ("error for Psi(+Inf)\n");
      printf ("expected +Inf\n");
      printf ("got      ");
      mpfr_dump (y);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  special ();

  test_generic (2, 100, 2);

  data_check ("data/digamma", mpfr_digamma, "mpfr_digamma");

  tests_end_mpfr ();
  return 0;
}
