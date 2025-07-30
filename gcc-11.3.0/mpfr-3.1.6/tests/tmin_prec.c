/* Test file for mpfr_min_prec.

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

int
main (int argc, char *argv[])
{
  mpfr_t x;
  mpfr_prec_t ret;
  unsigned long i;

  tests_start_mpfr ();

  mpfr_init2 (x, 53);

  /* Check special values */
  mpfr_set_nan (x);
  ret = mpfr_min_prec (x);
  MPFR_ASSERTN (ret == 0);

  mpfr_set_inf (x, 1);
  ret = mpfr_min_prec (x);
  MPFR_ASSERTN (ret == 0);

  mpfr_set_inf (x, -1);
  ret = mpfr_min_prec (x);
  MPFR_ASSERTN (ret == 0);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  ret = mpfr_min_prec (x);
  MPFR_ASSERTN (ret == 0);

  /* Some constants */

  mpfr_set_ui (x, 1, MPFR_RNDN);
  ret = mpfr_min_prec (x);
  MPFR_ASSERTN (ret == 1);

  mpfr_set_ui (x, 17, MPFR_RNDN);
  ret = mpfr_min_prec (x);
  MPFR_ASSERTN (ret == 5);

  mpfr_set_ui (x, 42, MPFR_RNDN);
  ret = mpfr_min_prec (x);
  MPFR_ASSERTN (ret == 5);

  mpfr_set_prec (x, 256);
  for (i = 0; i <= 255; i++)
    {
      mpfr_set_ui_2exp (x, 1, i, MPFR_RNDN);
      mpfr_add_ui (x, x, 1, MPFR_RNDN);
      ret = mpfr_min_prec (x);
      if (ret != i + 1)
        {
          printf ("Error for x = 2^%lu + 1\n", i);
          printf ("Expected %lu, got %lu\n", i + 1, (unsigned long) ret);
          exit (1);
        }
    }

  for (i = MPFR_PREC_MIN; i <= 255; i++)
    {
      mpfr_set_prec (x, i);
      mpfr_set_ui_2exp (x, 1, i, MPFR_RNDN);
      mpfr_sub_ui (x, x, 1, MPFR_RNDN);
      ret = mpfr_min_prec (x);
      if (ret != i)
        {
          printf ("Error for x = 2^%lu - 1\n", i);
          printf ("Expected %lu, got %lu\n", i, (unsigned long) ret);
          exit (1);
        }
    }

  mpfr_clear (x);

  tests_end_mpfr ();
  return 0;
}
