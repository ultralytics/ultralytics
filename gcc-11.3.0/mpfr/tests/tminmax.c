/* Test file for mpfr_min & mpfr_max.

Copyright 2004, 2006-2017 Free Software Foundation, Inc.
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
main (void)
{
  mpfr_t x, y, z;

  tests_start_mpfr ();

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (z);

  /* case x=NaN && y=NAN */
  mpfr_set_nan (x);
  mpfr_set_nan (y);
  mpfr_min (z, x, y, MPFR_RNDN);
  if (!mpfr_nan_p (z))
    {
      printf ("Error in mpfr_min (NaN, NaN)\n");
      exit (1);
    }
  mpfr_max (z, x, y, MPFR_RNDN);
  if (!mpfr_nan_p (z))
    {
      printf ("Error in mpfr_max (NaN, NaN)\n");
      exit (1);
    }
  /* case x=NaN */
  mpfr_set_nan (x);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_min (z, x, y, MPFR_RNDN);
  if (mpfr_cmp_ui (z, 0))
    {
      printf ("Error in mpfr_min (NaN, 0)\n");
      exit (1);
    }
  mpfr_min (z, y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (z, 0))
    {
      printf ("Error in mpfr_min (0, NaN)\n");
      exit (1);
    }
  mpfr_max (z, x, y, MPFR_RNDN);
  if (mpfr_cmp_ui (z, 0))
    {
      printf ("Error in mpfr_max (NaN, 0)\n");
      exit (1);
    }
  mpfr_max (z, y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (z, 0))
    {
      printf ("Error in mpfr_max (0, NaN)\n");
      exit (1);
    }
  /* Case x=0+ and x=0- */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_set_ui (y, 0, MPFR_RNDN); MPFR_SET_NEG(y);
  mpfr_max (z, x, y, MPFR_RNDN);
  if (!MPFR_IS_ZERO(z) || MPFR_IS_NEG(z))
    {
      printf ("Error in mpfr_max (0+, 0-)\n");
      exit (1);
    }
  mpfr_min (z, x, y, MPFR_RNDN);
  if (!MPFR_IS_ZERO(z) || MPFR_IS_POS(z))
    {
      printf ("Error in mpfr_min (0+, 0-)\n");
      exit (1);
    }
  /* Case x=0- and y=0+ */
  mpfr_set_ui (x, 0, MPFR_RNDN); MPFR_SET_NEG(x);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_max (z, x, y, MPFR_RNDN);
  if (!MPFR_IS_ZERO(z) || MPFR_IS_NEG(z))
    {
      printf ("Error in mpfr_max (0+, 0-)\n");
      exit (1);
    }
  mpfr_min (z, x, y, MPFR_RNDN);
  if (!MPFR_IS_ZERO(z) || MPFR_IS_POS(z))
    {
      printf ("Error in mpfr_min (0+, 0-)\n");
      exit (1);
    }

  /* case x=+Inf */
  mpfr_set_inf (x, 1);
  mpfr_set_si (y, -12, MPFR_RNDN);
  mpfr_min (z, x, y, MPFR_RNDN);
  if ( mpfr_cmp_si (z, -12) )
    {
      printf ("Error in mpfr_min (+Inf, -12)\n");
      exit (1);
    }
  mpfr_max (z, x, y, MPFR_RNDN);
  if ( !MPFR_IS_INF(z) || MPFR_IS_NEG(z) )
    {
      printf ("Error in mpfr_max (+Inf, 12)\n");
      exit (1);
    }
  /* case x=-Inf */
  mpfr_set_inf (x, -1);
  mpfr_set_ui (y, 12, MPFR_RNDN);
  mpfr_max (z, x, y, MPFR_RNDN);
  if ( mpfr_cmp_ui (z, 12) )
    {
      printf ("Error in mpfr_max (-Inf, 12)\n");
      exit (1);
    }
  mpfr_min (z, x, y, MPFR_RNDN);
  if ( !MPFR_IS_INF(z) || MPFR_IS_POS(z) )
    {
      printf ("Error in mpfr_min (-Inf, 12)\n");
      exit (1);
    }

  /* case x=17 and y=42 */
  mpfr_set_ui (x, 17, MPFR_RNDN);
  mpfr_set_ui (y, 42, MPFR_RNDN);
  mpfr_max (z, x, y, MPFR_RNDN);
  if ( mpfr_cmp_ui (z, 42) )
    {
      printf ("Error in mpfr_max (17, 42)\n");
      exit (1);
    }
  mpfr_max (z, y, x, MPFR_RNDN);
  if ( mpfr_cmp_ui (z, 42) )
    {
      printf ("Error in mpfr_max (42, 17)\n");
      exit (1);
    }
  mpfr_min (z, y, x, MPFR_RNDN);
  if ( mpfr_cmp_ui (z, 17) )
    {
      printf ("Error in mpfr_min (42, 17)\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);

  tests_end_mpfr ();
  return 0;
}
