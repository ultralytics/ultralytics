/* Test file for mpfr_nan_p, mpfr_inf_p, mpfr_number_p, mpfr_zero_p and
   mpfr_regular_p.

Copyright 2001-2004, 2006-2017 Free Software Foundation, Inc.
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
  mpfr_t  x;

  tests_start_mpfr ();

  mpfr_init (x);

  /* check +infinity gives non-zero for mpfr_inf_p only */
  mpfr_set_ui (x, 1L, MPFR_RNDZ);
  mpfr_div_ui (x, x, 0L, MPFR_RNDZ);
  if (mpfr_nan_p (x) || (mpfr_nan_p) (x) )
    {
      printf ("Error: mpfr_nan_p(+Inf) gives non-zero\n");
      exit (1);
    }
  if (mpfr_inf_p (x) == 0)
    {
      printf ("Error: mpfr_inf_p(+Inf) gives zero\n");
      exit (1);
    }
  if (mpfr_number_p (x) || (mpfr_number_p) (x) )
    {
      printf ("Error: mpfr_number_p(+Inf) gives non-zero\n");
      exit (1);
    }
  if (mpfr_zero_p (x) || (mpfr_zero_p) (x) )
    {
      printf ("Error: mpfr_zero_p(+Inf) gives non-zero\n");
      exit (1);
    }
  if (mpfr_regular_p (x) || (mpfr_regular_p) (x) )
    {
      printf ("Error: mpfr_regular_p(+Inf) gives non-zero\n");
      exit (1);
    }

  /* same for -Inf */
  mpfr_neg (x, x, MPFR_RNDN);
  if (mpfr_nan_p (x) || (mpfr_nan_p(x)))
    {
      printf ("Error: mpfr_nan_p(-Inf) gives non-zero\n");
      exit (1);
    }
  if (mpfr_inf_p (x) == 0)
    {
      printf ("Error: mpfr_inf_p(-Inf) gives zero\n");
      exit (1);
    }
  if (mpfr_number_p (x) || (mpfr_number_p)(x) )
    {
      printf ("Error: mpfr_number_p(-Inf) gives non-zero\n");
      exit (1);
    }
  if (mpfr_zero_p (x) || (mpfr_zero_p)(x) )
    {
      printf ("Error: mpfr_zero_p(-Inf) gives non-zero\n");
      exit (1);
    }
  if (mpfr_regular_p (x) || (mpfr_regular_p) (x) )
    {
      printf ("Error: mpfr_regular_p(-Inf) gives non-zero\n");
      exit (1);
    }

  /* same for NaN */
  mpfr_sub (x, x, x, MPFR_RNDN);
  if (mpfr_nan_p (x) == 0)
    {
      printf ("Error: mpfr_nan_p(NaN) gives zero\n");
      exit (1);
    }
  if (mpfr_inf_p (x) || (mpfr_inf_p)(x) )
    {
      printf ("Error: mpfr_inf_p(NaN) gives non-zero\n");
      exit (1);
    }
  if (mpfr_number_p (x) || (mpfr_number_p) (x) )
    {
      printf ("Error: mpfr_number_p(NaN) gives non-zero\n");
      exit (1);
    }
  if (mpfr_zero_p (x) || (mpfr_zero_p)(x) )
    {
      printf ("Error: mpfr_number_p(NaN) gives non-zero\n");
      exit (1);
    }
  if (mpfr_regular_p (x) || (mpfr_regular_p) (x) )
    {
      printf ("Error: mpfr_regular_p(NaN) gives non-zero\n");
      exit (1);
    }

  /* same for a regular number */
  mpfr_set_ui (x, 1, MPFR_RNDN);
  if (mpfr_nan_p (x) || (mpfr_nan_p)(x))
    {
      printf ("Error: mpfr_nan_p(1) gives non-zero\n");
      exit (1);
    }
  if (mpfr_inf_p (x) || (mpfr_inf_p)(x) )
    {
      printf ("Error: mpfr_inf_p(1) gives non-zero\n");
      exit (1);
    }
  if (mpfr_number_p (x) == 0)
    {
      printf ("Error: mpfr_number_p(1) gives zero\n");
      exit (1);
    }
  if (mpfr_zero_p (x) || (mpfr_zero_p) (x) )
    {
      printf ("Error: mpfr_zero_p(1) gives non-zero\n");
      exit (1);
    }
  if (mpfr_regular_p (x) == 0 || (mpfr_regular_p) (x) == 0)
    {
      printf ("Error: mpfr_regular_p(1) gives zero\n");
      exit (1);
    }

  /* Same for +0 */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  if (mpfr_nan_p (x) || (mpfr_nan_p)(x))
    {
      printf ("Error: mpfr_nan_p(+0) gives non-zero\n");
      exit (1);
    }
  if (mpfr_inf_p (x) || (mpfr_inf_p)(x) )
    {
      printf ("Error: mpfr_inf_p(+0) gives non-zero\n");
      exit (1);
    }
  if (mpfr_number_p (x) == 0)
    {
      printf ("Error: mpfr_number_p(+0) gives zero\n");
      exit (1);
    }
  if (mpfr_zero_p (x) == 0 )
    {
      printf ("Error: mpfr_zero_p(+0) gives zero\n");
      exit (1);
    }
  if (mpfr_regular_p (x) || (mpfr_regular_p) (x) )
    {
      printf ("Error: mpfr_regular_p(+0) gives non-zero\n");
      exit (1);
    }

  /* Same for -0 */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  if (mpfr_nan_p (x) || (mpfr_nan_p)(x))
    {
      printf ("Error: mpfr_nan_p(-0) gives non-zero\n");
      exit (1);
    }
  if (mpfr_inf_p (x) || (mpfr_inf_p)(x) )
    {
      printf ("Error: mpfr_inf_p(-0) gives non-zero\n");
      exit (1);
    }
  if (mpfr_number_p (x) == 0)
    {
      printf ("Error: mpfr_number_p(-0) gives zero\n");
      exit (1);
    }
  if (mpfr_zero_p (x) == 0 )
    {
      printf ("Error: mpfr_zero_p(-0) gives zero\n");
      exit (1);
    }
  if (mpfr_regular_p (x) || (mpfr_regular_p) (x) )
    {
      printf ("Error: mpfr_regular_p(-0) gives non-zero\n");
      exit (1);
    }

  mpfr_clear (x);

  tests_end_mpfr ();
  return 0;
}
