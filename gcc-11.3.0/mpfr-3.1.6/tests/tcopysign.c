/* Test file for mpfr_copysign, mpfr_setsign and mpfr_signbit.

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

static void
copysign_variant (mpfr_ptr z, mpfr_srcptr x, mpfr_srcptr y,
                  mpfr_rnd_t rnd_mode, int k)
{
  mpfr_clear_flags ();
  switch (k)
    {
    case 0:
      mpfr_copysign (z, x, y, MPFR_RNDN);
      return;
    case 1:
      (mpfr_copysign) (z, x, y, MPFR_RNDN);
      return;
    case 2:
      mpfr_setsign (z, x, mpfr_signbit (y), MPFR_RNDN);
      return;
    case 3:
      mpfr_setsign (z, x, (mpfr_signbit) (y), MPFR_RNDN);
      return;
    case 4:
      (mpfr_setsign) (z, x, mpfr_signbit (y), MPFR_RNDN);
      return;
    case 5:
      (mpfr_setsign) (z, x, (mpfr_signbit) (y), MPFR_RNDN);
      return;
    }
}

int
main (void)
{
  mpfr_t x, y, z;
  int i, j, k;

  tests_start_mpfr ();

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (z);

  for (i = 0; i <= 1; i++)
    for (j = 0; j <= 1; j++)
      for (k = 0; k <= 5; k++)
        {
          mpfr_set_nan (x);
          i ? MPFR_SET_NEG (x) : MPFR_SET_POS (x);
          mpfr_set_nan (y);
          j ? MPFR_SET_NEG (y) : MPFR_SET_POS (y);
          copysign_variant (z, x, y, MPFR_RNDN, k);
          if (MPFR_SIGN (z) != MPFR_SIGN (y) || !mpfr_nanflag_p ())
            {
              printf ("Error in mpfr_copysign (%cNaN, %cNaN)\n",
                      i ? '-' : '+', j ? '-' : '+');
              exit (1);
            }

          mpfr_set_si (x, i ? -1250 : 1250, MPFR_RNDN);
          mpfr_set_nan (y);
          j ? MPFR_SET_NEG (y) : MPFR_SET_POS (y);
          copysign_variant (z, x, y, MPFR_RNDN, k);
          if (i != j)
            mpfr_neg (x, x, MPFR_RNDN);
          if (! mpfr_equal_p (z, x) || mpfr_nanflag_p ())
            {
              printf ("Error in mpfr_copysign (%c1250, %cNaN)\n",
                      i ? '-' : '+', j ? '-' : '+');
              exit (1);
            }

          mpfr_set_si (x, i ? -1250 : 1250, MPFR_RNDN);
          mpfr_set_si (y, j ? -1717 : 1717, MPFR_RNDN);
          copysign_variant (z, x, y, MPFR_RNDN, k);
          if (i != j)
            mpfr_neg (x, x, MPFR_RNDN);
          if (! mpfr_equal_p (z, x) || mpfr_nanflag_p ())
            {
              printf ("Error in mpfr_copysign (%c1250, %c1717)\n",
                      i ? '-' : '+', j ? '-' : '+');
              exit (1);
            }
        }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);

  tests_end_mpfr ();
  return 0;
}
