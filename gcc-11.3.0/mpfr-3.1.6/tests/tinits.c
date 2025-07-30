/* Test file for mpfr_init2, mpfr_inits, mpfr_inits2 and mpfr_clears.

Copyright 2003, 2006-2017 Free Software Foundation, Inc.
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

#include <stdlib.h>

#include "mpfr-test.h"

int
main (void)
{
  mpfr_t a, b, c;
  long large_prec;

  tests_start_mpfr ();

  mpfr_inits (a, b, c, (mpfr_ptr) 0);
  mpfr_clears (a, b, c, (mpfr_ptr) 0);
  mpfr_inits2 (200, a, b, c, (mpfr_ptr) 0);
  mpfr_clears (a, b, c, (mpfr_ptr) 0);

  /* test for precision 2^31-1, see
     https://gforge.inria.fr/tracker/index.php?func=detail&aid=13918 */
  large_prec = 2147483647;
  if (getenv ("MPFR_CHECK_LARGEMEM") != NULL)
    {
      /* We assume that the precision won't be increased internally. */
      if (large_prec > MPFR_PREC_MAX)
        large_prec = MPFR_PREC_MAX;
      mpfr_inits2 (large_prec, a, b, (mpfr_ptr) 0);
      mpfr_set_ui (a, 17, MPFR_RNDN);
      mpfr_set (b, a, MPFR_RNDN);
      if (mpfr_get_ui (a, MPFR_RNDN) != 17)
        {
          printf ("Error in mpfr_init2 with precision 2^31-1\n");
          exit (1);
        }
      mpfr_clears (a, b, (mpfr_ptr) 0);
    }

  tests_end_mpfr ();

  return 0;
}
