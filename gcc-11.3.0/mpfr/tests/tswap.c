/* Test file for mpfr_swap.

Copyright 2000-2004, 2006-2017 Free Software Foundation, Inc.
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
  mpfr_t u, v;

  tests_start_mpfr ();

  mpfr_init2 (u, 24);
  mpfr_init2 (v, 53);
  mpfr_set_ui (u, 16777215, MPFR_RNDN); /* 2^24 - 1 */
  mpfr_set_str1 (v, "9007199254740991.0"); /* 2^53 - 1 */
  mpfr_swap (u, v);
  mpfr_swap (u, v);
  if (mpfr_cmp_ui (u, 16777215) || mpfr_cmp_str1 (v, "9007199254740991.0"))
    {
      printf ("Error in mpfr_swap\n");
      exit (1);
    }
  mpfr_clear (u);
  mpfr_clear (v);

  tests_end_mpfr ();
  return 0;
}
