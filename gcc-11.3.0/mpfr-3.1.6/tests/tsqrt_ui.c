/* Test file for mpfr_sqrt_ui.

Copyright 2000-2003, 2006-2017 Free Software Foundation, Inc.
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
check (unsigned long a, mpfr_rnd_t rnd_mode, const char *qs)
{
  mpfr_t q;

  mpfr_init2 (q, 53);
  mpfr_sqrt_ui (q, a, rnd_mode);
  if (mpfr_cmp_str1 (q, qs))
    {
      printf ("mpfr_sqrt_ui failed for a=%lu, rnd_mode=%s\n",
              a, mpfr_print_rnd_mode (rnd_mode));
      printf ("sqrt gives %s, mpfr_sqrt_ui gives ", qs);
      mpfr_out_str(stdout, 10, 0, q, MPFR_RNDN);
      exit (1);
    }
  mpfr_clear (q);
}

int
main (void)
{
  tests_start_mpfr ();

  check (0, MPFR_RNDN, "0.0");
  check (2116118, MPFR_RNDU, "1.45468828276026215e3");

  tests_end_mpfr ();
  return 0;
}
