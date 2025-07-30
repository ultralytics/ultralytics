/* Test file for mpfr_set_exp.

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
main (int argc, char *argv[])
{
  mpfr_t x;
  int ret;
  mpfr_exp_t emin, emax;

  tests_start_mpfr ();

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  mpfr_init (x);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  ret = mpfr_set_exp (x, 2);
  MPFR_ASSERTN(ret == 0 && mpfr_cmp_ui (x, 2) == 0);

  set_emin (-1);
  ret = mpfr_set_exp (x, -1);
  MPFR_ASSERTN(ret == 0 && mpfr_cmp_ui_2exp (x, 1, -2) == 0);

  set_emax (1);
  ret = mpfr_set_exp (x, 1);
  MPFR_ASSERTN(ret == 0 && mpfr_cmp_ui (x, 1) == 0);

  ret = mpfr_set_exp (x, -2);
  MPFR_ASSERTN(ret != 0 && mpfr_cmp_ui (x, 1) == 0);

  ret = mpfr_set_exp (x, 2);
  MPFR_ASSERTN(ret != 0 && mpfr_cmp_ui (x, 1) == 0);

  mpfr_clear (x);

  set_emin (emin);
  set_emax (emax);

  tests_end_mpfr ();
  return 0;
}
