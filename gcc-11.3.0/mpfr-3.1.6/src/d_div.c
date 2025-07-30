/* mpfr_d_div -- divide a machine double precision float
                 by a multiple precision floating-point number

Copyright 2007-2017 Free Software Foundation, Inc.
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

#include "mpfr-impl.h"

int
mpfr_d_div (mpfr_ptr a, double b, mpfr_srcptr c, mpfr_rnd_t rnd_mode)
{
  int inexact;
  mpfr_t d;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC (
    ("b=%.20g c[%Pu]=%*.Rg rnd=%d", b, mpfr_get_prec (c), mpfr_log_prec, c, rnd_mode),
    ("a[%Pu]=%*.Rg", mpfr_get_prec (a), mpfr_log_prec, a));

  MPFR_SAVE_EXPO_MARK (expo);

  mpfr_init2 (d, IEEE_DBL_MANT_DIG);
  inexact = mpfr_set_d (d, b, rnd_mode);
  MPFR_ASSERTN (inexact == 0);

  mpfr_clear_flags ();
  inexact = mpfr_div (a, d, c, rnd_mode);
  MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);

  mpfr_clear(d);
  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (a, inexact, rnd_mode);
}
