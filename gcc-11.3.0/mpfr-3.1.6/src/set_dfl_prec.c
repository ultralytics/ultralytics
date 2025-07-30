/* mpfr_set_default_prec, mpfr_get_default_prec -- set/get default precision

Copyright 1999-2001, 2004-2017 Free Software Foundation, Inc.
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

/* default is IEEE double precision, i.e. 53 bits */
MPFR_THREAD_VAR (mpfr_prec_t, __gmpfr_default_fp_bit_precision,
                 IEEE_DBL_MANT_DIG)

void
mpfr_set_default_prec (mpfr_prec_t prec)
{
  MPFR_ASSERTN (prec >= MPFR_PREC_MIN && prec <= MPFR_PREC_MAX);
  __gmpfr_default_fp_bit_precision = prec;
}

#undef mpfr_get_default_prec
mpfr_prec_t
mpfr_get_default_prec (void)
{
  return __gmpfr_default_fp_bit_precision;
}
