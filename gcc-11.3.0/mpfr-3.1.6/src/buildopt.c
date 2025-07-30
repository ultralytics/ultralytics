/* buildopt.c -- functions giving information about options used during the
   mpfr library compilation

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

#include "mpfr-impl.h"

int
mpfr_buildopt_tls_p (void)
{
#ifdef MPFR_USE_THREAD_SAFE
  return 1;
#else
  return 0;
#endif
}

int
mpfr_buildopt_decimal_p (void)
{
#ifdef MPFR_WANT_DECIMAL_FLOATS
  return 1;
#else
  return 0;
#endif
}

int
mpfr_buildopt_gmpinternals_p (void)
{
#if defined(MPFR_HAVE_GMP_IMPL) || defined(WANT_GMP_INTERNALS)
  return 1;
#else
  return 0;
#endif
}

const char *mpfr_buildopt_tune_case (void)
{
#ifdef MPFR_TUNE_CASE
  return MPFR_TUNE_CASE;
#else
  return "Generic thresholds";
#endif
}
