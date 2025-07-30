/* mpfr_set_str -- set a floating-point number from a string

Copyright 2000-2017 Free Software Foundation, Inc.
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
mpfr_set_str (mpfr_t x, const char *str, int base, mpfr_rnd_t rnd)
{
  char *p;

  if (MPFR_UNLIKELY (*str == 0))
    return -1;
  mpfr_strtofr (x, str, &p, base, rnd);
  return (*p == 0) ? 0 : -1;
}


int
mpfr_init_set_str (mpfr_ptr x, const char *str, int base, mpfr_rnd_t rnd)
{
  mpfr_init (x);
  return mpfr_set_str (x, str, base, rnd);
}
