/* mpfr_sec - secant function = 1/cos.

Copyright 2005-2017 Free Software Foundation, Inc.
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

#define FUNCTION mpfr_sec
#define INVERSE  mpfr_cos
#define ACTION_NAN(y) do { MPFR_SET_NAN(y); MPFR_RET_NAN; } while (1)
#define ACTION_INF(y) do { MPFR_SET_NAN(y); MPFR_RET_NAN; } while (1)
#define ACTION_ZERO(y,x) return mpfr_set_ui (y, 1, rnd_mode)
/* for x near 0, sec(x) = 1 + x^2/2 + ..., more precisely |sec(x)-1| < x^2
   for |x| <= 1. */
#define ACTION_TINY(y,x,r) \
  MPFR_FAST_COMPUTE_IF_SMALL_INPUT(y, __gmpfr_one, -2 * MPFR_GET_EXP (x), 0, \
                                   1, r, inexact = _inexact; goto end)

#include "gen_inverse.h"
