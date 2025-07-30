/* mpfr_set_sj -- set a MPFR number from a huge machine signed integer

Copyright 2004-2017 Free Software Foundation, Inc.
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

#ifdef HAVE_CONFIG_H
# include "config.h"       /* for a build within gmp */
#endif

#include "mpfr-intmax.h"
#include "mpfr-impl.h"

#ifdef _MPFR_H_HAVE_INTMAX_T

int
mpfr_set_sj (mpfr_t x, intmax_t j, mpfr_rnd_t rnd)
{
  return mpfr_set_sj_2exp (x, j, 0, rnd);
}

int
mpfr_set_sj_2exp (mpfr_t x, intmax_t j, intmax_t e, mpfr_rnd_t rnd)
{
  if (j>=0)
    return mpfr_set_uj_2exp (x, j, e, rnd);
  else
    {
      int inex;
      inex = mpfr_set_uj_2exp (x, - (uintmax_t) j, e, MPFR_INVERT_RND (rnd));
      MPFR_CHANGE_SIGN (x);
      return -inex;
    }
}

#endif
