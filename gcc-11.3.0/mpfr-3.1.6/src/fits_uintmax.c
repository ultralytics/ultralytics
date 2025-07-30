/* mpfr_fits_uintmax_p -- test whether an mpfr fits an uintmax_t.

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

#ifdef HAVE_CONFIG_H
# include "config.h"            /* for a build within gmp */
#endif

#include "mpfr-intmax.h"
#include "mpfr-impl.h"

/* Note: though mpfr-impl.h is included in fits_u.h, we also include it
   above so that it gets included even when _MPFR_H_HAVE_INTMAX_T is not
   defined; this is necessary to avoid an empty translation unit, which
   is forbidden by ISO C. Without this, a failing test can be reproduced
   by creating an invalid stdint.h somewhere in the default include path
   and by compiling MPFR with "gcc -ansi -pedantic-errors". */

#ifdef _MPFR_H_HAVE_INTMAX_T

#define FUNCTION   mpfr_fits_uintmax_p
#define MAXIMUM    MPFR_UINTMAX_MAX
#define TYPE       uintmax_t

#include "fits_u.h"

#endif
