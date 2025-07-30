/* mpfr_set_machine_rnd_mode -- set the rounding mode for machine floats

Copyright 1999, 2001-2002, 2006-2017 Free Software Foundation, Inc.
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

#include "mpfr-test.h"

/* It is important to test each FE_* macro -- see the ISO C99 standard.
   For instance, with some ARM implementations, only FE_TONEAREST may
   be defined. */

/* sets the machine rounding mode to the value rnd_mode */
int
mpfr_set_machine_rnd_mode (mpfr_rnd_t rnd_mode)
{
  switch (rnd_mode)
    {
    case MPFR_RNDN:
      return
#if defined (MPFR_HAVE_FESETROUND) && defined (FE_TONEAREST)
        fesetround(FE_TONEAREST)
#else
        -1
#endif
        ;
    case MPFR_RNDZ:
      return
#if defined (MPFR_HAVE_FESETROUND) && defined (FE_TOWARDZERO)
        fesetround(FE_TOWARDZERO)
#else
        -1
#endif
        ;
    case MPFR_RNDU:
      return
#if defined (MPFR_HAVE_FESETROUND) && defined (FE_UPWARD)
        fesetround(FE_UPWARD)
#else
        -1
#endif
        ;
    case MPFR_RNDD:
      return
#if defined (MPFR_HAVE_FESETROUND) && defined (FE_DOWNWARD)
        fesetround(FE_DOWNWARD)
#else
        -1
#endif
        ;
    default:
      return -1;
    }
}
