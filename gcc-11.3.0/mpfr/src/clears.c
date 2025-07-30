/* mpfr_clears --  free the memory space allocated for several
   floating-point numbers

Copyright 2003-2004, 2006-2017 Free Software Foundation, Inc.
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
#undef HAVE_STDARG
#include "config.h"     /* for a build within gmp */
#endif

#if HAVE_STDARG
# include <stdarg.h>
#else
# include <varargs.h>
#endif

#include "mpfr-impl.h"

void
#if HAVE_STDARG
mpfr_clears (mpfr_ptr x, ...)
#else
mpfr_clears (va_alist)
 va_dcl
#endif
{
  va_list arg;

#if HAVE_STDARG
  va_start (arg, x);
#else
  mpfr_ptr x;
  va_start(arg);
  x =  va_arg (arg, mpfr_ptr);
#endif

  while (x != 0)
    {
      mpfr_clear (x);
      x = (mpfr_ptr) va_arg (arg, mpfr_ptr);
    }
  va_end (arg);
}
