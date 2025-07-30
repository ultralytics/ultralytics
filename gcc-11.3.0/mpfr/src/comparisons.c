/* comparison predicates

Copyright 2002-2004, 2006-2017 Free Software Foundation, Inc.
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

/* Note: these functions currently use mpfr_cmp; they could have their
   own code to be faster. */

/*                          =     <     >     unordered
 * mpfr_greater_p           0     0     1     0
 * mpfr_greaterequal_p      1     0     1     0
 * mpfr_less_p              0     1     0     0
 * mpfr_lessequal_p         1     1     0     0
 * mpfr_lessgreater_p       0     1     1     0
 * mpfr_equal_p             1     0     0     0
 * mpfr_unordered_p         0     0     0     1
 */

int
mpfr_greater_p (mpfr_srcptr x, mpfr_srcptr y)
{
  return MPFR_IS_NAN(x) || MPFR_IS_NAN(y) ? 0 : (mpfr_cmp (x, y) > 0);
}

int
mpfr_greaterequal_p (mpfr_srcptr x, mpfr_srcptr y)
{
  return MPFR_IS_NAN(x) || MPFR_IS_NAN(y) ? 0 : (mpfr_cmp (x, y) >= 0);
}

int
mpfr_less_p (mpfr_srcptr x, mpfr_srcptr y)
{
  return MPFR_IS_NAN(x) || MPFR_IS_NAN(y) ? 0 : (mpfr_cmp (x, y) < 0);
}

int
mpfr_lessequal_p (mpfr_srcptr x, mpfr_srcptr y)
{
  return MPFR_IS_NAN(x) || MPFR_IS_NAN(y) ? 0 : (mpfr_cmp (x, y) <= 0);
}

int
mpfr_lessgreater_p (mpfr_srcptr x, mpfr_srcptr y)
{
  return MPFR_IS_NAN(x) || MPFR_IS_NAN(y) ? 0 : (mpfr_cmp (x, y) != 0);
}

int
mpfr_equal_p (mpfr_srcptr x, mpfr_srcptr y)
{
  return MPFR_IS_NAN(x) || MPFR_IS_NAN(y) ? 0 : (mpfr_cmp (x, y) == 0);
}

int
mpfr_unordered_p (mpfr_srcptr x, mpfr_srcptr y)
{
  return MPFR_IS_NAN(x) || MPFR_IS_NAN(y);
}
