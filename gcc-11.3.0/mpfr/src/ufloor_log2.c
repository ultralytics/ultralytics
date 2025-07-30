/* __gmpfr_floor_log2 - returns floor(log(d)/log(2))

Copyright 1999-2004, 2006-2017 Free Software Foundation, Inc.
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

/* returns floor(log2(d)) */
long
__gmpfr_floor_log2 (double d)
{
#if _GMP_IEEE_FLOATS
  union ieee_double_extract x;

  x.d = d;
  return (long) x.s.exp - 1023;
#else
  long exp;
  double m;

  MPFR_ASSERTD (d >= 0);
  if (d >= 1.0)
    {
      exp = -1;
      for( m= 1.0 ; m <= d ; m *=2.0 )
        exp++;
    }
  else
    {
      exp = 0;
      for( m= 1.0 ; m > d ; m *= (1.0/2.0) )
        exp--;
    }
  return exp;
#endif
}
