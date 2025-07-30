/* __gmpfr_isqrt && __gmpfr_cuberoot -- Integer square root and cube root

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

#include "mpfr-impl.h"

/* returns floor(sqrt(n)) */
unsigned long
__gmpfr_isqrt (unsigned long n)
{
  unsigned long i, s;

  /* First find an approximation to floor(sqrt(n)) of the form 2^k. */
  i = n;
  s = 1;
  while (i >= 2)
    {
      i >>= 2;
      s <<= 1;
    }

  do
    {
      s = (s + n / s) / 2;
    }
  while (!(s*s <= n && (s*s > s*(s+2) || n <= s*(s+2))));
  /* Short explanation: As mathematically s*(s+2) < 2*ULONG_MAX,
     the condition s*s > s*(s+2) is evaluated as true when s*(s+2)
     "overflows" but not s*s. This implies that mathematically, one
     has s*s <= n <= s*(s+2). If s*s "overflows", this means that n
     is "large" and the inequality n <= s*(s+2) cannot be satisfied. */
  return s;
}

/* returns floor(n^(1/3)) */
unsigned long
__gmpfr_cuberoot (unsigned long n)
{
  unsigned long i, s;

  /* First find an approximation to floor(cbrt(n)) of the form 2^k. */
  i = n;
  s = 1;
  while (i >= 4)
    {
      i >>= 3;
      s <<= 1;
    }

  /* Improve the approximation (this is necessary if n is large, so that
     mathematically (s+1)*(s+1)*(s+1) isn't much larger than ULONG_MAX). */
  if (n >= 256)
    {
      s = (2 * s + n / (s * s)) / 3;
      s = (2 * s + n / (s * s)) / 3;
      s = (2 * s + n / (s * s)) / 3;
    }

  do
    {
      s = (2 * s + n / (s * s)) / 3;
    }
  while (!(s*s*s <= n && (s*s*s > (s+1)*(s+1)*(s+1) ||
                          n < (s+1)*(s+1)*(s+1))));
  return s;
}
