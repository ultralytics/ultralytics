/* mpz_cmpabs(u,v) -- Compare U, V.  Return positive, zero, or negative
   based on if U > V, U == V, or U < V.

Copyright 1991, 1993, 1994, 1996, 1997, 2000-2002 Free Software Foundation,
Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

#include "gmp.h"
#include "gmp-impl.h"


int
mpz_cmpabs (mpz_srcptr u, mpz_srcptr v) __GMP_NOTHROW
{
  mp_size_t  usize, vsize, dsize;
  mp_srcptr  up, vp;
  int        cmp;

  usize = ABSIZ (u);
  vsize = ABSIZ (v);
  dsize = usize - vsize;
  if (dsize != 0)
    return dsize;

  up = PTR(u);
  vp = PTR(v);
  MPN_CMP (cmp, up, vp, usize);
  return cmp;
}
