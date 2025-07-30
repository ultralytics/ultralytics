/* mpf_trunc -- truncate an mpf to an integer.

Copyright 2001 Free Software Foundation, Inc.

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


/* Notice the use of prec+1 ensures mpf_trunc is equivalent to mpf_set if u
   is already an integer.  */

void
mpf_trunc (mpf_ptr r, mpf_srcptr u)
{
  mp_ptr     rp;
  mp_srcptr  up;
  mp_size_t  size, asize, prec;
  mp_exp_t   exp;

  exp = EXP(u);
  size = SIZ(u);
  if (size == 0 || exp <= 0)
    {
      /* u is only a fraction */
      SIZ(r) = 0;
      EXP(r) = 0;
      return;
    }

  up = PTR(u);
  EXP(r) = exp;
  asize = ABS (size);
  up += asize;

  /* skip fraction part of u */
  asize = MIN (asize, exp);

  /* don't lose precision in the copy */
  prec = PREC(r) + 1;

  /* skip excess over target precision */
  asize = MIN (asize, prec);

  up -= asize;
  rp = PTR(r);
  SIZ(r) = (size >= 0 ? asize : -asize);
  if (rp != up)
    MPN_COPY_INCR (rp, up, asize);
}
