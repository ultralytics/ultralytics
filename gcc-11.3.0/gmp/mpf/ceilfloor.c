/* mpf_ceil, mpf_floor -- round an mpf to an integer.

Copyright 2001, 2004, 2012 Free Software Foundation, Inc.

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


/* dir==1 for ceil, dir==-1 for floor

   Notice the use of prec+1 ensures mpf_ceil and mpf_floor are equivalent to
   mpf_set if u is already an integer.  */

static void __gmpf_ceil_or_floor (REGPARM_2_1 (mpf_ptr, mpf_srcptr, int)) REGPARM_ATTR (1);
#define mpf_ceil_or_floor(r,u,dir)  __gmpf_ceil_or_floor (REGPARM_2_1 (r, u, dir))

REGPARM_ATTR (1) static void
mpf_ceil_or_floor (mpf_ptr r, mpf_srcptr u, int dir)
{
  mp_ptr     rp, up, p;
  mp_size_t  size, asize, prec;
  mp_exp_t   exp;

  size = SIZ(u);
  if (size == 0)
    {
    zero:
      SIZ(r) = 0;
      EXP(r) = 0;
      return;
    }

  rp = PTR(r);
  exp = EXP(u);
  if (exp <= 0)
    {
      /* u is only a fraction */
      if ((size ^ dir) < 0)
        goto zero;
      rp[0] = 1;
      EXP(r) = 1;
      SIZ(r) = dir;
      return;
    }
  EXP(r) = exp;

  up = PTR(u);
  asize = ABS (size);
  up += asize;

  /* skip fraction part of u */
  asize = MIN (asize, exp);

  /* don't lose precision in the copy */
  prec = PREC (r) + 1;

  /* skip excess over target precision */
  asize = MIN (asize, prec);

  up -= asize;

  if ((size ^ dir) >= 0)
    {
      /* rounding direction matches sign, must increment if ignored part is
         non-zero */
      for (p = PTR(u); p != up; p++)
        {
          if (*p != 0)
            {
              if (mpn_add_1 (rp, up, asize, CNST_LIMB(1)))
                {
                  /* was all 0xFF..FFs, which have become zeros, giving just
                     a carry */
                  rp[0] = 1;
                  asize = 1;
                  EXP(r)++;
                }
              SIZ(r) = (size >= 0 ? asize : -asize);
              return;
            }
        }
    }

  SIZ(r) = (size >= 0 ? asize : -asize);
  if (rp != up)
    MPN_COPY_INCR (rp, up, asize);
}


void
mpf_ceil (mpf_ptr r, mpf_srcptr u)
{
  mpf_ceil_or_floor (r, u, 1);
}

void
mpf_floor (mpf_ptr r, mpf_srcptr u)
{
  mpf_ceil_or_floor (r, u, -1);
}
