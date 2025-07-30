/* mpf_set_prec(x) -- Change the precision of x.

Copyright 1993-1995, 2000, 2001 Free Software Foundation, Inc.

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


/* A full new_prec+1 limbs are always retained, even though just new_prec
   would satisfy the requested precision.  If size==new_prec+1 then
   certainly new_prec+1 should be kept since no copying is needed in that
   case.  If just new_prec was kept for size>new_prec+1 it'd be a bit
   inconsistent.  */

void
mpf_set_prec (mpf_ptr x, mp_bitcnt_t new_prec_in_bits)
{
  mp_size_t  old_prec, new_prec, new_prec_plus1;
  mp_size_t  size, sign;
  mp_ptr     xp;

  new_prec = __GMPF_BITS_TO_PREC (new_prec_in_bits);
  old_prec = PREC(x);

  /* do nothing if already the right precision */
  if (new_prec == old_prec)
    return;

  PREC(x) = new_prec;
  new_prec_plus1 = new_prec + 1;

  /* retain most significant limbs */
  sign = SIZ(x);
  size = ABS (sign);
  xp = PTR(x);
  if (size > new_prec_plus1)
    {
      SIZ(x) = (sign >= 0 ? new_prec_plus1 : -new_prec_plus1);
      MPN_COPY_INCR (xp, xp + size - new_prec_plus1, new_prec_plus1);
    }

  PTR(x) = __GMP_REALLOCATE_FUNC_LIMBS (xp, old_prec+1, new_prec_plus1);
}
