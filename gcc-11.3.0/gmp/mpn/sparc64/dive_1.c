/* UltraSPARC 64 mpn_divexact_1 -- mpn by limb exact division.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2000, 2001, 2003 Free Software Foundation, Inc.

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
#include "longlong.h"

#include "mpn/sparc64/sparc64.h"


/*                 64-bit divisor   32-bit divisor
                    cycles/limb      cycles/limb
                     (approx)         (approx)
   Ultrasparc 2i:      110               70
*/


/* There are two key ideas here to reduce mulx's.  Firstly when the divisor
   is 32-bits the high of q*d can be calculated without the two 32x32->64
   cross-products involving the high 32-bits of the divisor, that being zero
   of course.  Secondly umul_ppmm_lowequal and umul_ppmm_half_lowequal save
   one mulx (each) knowing the low of q*d is equal to the input limb l.

   For size==1, a simple udivx is used.  This is faster than calculating an
   inverse.

   For a 32-bit divisor and small sizes, an attempt was made at a simple
   udivx loop (two per 64-bit limb), but it turned out to be slower than
   mul-by-inverse.  At size==2 the inverse is about 260 cycles total
   compared to a udivx at 291.  Perhaps the latter would suit when size==2
   but the high 32-bits of the second limb is zero (saving one udivx), but
   it doesn't seem worth a special case just for that.  */

void
mpn_divexact_1 (mp_ptr dst, mp_srcptr src, mp_size_t size, mp_limb_t divisor)
{
  mp_limb_t  inverse, s, s_next, c, l, ls, q;
  unsigned   rshift, lshift;
  mp_limb_t  lshift_mask;
  mp_limb_t  divisor_h;

  ASSERT (size >= 1);
  ASSERT (divisor != 0);
  ASSERT (MPN_SAME_OR_SEPARATE_P (dst, src, size));
  ASSERT_MPN (src, size);
  ASSERT_LIMB (divisor);

  s = *src++;                 /* src low limb */
  size--;
  if (size == 0)
    {
      *dst = s / divisor;
      return;
    }

  if ((divisor & 1) == 0)
    {
      count_trailing_zeros (rshift, divisor);
      divisor >>= rshift;
    }
  else
    rshift = 0;

  binvert_limb (inverse, divisor);

  lshift = 64 - rshift;

  /* lshift==64 means no shift, so must mask out other part in this case */
  lshift_mask = (rshift == 0 ? 0 : MP_LIMB_T_MAX);

  c = 0;
  divisor_h = HIGH32 (divisor);

  if (divisor_h == 0)
    {
      /* 32-bit divisor */
      do
        {
          s_next = *src++;
          ls = (s >> rshift) | ((s_next << lshift) & lshift_mask);
          s = s_next;

          SUBC_LIMB (c, l, ls, c);

          q = l * inverse;
          *dst++ = q;

          umul_ppmm_half_lowequal (l, q, divisor, l);
          c += l;

          size--;
        }
      while (size != 0);

      ls = s >> rshift;
      l = ls - c;
      q = l * inverse;
      *dst = q;
    }
  else
    {
      /* 64-bit divisor */
      mp_limb_t  divisor_l = LOW32 (divisor);
      do
        {
          s_next = *src++;
          ls = (s >> rshift) | ((s_next << lshift) & lshift_mask);
          s = s_next;

          SUBC_LIMB (c, l, ls, c);

          q = l * inverse;
          *dst++ = q;

          umul_ppmm_lowequal (l, q, divisor, divisor_h, divisor_l, l);
          c += l;

          size--;
        }
      while (size != 0);

      ls = s >> rshift;
      l = ls - c;
      q = l * inverse;
      *dst = q;
    }
}
