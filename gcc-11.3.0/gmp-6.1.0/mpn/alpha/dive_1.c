/* Alpha mpn_divexact_1 -- mpn by limb exact division.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2000-2003 Free Software Foundation, Inc.

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


/*      cycles/limb
   EV4:    47.0
   EV5:    30.0
   EV6:    15.0
*/


/* The dependent chain is as follows (the same as modexact), and this is
   what the code runs as.

       ev4    ev5   ev6
        1      1     1    sub    y = x - h
       23     13     7    mulq   q = y * inverse
       23     15     7    umulh  h = high (q * d)
       --     --    --
       47     30    15

   The time to load src[i+1] and establish x hides under the umulh latency.  */

void
mpn_divexact_1 (mp_ptr dst, mp_srcptr src, mp_size_t size, mp_limb_t divisor)
{
  mp_limb_t  inverse, lshift_mask, s, sr, s_next, c, h, x, y, q, dummy;
  unsigned   rshift, lshift;

  ASSERT (size >= 1);
  ASSERT (divisor != 0);
  ASSERT (MPN_SAME_OR_SEPARATE_P (dst, src, size));
  ASSERT_MPN (src, size);
  ASSERT_LIMB (divisor);

  s_next = *src++;   /* src[0] */

  rshift = 0;
  lshift_mask = 0;
  if ((divisor & 1) == 0)
    {
      count_trailing_zeros (rshift, divisor);
      lshift_mask = MP_LIMB_T_MAX;
      divisor >>= rshift;
    }

  binvert_limb (inverse, divisor);
  lshift = 64 - rshift;

  c = 0;
  h = 0;
  sr = s_next >> rshift;

  size--;
  if (LIKELY (size != 0))
    {
      do
        {
          s_next = *src++;      /* src[i+1] */
          s = sr | ((s_next << lshift) & lshift_mask);
          x = s - c;
          c = s < c;
          sr = s_next >> rshift;

          y = x - h;
          c += (x < h);
          q = y * inverse;
          *dst++ = q;
          umul_ppmm (h, dummy, q, divisor);

          size--;
        }
      while (size != 0);
    }

  x = sr - c;
  y = x - h;
  q = y * inverse;
  *dst = q;         /* dst[size-1] */
}
