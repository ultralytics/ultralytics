/* UltraSPARC 64 mpn_modexact_1c_odd -- mpn by limb exact style remainder.

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

#include "mpn/sparc64/sparc64.h"


/*                 64-bit divisor   32-bit divisor
                    cycles/limb      cycles/limb
                     (approx)         (approx)
   Ultrasparc 2i:       ?                ?
*/


/* This implementation reduces the number of multiplies done, knowing that
   on ultrasparc 1 and 2 the mulx instruction stalls the whole chip.

   The key idea is to use the fact that the low limb of q*d equals l, this
   being the whole purpose of the q calculated.  It means there's no need to
   calculate the lowest 32x32->64 part of the q*d, instead it can be
   inferred from l and the other three 32x32->64 parts.  See sparc64.h for
   details.

   When d is 32-bits, the same applies, but in this case there's only one
   other 32x32->64 part (ie. HIGH(q)*d).

   The net effect is that for 64-bit divisor each limb is 4 mulx, or for
   32-bit divisor each is 2 mulx.

   Enhancements:

   No doubt this could be done in assembler, if that helped the scheduling,
   or perhaps guaranteed good code irrespective of the compiler.

   Alternatives:

   It might be possibly to use floating point.  The loop is dominated by
   multiply latency, so not sure if floats would improve that.  One
   possibility would be to take two limbs at a time, with a 128 bit inverse,
   if there's enough registers, which could effectively use float throughput
   to reduce total latency across two limbs.  */

#define ASSERT_RETVAL(r)                \
  ASSERT (orig_c < d ? r < d : r <= d)

mp_limb_t
mpn_modexact_1c_odd (mp_srcptr src, mp_size_t size, mp_limb_t d, mp_limb_t orig_c)
{
  mp_limb_t  c = orig_c;
  mp_limb_t  s, l, q, h, inverse;

  ASSERT (size >= 1);
  ASSERT (d & 1);
  ASSERT_MPN (src, size);
  ASSERT_LIMB (d);
  ASSERT_LIMB (c);

  /* udivx is faster than 10 or 12 mulx's for one limb via an inverse */
  if (size == 1)
    {
      s = src[0];
      if (s > c)
	{
	  l = s-c;
	  h = l % d;
	  if (h != 0)
	    h = d - h;
	}
      else
	{
	  l = c-s;
	  h = l % d;
	}
      return h;
    }

  binvert_limb (inverse, d);

  if (d <= 0xFFFFFFFF)
    {
      s = *src++;
      size--;
      do
        {
          SUBC_LIMB (c, l, s, c);
          s = *src++;
          q = l * inverse;
          umul_ppmm_half_lowequal (h, q, d, l);
          c += h;
          size--;
        }
      while (size != 0);

      if (s <= d)
        {
          /* With high s <= d the final step can be a subtract and addback.
             If c==0 then the addback will restore to l>=0.  If c==d then
             will get l==d if s==0, but that's ok per the function
             definition.  */

          l = c - s;
          l += (l > c ? d : 0);

          ASSERT_RETVAL (l);
          return l;
        }
      else
        {
          /* Can't skip a divide, just do the loop code once more. */
          SUBC_LIMB (c, l, s, c);
          q = l * inverse;
          umul_ppmm_half_lowequal (h, q, d, l);
          c += h;

          ASSERT_RETVAL (c);
          return c;
        }
    }
  else
    {
      mp_limb_t  dl = LOW32 (d);
      mp_limb_t  dh = HIGH32 (d);
      long i;

      s = *src++;
      size--;
      do
        {
          SUBC_LIMB (c, l, s, c);
          s = *src++;
          q = l * inverse;
          umul_ppmm_lowequal (h, q, d, dh, dl, l);
          c += h;
          size--;
        }
      while (size != 0);

      if (s <= d)
        {
          /* With high s <= d the final step can be a subtract and addback.
             If c==0 then the addback will restore to l>=0.  If c==d then
             will get l==d if s==0, but that's ok per the function
             definition.  */

          l = c - s;
          l += (l > c ? d : 0);

          ASSERT_RETVAL (l);
          return l;
        }
      else
        {
          /* Can't skip a divide, just do the loop code once more. */
          SUBC_LIMB (c, l, s, c);
          q = l * inverse;
          umul_ppmm_lowequal (h, q, d, dh, dl, l);
          c += h;

          ASSERT_RETVAL (c);
          return c;
        }
    }
}
