/* mpn_divexact_1 -- mpn by limb exact division.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2000-2003, 2005, 2013 Free Software Foundation, Inc.

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



/* Divide a={src,size} by d=divisor and store the quotient in q={dst,size}.
   q will only be correct if d divides a exactly.

   A separate loop is used for shift==0 because n<<GMP_LIMB_BITS doesn't
   give zero on all CPUs (for instance it doesn't on the x86s).  This
   separate loop might run faster too, helping odd divisors.

   Possibilities:

   mpn_divexact_1c could be created, accepting and returning c.  This would
   let a long calculation be done piece by piece.  Currently there's no
   particular need for that, and not returning c means that a final umul can
   be skipped.

   Another use for returning c would be letting the caller know whether the
   division was in fact exact.  It would work just to return the carry bit
   "c=(l>s)" and let the caller do a final umul if interested.

   When the divisor is even, the factors of two could be handled with a
   separate mpn_rshift, instead of shifting on the fly.  That might be
   faster on some CPUs and would mean just the shift==0 style loop would be
   needed.

   If n<<GMP_LIMB_BITS gives zero on a particular CPU then the separate
   shift==0 loop is unnecessary, and could be eliminated if there's no great
   speed difference.

   It's not clear whether "/" is the best way to handle size==1.  Alpha gcc
   2.95 for instance has a poor "/" and might prefer the modular method.
   Perhaps a tuned parameter should control this.

   If src[size-1] < divisor then dst[size-1] will be zero, and one divide
   step could be skipped.  A test at last step for s<divisor (or ls in the
   even case) might be a good way to do that.  But if this code is often
   used with small divisors then it might not be worth bothering  */

void
mpn_divexact_1 (mp_ptr dst, mp_srcptr src, mp_size_t size, mp_limb_t divisor)
{
  mp_size_t  i;
  mp_limb_t  c, h, l, ls, s, s_next, inverse, dummy;
  unsigned   shift;

  ASSERT (size >= 1);
  ASSERT (divisor != 0);
  ASSERT (MPN_SAME_OR_SEPARATE_P (dst, src, size));
  ASSERT_MPN (src, size);
  ASSERT_LIMB (divisor);

  if ((divisor & 1) == 0)
    {
      count_trailing_zeros (shift, divisor);
      divisor >>= shift;
    }
  else
    shift = 0;

  binvert_limb (inverse, divisor);
  divisor <<= GMP_NAIL_BITS;

  if (shift != 0)
    {
      c = 0;

      s = src[0];

      for (i = 1; i < size; i++)
	{
	  s_next = src[i];
	  ls = ((s >> shift) | (s_next << (GMP_NUMB_BITS-shift))) & GMP_NUMB_MASK;
	  s = s_next;

	  SUBC_LIMB (c, l, ls, c);

	  l = (l * inverse) & GMP_NUMB_MASK;
	  dst[i - 1] = l;

	  umul_ppmm (h, dummy, l, divisor);
	  c += h;
	}
      while (i < size);

      ls = s >> shift;
      l = ls - c;
      l = (l * inverse) & GMP_NUMB_MASK;
      dst[size - 1] = l;
    }
  else
    {
      s = src[0];

      l = (s * inverse) & GMP_NUMB_MASK;
      dst[0] = l;
      c = 0;

      for (i = 1; i < size; i++)
	{
	  umul_ppmm (h, dummy, l, divisor);
	  c += h;

	  s = src[i];
	  SUBC_LIMB (c, l, s, c);

	  l = (l * inverse) & GMP_NUMB_MASK;
	  dst[i] = l;
	}
    }
}
