/* mpn_divexact_by3c -- mpn exact division by 3.

Copyright 2000-2003, 2008 Free Software Foundation, Inc.

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

#if DIVEXACT_BY3_METHOD == 0

mp_limb_t
mpn_divexact_by3c (mp_ptr rp, mp_srcptr up, mp_size_t un, mp_limb_t c)
{
  mp_limb_t r;
  r = mpn_bdiv_dbm1c (rp, up, un, GMP_NUMB_MASK / 3, GMP_NUMB_MASK / 3 * c);

  /* Possible bdiv_dbm1 return values are C * (GMP_NUMB_MASK / 3), 0 <= C < 3.
     We want to return C.  We compute the remainder mod 4 and notice that the
     inverse of (2^(2k)-1)/3 mod 4 is 1.  */
  return r & 3;
}

#endif

#if DIVEXACT_BY3_METHOD == 1

/* The algorithm here is basically the same as mpn_divexact_1, as described
   in the manual.  Namely at each step q = (src[i]-c)*inverse, and new c =
   borrow(src[i]-c) + high(divisor*q).  But because the divisor is just 3,
   high(divisor*q) can be determined with two comparisons instead of a
   multiply.

   The "c += ..."s add the high limb of 3*l to c.  That high limb will be 0,
   1 or 2.  Doing two separate "+="s seems to give better code on gcc (as of
   2.95.2 at least).

   It will be noted that the new c is formed by adding three values each 0
   or 1.  But the total is only 0, 1 or 2.  When the subtraction src[i]-c
   causes a borrow, that leaves a limb value of either 0xFF...FF or
   0xFF...FE.  The multiply by MODLIMB_INVERSE_3 gives 0x55...55 or
   0xAA...AA respectively, and in those cases high(3*q) is only 0 or 1
   respectively, hence a total of no more than 2.

   Alternatives:

   This implementation has each multiply on the dependent chain, due to
   "l=s-c".  See below for alternative code which avoids that.  */

mp_limb_t
mpn_divexact_by3c (mp_ptr restrict rp, mp_srcptr restrict up, mp_size_t un, mp_limb_t c)
{
  mp_limb_t  l, q, s;
  mp_size_t  i;

  ASSERT (un >= 1);
  ASSERT (c == 0 || c == 1 || c == 2);
  ASSERT (MPN_SAME_OR_SEPARATE_P (rp, up, un));

  i = 0;
  do
    {
      s = up[i];
      SUBC_LIMB (c, l, s, c);

      q = (l * MODLIMB_INVERSE_3) & GMP_NUMB_MASK;
      rp[i] = q;

      c += (q >= GMP_NUMB_CEIL_MAX_DIV3);
      c += (q >= GMP_NUMB_CEIL_2MAX_DIV3);
    }
  while (++i < un);

  ASSERT (c == 0 || c == 1 || c == 2);
  return c;
}


#endif

#if DIVEXACT_BY3_METHOD == 2

/* The following alternative code re-arranges the quotient calculation from
   (src[i]-c)*inverse to instead

       q = src[i]*inverse - c*inverse

   thereby allowing src[i]*inverse to be scheduled back as far as desired,
   making full use of multiplier throughput and leaving just some carry
   handing on the dependent chain.

   The carry handling consists of determining the c for the next iteration.
   This is the same as described above, namely look for any borrow from
   src[i]-c, and at the high of 3*q.

   high(3*q) is done with two comparisons as above (in c2 and c3).  The
   borrow from src[i]-c is incorporated into those by noting that if there's
   a carry then then we have src[i]-c == 0xFF..FF or 0xFF..FE, in turn
   giving q = 0x55..55 or 0xAA..AA.  Adding 1 to either of those q values is
   enough to make high(3*q) come out 1 bigger, as required.

   l = -c*inverse is calculated at the same time as c, since for most chips
   it can be more conveniently derived from separate c1/c2/c3 values than
   from a combined c equal to 0, 1 or 2.

   The net effect is that with good pipelining this loop should be able to
   run at perhaps 4 cycles/limb, depending on available execute resources
   etc.

   Usage:

   This code is not used by default, since we really can't rely on the
   compiler generating a good software pipeline, nor on such an approach
   even being worthwhile on all CPUs.

   Itanium is one chip where this algorithm helps though, see
   mpn/ia64/diveby3.asm.  */

mp_limb_t
mpn_divexact_by3c (mp_ptr restrict rp, mp_srcptr restrict up, mp_size_t un, mp_limb_t cy)
{
  mp_limb_t  s, sm, cl, q, qx, c2, c3;
  mp_size_t  i;

  ASSERT (un >= 1);
  ASSERT (cy == 0 || cy == 1 || cy == 2);
  ASSERT (MPN_SAME_OR_SEPARATE_P (rp, up, un));

  cl = cy == 0 ? 0 : cy == 1 ? -MODLIMB_INVERSE_3 : -2*MODLIMB_INVERSE_3;

  for (i = 0; i < un; i++)
    {
      s = up[i];
      sm = (s * MODLIMB_INVERSE_3) & GMP_NUMB_MASK;

      q = (cl + sm) & GMP_NUMB_MASK;
      rp[i] = q;
      qx = q + (s < cy);

      c2 = qx >= GMP_NUMB_CEIL_MAX_DIV3;
      c3 = qx >= GMP_NUMB_CEIL_2MAX_DIV3 ;

      cy = c2 + c3;
      cl = (-c2 & -MODLIMB_INVERSE_3) + (-c3 & -MODLIMB_INVERSE_3);
    }

  return cy;
}

#endif
