/* mpf_sqrt_ui -- Compute the square root of an unsigned integer.

Copyright 1993, 1994, 1996, 2000, 2001, 2004, 2005, 2015 Free Software
Foundation, Inc.

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

#include <stdio.h> /* for NULL */
#include "gmp.h"
#include "gmp-impl.h"


/* As usual the aim is to produce PREC(r) limbs of result with the high limb
   non-zero.  That high limb will end up floor(sqrt(u)), and limbs below are
   produced by padding the input with zeros, two for each desired result
   limb, being 2*(prec-1) for a total 2*prec-1 limbs passed to mpn_sqrtrem.
   The way mpn_sqrtrem calculates floor(sqrt(x)) ensures the root is correct
   to the intended accuracy, ie. truncated to prec limbs.

   With nails, u might be two limbs, in which case a total 2*prec limbs is
   passed to mpn_sqrtrem (still giving a prec limb result).  If uhigh is
   zero we adjust back to 2*prec-1, since mpn_sqrtrem requires the high
   non-zero.  2*prec limbs are always allocated, even when uhigh is zero, so
   the store of uhigh can be done without a conditional.

   u==0 is a special case so the rest of the code can assume the result is
   non-zero (ie. will have a non-zero high limb on the result).

   Not done:

   No attempt is made to identify perfect squares.  It's considered this can
   be left to an application if it might occur with any frequency.  As it
   stands, mpn_sqrtrem does its normal amount of work on a perfect square
   followed by zero limbs, though of course only an mpn_sqrtrem1 would be
   actually needed.  We also end up leaving our mpf result with lots of low
   trailing zeros, slowing down subsequent operations.

   We're not aware of any optimizations that can be made using the fact the
   input has lots of trailing zeros (apart from the perfect square
   case).  */


/* 1 if we (might) need two limbs for u */
#define U2   (GMP_NUMB_BITS < BITS_PER_ULONG)

void
mpf_sqrt_ui (mpf_ptr r, unsigned long int u)
{
  mp_size_t rsize, zeros;
  mp_ptr tp;
  mp_size_t prec;
  TMP_DECL;

  if (UNLIKELY (u <= 1))
    {
      SIZ (r) = EXP (r) = u;
      *PTR (r) = u;
      return;
    }

  TMP_MARK;

  prec = PREC (r);
  zeros = 2 * prec - 2;
  rsize = zeros + 1 + U2;

  tp = TMP_ALLOC_LIMBS (rsize);

  MPN_ZERO (tp, zeros);
  tp[zeros] = u & GMP_NUMB_MASK;

#if U2
  {
    mp_limb_t uhigh = u >> GMP_NUMB_BITS;
    tp[zeros + 1] = uhigh;
    rsize -= (uhigh == 0);
  }
#endif

  mpn_sqrtrem (PTR (r), NULL, tp, rsize);

  SIZ (r) = prec;
  EXP (r) = 1;
  TMP_FREE;
}
