/* gcd_subdiv_step.c.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2003-2005, 2008, 2010, 2011 Free Software Foundation, Inc.

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

#include <stdlib.h>		/* for NULL */

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"

/* Used when mpn_hgcd or mpn_hgcd2 has failed. Then either one of a or
   b is small, or the difference is small. Perform one subtraction
   followed by one division. The normal case is to compute the reduced
   a and b, and return the new size.

   If s == 0 (used for gcd and gcdext), returns zero if the gcd is
   found.

   If s > 0, don't reduce to size <= s, and return zero if no
   reduction is possible (if either a, b or |a-b| is of size <= s). */

/* The hook function is called as

     hook(ctx, gp, gn, qp, qn, d)

   in the following cases:

   + If A = B at the start, G is the gcd, Q is NULL, d = -1.

   + If one input is zero at the start, G is the gcd, Q is NULL,
     d = 0 if A = G and d = 1 if B = G.

   Otherwise, if d = 0 we have just subtracted a multiple of A from B,
   and if d = 1 we have subtracted a multiple of B from A.

   + If A = B after subtraction, G is the gcd, Q is NULL.

   + If we get a zero remainder after division, G is the gcd, Q is the
     quotient.

   + Otherwise, G is NULL, Q is the quotient (often 1).

 */

mp_size_t
mpn_gcd_subdiv_step (mp_ptr ap, mp_ptr bp, mp_size_t n, mp_size_t s,
		     gcd_subdiv_step_hook *hook, void *ctx,
		     mp_ptr tp)
{
  static const mp_limb_t one = CNST_LIMB(1);
  mp_size_t an, bn, qn;

  int swapped;

  ASSERT (n > 0);
  ASSERT (ap[n-1] > 0 || bp[n-1] > 0);

  an = bn = n;
  MPN_NORMALIZE (ap, an);
  MPN_NORMALIZE (bp, bn);

  swapped = 0;

  /* Arrange so that a < b, subtract b -= a, and maintain
     normalization. */
  if (an == bn)
    {
      int c;
      MPN_CMP (c, ap, bp, an);
      if (UNLIKELY (c == 0))
	{
	  /* For gcdext, return the smallest of the two cofactors, so
	     pass d = -1. */
	  if (s == 0)
	    hook (ctx, ap, an, NULL, 0, -1);
	  return 0;
	}
      else if (c > 0)
	{
	  MP_PTR_SWAP (ap, bp);
	  swapped ^= 1;
	}
    }
  else
    {
      if (an > bn)
	{
	  MPN_PTR_SWAP (ap, an, bp, bn);
	  swapped ^= 1;
	}
    }
  if (an <= s)
    {
      if (s == 0)
	hook (ctx, bp, bn, NULL, 0, swapped ^ 1);
      return 0;
    }

  ASSERT_NOCARRY (mpn_sub (bp, bp, bn, ap, an));
  MPN_NORMALIZE (bp, bn);
  ASSERT (bn > 0);

  if (bn <= s)
    {
      /* Undo subtraction. */
      mp_limb_t cy = mpn_add (bp, ap, an, bp, bn);
      if (cy > 0)
	bp[an] = cy;
      return 0;
    }

  /* Arrange so that a < b */
  if (an == bn)
    {
      int c;
      MPN_CMP (c, ap, bp, an);
      if (UNLIKELY (c == 0))
	{
	  if (s > 0)
	    /* Just record subtraction and return */
	    hook (ctx, NULL, 0, &one, 1, swapped);
	  else
	    /* Found gcd. */
	    hook (ctx, bp, bn, NULL, 0, swapped);
	  return 0;
	}

      hook (ctx, NULL, 0, &one, 1, swapped);

      if (c > 0)
	{
	  MP_PTR_SWAP (ap, bp);
	  swapped ^= 1;
	}
    }
  else
    {
      hook (ctx, NULL, 0, &one, 1, swapped);

      if (an > bn)
	{
	  MPN_PTR_SWAP (ap, an, bp, bn);
	  swapped ^= 1;
	}
    }

  mpn_tdiv_qr (tp, bp, 0, bp, bn, ap, an);
  qn = bn - an + 1;
  bn = an;
  MPN_NORMALIZE (bp, bn);

  if (UNLIKELY (bn <= s))
    {
      if (s == 0)
	{
	  hook (ctx, ap, an, tp, qn, swapped);
	  return 0;
	}

      /* Quotient is one too large, so decrement it and add back A. */
      if (bn > 0)
	{
	  mp_limb_t cy = mpn_add (bp, ap, an, bp, bn);
	  if (cy)
	    bp[an++] = cy;
	}
      else
	MPN_COPY (bp, ap, an);

      MPN_DECR_U (tp, qn, 1);
    }

  hook (ctx, NULL, 0, tp, qn, swapped);
  return an;
}
