/* mpn_modexact_1c_odd -- mpn by limb exact division style remainder.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2000-2004 Free Software Foundation, Inc.

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


/* Calculate an r satisfying

           r*B^k + a - c == q*d

   where B=2^GMP_LIMB_BITS, a is {src,size}, k is either size or size-1
   (the caller won't know which), and q is the quotient (discarded).  d must
   be odd, c can be any limb value.

   If c<d then r will be in the range 0<=r<d, or if c>=d then 0<=r<=d.

   This slightly strange function suits the initial Nx1 reduction for GCDs
   or Jacobi symbols since the factors of 2 in B^k can be ignored, leaving
   -r == a mod d (by passing c=0).  For a GCD the factor of -1 on r can be
   ignored, or for the Jacobi symbol it can be accounted for.  The function
   also suits divisibility and congruence testing since if r=0 (or r=d) is
   obtained then a==c mod d.


   r is a bit like the remainder returned by mpn_divexact_by3c, and is the
   sort of remainder mpn_divexact_1 might return.  Like mpn_divexact_by3c, r
   represents a borrow, since effectively quotient limbs are chosen so that
   subtracting that multiple of d from src at each step will produce a zero
   limb.

   A long calculation can be done piece by piece from low to high by passing
   the return value from one part as the carry parameter to the next part.
   The effective final k becomes anything between size and size-n, if n
   pieces are used.


   A similar sort of routine could be constructed based on adding multiples
   of d at each limb, much like redc in mpz_powm does.  Subtracting however
   has a small advantage that when subtracting to cancel out l there's never
   a borrow into h, whereas using an addition would put a carry into h
   depending whether l==0 or l!=0.


   In terms of efficiency, this function is similar to a mul-by-inverse
   mpn_mod_1.  Both are essentially two multiplies and are best suited to
   CPUs with low latency multipliers (in comparison to a divide instruction
   at least.)  But modexact has a few less supplementary operations, only
   needs low part and high part multiplies, and has fewer working quantities
   (helping CPUs with few registers).


   In the main loop it will be noted that the new carry (call it r) is the
   sum of the high product h and any borrow from l=s-c.  If c<d then we will
   have r<d too, for the following reasons.  Let q=l*inverse be the quotient
   limb, so that q*d = B*h + l, where B=2^GMP_NUMB_BITS.  Now if h=d-1 then

       l = q*d - B*(d-1) <= (B-1)*d - B*(d-1) = B-d

   But if l=s-c produces a borrow when c<d, then l>=B-d+1 and hence will
   never have h=d-1 and so r=h+borrow <= d-1.

   When c>=d, on the other hand, h=d-1 can certainly occur together with a
   borrow, thereby giving only r<=d, as per the function definition above.

   As a design decision it's left to the caller to check for r=d if it might
   be passing c>=d.  Several applications have c<d initially so the extra
   test is often unnecessary, for example the GCDs or a plain divisibility
   d|a test will pass c=0.


   The special case for size==1 is so that it can be assumed c<=d in the
   high<=divisor test at the end.  c<=d is only guaranteed after at least
   one iteration of the main loop.  There's also a decent chance one % is
   faster than a binvert_limb, though that will depend on the processor.

   A CPU specific implementation might want to omit the size==1 code or the
   high<divisor test.  mpn/x86/k6/mode1o.asm for instance finds neither
   useful.  */


mp_limb_t
mpn_modexact_1c_odd (mp_srcptr src, mp_size_t size, mp_limb_t d,
                     mp_limb_t orig_c)
{
  mp_limb_t  s, h, l, inverse, dummy, dmul, ret;
  mp_limb_t  c = orig_c;
  mp_size_t  i;

  ASSERT (size >= 1);
  ASSERT (d & 1);
  ASSERT_MPN (src, size);
  ASSERT_LIMB (d);
  ASSERT_LIMB (c);

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
  dmul = d << GMP_NAIL_BITS;

  i = 0;
  do
    {
      s = src[i];
      SUBC_LIMB (c, l, s, c);
      l = (l * inverse) & GMP_NUMB_MASK;
      umul_ppmm (h, dummy, l, dmul);
      c += h;
    }
  while (++i < size-1);


  s = src[i];
  if (s <= d)
    {
      /* With high<=d the final step can be a subtract and addback.  If c==0
	 then the addback will restore to l>=0.  If c==d then will get l==d
	 if s==0, but that's ok per the function definition.  */

      l = c - s;
      if (c < s)
	l += d;

      ret = l;
    }
  else
    {
      /* Can't skip a divide, just do the loop code once more. */

      SUBC_LIMB (c, l, s, c);
      l = (l * inverse) & GMP_NUMB_MASK;
      umul_ppmm (h, dummy, l, dmul);
      c += h;
      ret = c;
    }

  ASSERT (orig_c < d ? ret < d : ret <= d);
  return ret;
}



#if 0

/* The following is an alternate form that might shave one cycle on a
   superscalar processor since it takes c+=h off the dependent chain,
   leaving just a low product, high product, and a subtract.

   This is for CPU specific implementations to consider.  A special case for
   high<divisor and/or size==1 can be added if desired.

   Notice that c is only ever 0 or 1, since if s-c produces a borrow then
   x=0xFF..FF and x-h cannot produce a borrow.  The c=(x>s) could become
   c=(x==0xFF..FF) too, if that helped.  */

mp_limb_t
mpn_modexact_1c_odd (mp_srcptr src, mp_size_t size, mp_limb_t d, mp_limb_t h)
{
  mp_limb_t  s, x, y, inverse, dummy, dmul, c1, c2;
  mp_limb_t  c = 0;
  mp_size_t  i;

  ASSERT (size >= 1);
  ASSERT (d & 1);

  binvert_limb (inverse, d);
  dmul = d << GMP_NAIL_BITS;

  for (i = 0; i < size; i++)
    {
      ASSERT (c==0 || c==1);

      s = src[i];
      SUBC_LIMB (c1, x, s, c);

      SUBC_LIMB (c2, y, x, h);
      c = c1 + c2;

      y = (y * inverse) & GMP_NUMB_MASK;
      umul_ppmm (h, dummy, y, dmul);
    }

  h += c;
  return h;
}

#endif
