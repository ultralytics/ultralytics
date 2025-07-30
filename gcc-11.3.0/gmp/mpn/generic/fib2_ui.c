/* mpn_fib2_ui -- calculate Fibonacci numbers.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2001, 2002, 2005, 2009 Free Software Foundation, Inc.

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

#include <stdio.h>
#include "gmp.h"
#include "gmp-impl.h"

/* change this to "#define TRACE(x) x" for diagnostics */
#define TRACE(x)


/* Store F[n] at fp and F[n-1] at f1p.  fp and f1p should have room for
   MPN_FIB2_SIZE(n) limbs.

   The return value is the actual number of limbs stored, this will be at
   least 1.  fp[size-1] will be non-zero, except when n==0, in which case
   fp[0] is 0 and f1p[0] is 1.  f1p[size-1] can be zero, since F[n-1]<F[n]
   (for n>0).

   Notes:

   In F[2k+1] with k even, +2 is applied to 4*F[k]^2 just by ORing into the
   low limb.

   In F[2k+1] with k odd, -2 is applied to the low limb of 4*F[k]^2 -
   F[k-1]^2.  This F[2k+1] is an F[4m+3] and such numbers are congruent to
   1, 2 or 5 mod 8, which means no underflow reaching it with a -2 (since
   that would leave 6 or 7 mod 8).

   This property of F[4m+3] can be verified by induction on F[4m+3] =
   7*F[4m-1] - F[4m-5], that formula being a standard lucas sequence
   identity U[i+j] = U[i]*V[j] - U[i-j]*Q^j.
*/

mp_size_t
mpn_fib2_ui (mp_ptr fp, mp_ptr f1p, unsigned long int n)
{
  mp_size_t      size;
  unsigned long  nfirst, mask;

  TRACE (printf ("mpn_fib2_ui n=%lu\n", n));

  ASSERT (! MPN_OVERLAP_P (fp, MPN_FIB2_SIZE(n), f1p, MPN_FIB2_SIZE(n)));

  /* Take a starting pair from the table. */
  mask = 1;
  for (nfirst = n; nfirst > FIB_TABLE_LIMIT; nfirst /= 2)
    mask <<= 1;
  TRACE (printf ("nfirst=%lu mask=0x%lX\n", nfirst, mask));

  f1p[0] = FIB_TABLE ((int) nfirst - 1);
  fp[0]  = FIB_TABLE (nfirst);
  size = 1;

  /* Skip to the end if the table lookup gives the final answer. */
  if (mask != 1)
    {
      mp_size_t  alloc;
      mp_ptr        xp;
      TMP_DECL;

      TMP_MARK;
      alloc = MPN_FIB2_SIZE (n);
      xp = TMP_ALLOC_LIMBS (alloc);

      do
	{
	  /* Here fp==F[k] and f1p==F[k-1], with k being the bits of n from
	     n&mask upwards.

	     The next bit of n is n&(mask>>1) and we'll double to the pair
	     fp==F[2k],f1p==F[2k-1] or fp==F[2k+1],f1p==F[2k], according as
	     that bit is 0 or 1 respectively.  */

	  TRACE (printf ("k=%lu mask=0x%lX size=%ld alloc=%ld\n",
			 n >> refmpn_count_trailing_zeros(mask),
			 mask, size, alloc);
		 mpn_trace ("fp ", fp, size);
		 mpn_trace ("f1p", f1p, size));

	  /* fp normalized, f1p at most one high zero */
	  ASSERT (fp[size-1] != 0);
	  ASSERT (f1p[size-1] != 0 || f1p[size-2] != 0);

	  /* f1p[size-1] might be zero, but this occurs rarely, so it's not
	     worth bothering checking for it */
	  ASSERT (alloc >= 2*size);
	  mpn_sqr (xp, fp,  size);
	  mpn_sqr (fp, f1p, size);
	  size *= 2;

	  /* Shrink if possible.  Since fp was normalized there'll be at
	     most one high zero on xp (and if there is then there's one on
	     yp too).  */
	  ASSERT (xp[size-1] != 0 || fp[size-1] == 0);
	  size -= (xp[size-1] == 0);
	  ASSERT (xp[size-1] != 0);  /* only one xp high zero */

	  /* Calculate F[2k-1] = F[k]^2 + F[k-1]^2. */
	  f1p[size] = mpn_add_n (f1p, xp, fp, size);

	  /* Calculate F[2k+1] = 4*F[k]^2 - F[k-1]^2 + 2*(-1)^k.
	     n&mask is the low bit of our implied k.  */
#if HAVE_NATIVE_mpn_rsblsh2_n
	  fp[size] = mpn_rsblsh2_n (fp, fp, xp, size);
	  if ((n & mask) == 0)
	    MPN_INCR_U(fp, size + 1, 2);	/* possible +2 */
	  else
	  {
	    ASSERT (fp[0] >= 2);
	    fp[0] -= 2;				/* possible -2 */
	  }
#else
	  {
	    mp_limb_t  c;

	    c = mpn_lshift (xp, xp, size, 2);
	    xp[0] |= (n & mask ? 0 : 2);	/* possible +2 */
	    c -= mpn_sub_n (fp, xp, fp, size);
	    ASSERT (n & mask ? fp[0] != 0 && fp[0] != 1 : 1);
	    fp[0] -= (n & mask ? 2 : 0);	/* possible -2 */
	    fp[size] = c;
	  }
#endif
	  ASSERT (alloc >= size+1);
	  size += (fp[size] != 0);

	  /* now n&mask is the new bit of n being considered */
	  mask >>= 1;

	  /* Calculate F[2k] = F[2k+1] - F[2k-1], replacing the unwanted one of
	     F[2k+1] and F[2k-1].  */
	  if (n & mask)
	    ASSERT_NOCARRY (mpn_sub_n (f1p, fp, f1p, size));
	  else {
	    ASSERT_NOCARRY (mpn_sub_n ( fp, fp, f1p, size));

	    /* Can have a high zero after replacing F[2k+1] with F[2k].
	       f1p will have a high zero if fp does. */
	    ASSERT (fp[size-1] != 0 || f1p[size-1] == 0);
	    size -= (fp[size-1] == 0);
	  }
	}
      while (mask != 1);

      TMP_FREE;
    }

  TRACE (printf ("done size=%ld\n", size);
	 mpn_trace ("fp ", fp, size);
	 mpn_trace ("f1p", f1p, size));

  return size;
}
