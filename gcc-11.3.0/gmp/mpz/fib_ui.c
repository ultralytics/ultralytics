/* mpz_fib_ui -- calculate Fibonacci numbers.

Copyright 2000-2002, 2005, 2012, 2014 Free Software Foundation, Inc.

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
#include "longlong.h"


/* change to "#define TRACE(x) x" to get some traces */
#define TRACE(x)


/* In the F[2k+1] below for k odd, the -2 won't give a borrow from the low
   limb because the result F[2k+1] is an F[4m+3] and such numbers are always
   == 1, 2 or 5 mod 8, whereas an underflow would leave 6 or 7.  (This is
   the same as in mpn_fib2_ui.)

   In the F[2k+1] for k even, the +2 won't give a carry out of the low limb
   in normal circumstances.  This is an F[4m+1] and we claim that F[3*2^b+1]
   == 1 mod 2^b is the first F[4m+1] congruent to 0 or 1 mod 2^b, and hence
   if n < 2^GMP_NUMB_BITS then F[n] cannot have a low limb of 0 or 1.  No
   proof for this claim, but it's been verified up to b==32 and has such a
   nice pattern it must be true :-).  Of interest is that F[3*2^b] == 0 mod
   2^(b+1) seems to hold too.

   When n >= 2^GMP_NUMB_BITS, which can arise in a nails build, then the low
   limb of F[4m+1] can certainly be 1, and an mpn_add_1 must be used.  */

void
mpz_fib_ui (mpz_ptr fn, unsigned long n)
{
  mp_ptr         fp, xp, yp;
  mp_size_t      size, xalloc;
  unsigned long  n2;
  mp_limb_t      c;
  TMP_DECL;

  if (n <= FIB_TABLE_LIMIT)
    {
      PTR(fn)[0] = FIB_TABLE (n);
      SIZ(fn) = (n != 0);      /* F[0]==0, others are !=0 */
      return;
    }

  n2 = n/2;
  xalloc = MPN_FIB2_SIZE (n2) + 1;
  fp = MPZ_NEWALLOC (fn, 2 * xalloc);

  TMP_MARK;
  TMP_ALLOC_LIMBS_2 (xp,xalloc, yp,xalloc);
  size = mpn_fib2_ui (xp, yp, n2);

  TRACE (printf ("mpz_fib_ui last step n=%lu size=%ld bit=%lu\n",
		 n >> 1, size, n&1);
	 mpn_trace ("xp", xp, size);
	 mpn_trace ("yp", yp, size));

  if (n & 1)
    {
      /* F[2k+1] = (2F[k]+F[k-1])*(2F[k]-F[k-1]) + 2*(-1)^k  */
      mp_size_t  xsize, ysize;

#if HAVE_NATIVE_mpn_add_n_sub_n
      xp[size] = mpn_lshift (xp, xp, size, 1);
      yp[size] = 0;
      ASSERT_NOCARRY (mpn_add_n_sub_n (xp, yp, xp, yp, size+1));
      xsize = size + (xp[size] != 0);
      ASSERT (yp[size] <= 1);
      ysize = size + yp[size];
#else
      mp_limb_t  c2;

      c2 = mpn_lshift (fp, xp, size, 1);
      c = c2 + mpn_add_n (xp, fp, yp, size);
      xp[size] = c;
      xsize = size + (c != 0);
      c2 -= mpn_sub_n (yp, fp, yp, size);
      yp[size] = c2;
      ASSERT (c2 <= 1);
      ysize = size + c2;
#endif

      size = xsize + ysize;
      c = mpn_mul (fp, xp, xsize, yp, ysize);

#if GMP_NUMB_BITS >= BITS_PER_ULONG
      /* no overflow, see comments above */
      ASSERT (n & 2 ? fp[0] >= 2 : fp[0] <= GMP_NUMB_MAX-2);
      fp[0] += (n & 2 ? -CNST_LIMB(2) : CNST_LIMB(2));
#else
      if (n & 2)
	{
	  ASSERT (fp[0] >= 2);
	  fp[0] -= 2;
	}
      else
	{
	  ASSERT (c != GMP_NUMB_MAX); /* because it's the high of a mul */
	  c += mpn_add_1 (fp, fp, size-1, CNST_LIMB(2));
	  fp[size-1] = c;
	}
#endif
    }
  else
    {
      /* F[2k] = F[k]*(F[k]+2F[k-1]) */

      mp_size_t  xsize, ysize;
#if HAVE_NATIVE_mpn_addlsh1_n
      c = mpn_addlsh1_n (yp, xp, yp, size);
#else
      c = mpn_lshift (yp, yp, size, 1);
      c += mpn_add_n (yp, yp, xp, size);
#endif
      yp[size] = c;
      xsize = size;
      ysize = size + (c != 0);
      size += ysize;
      c = mpn_mul (fp, yp, ysize, xp, xsize);
    }

  /* one or two high zeros */
  size -= (c == 0);
  size -= (fp[size-1] == 0);
  SIZ(fn) = size;

  TRACE (printf ("done special, size=%ld\n", size);
	 mpn_trace ("fp ", fp, size));

  TMP_FREE;
}
