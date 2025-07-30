/* mpz_lucnum_ui -- calculate Lucas number.

Copyright 2001, 2003, 2005, 2011, 2012 Free Software Foundation, Inc.

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


/* Notes:

   For the +4 in L[2k+1] when k is even, all L[4m+3] == 4, 5 or 7 mod 8, so
   there can't be an overflow applying +4 to just the low limb (since that
   would leave 0, 1, 2 or 3 mod 8).

   For the -4 in L[2k+1] when k is even, it seems (no proof) that
   L[3*2^(b-2)-3] == -4 mod 2^b, so for instance with a 32-bit limb
   L[0xBFFFFFFD] == 0xFFFFFFFC mod 2^32, and this implies a borrow from the
   low limb.  Obviously L[0xBFFFFFFD] is a huge number, but it's at least
   conceivable to calculate it, so it probably should be handled.

   For the -2 in L[2k] with k even, it seems (no proof) L[2^(b-1)] == -1 mod
   2^b, so for instance in 32-bits L[0x80000000] has a low limb of
   0xFFFFFFFF so there would have been a borrow.  Again L[0x80000000] is
   obviously huge, but probably should be made to work.  */

void
mpz_lucnum_ui (mpz_ptr ln, unsigned long n)
{
  mp_size_t  lalloc, xalloc, lsize, xsize;
  mp_ptr     lp, xp;
  mp_limb_t  c;
  int        zeros;
  TMP_DECL;

  TRACE (printf ("mpn_lucnum_ui n=%lu\n", n));

  if (n <= FIB_TABLE_LUCNUM_LIMIT)
    {
      /* L[n] = F[n] + 2F[n-1] */
      PTR(ln)[0] = FIB_TABLE(n) + 2 * FIB_TABLE ((int) n - 1);
      SIZ(ln) = 1;
      return;
    }

  /* +1 since L[n]=F[n]+2F[n-1] might be 1 limb bigger than F[n], further +1
     since square or mul used below might need an extra limb over the true
     size */
  lalloc = MPN_FIB2_SIZE (n) + 2;
  lp = MPZ_REALLOC (ln, lalloc);

  TMP_MARK;
  xalloc = lalloc;
  xp = TMP_ALLOC_LIMBS (xalloc);

  /* Strip trailing zeros from n, until either an odd number is reached
     where the L[2k+1] formula can be used, or until n fits within the
     FIB_TABLE data.  The table is preferred of course.  */
  zeros = 0;
  for (;;)
    {
      if (n & 1)
	{
	  /* L[2k+1] = 5*F[k-1]*(2*F[k]+F[k-1]) - 4*(-1)^k */

	  mp_size_t  yalloc, ysize;
	  mp_ptr     yp;

	  TRACE (printf ("  initial odd n=%lu\n", n));

	  yalloc = MPN_FIB2_SIZE (n/2);
	  yp = TMP_ALLOC_LIMBS (yalloc);
	  ASSERT (xalloc >= yalloc);

	  xsize = mpn_fib2_ui (xp, yp, n/2);

	  /* possible high zero on F[k-1] */
	  ysize = xsize;
	  ysize -= (yp[ysize-1] == 0);
	  ASSERT (yp[ysize-1] != 0);

	  /* xp = 2*F[k] + F[k-1] */
#if HAVE_NATIVE_mpn_addlsh1_n
	  c = mpn_addlsh1_n (xp, yp, xp, xsize);
#else
	  c = mpn_lshift (xp, xp, xsize, 1);
	  c += mpn_add_n (xp, xp, yp, xsize);
#endif
	  ASSERT (xalloc >= xsize+1);
	  xp[xsize] = c;
	  xsize += (c != 0);
	  ASSERT (xp[xsize-1] != 0);

	  ASSERT (lalloc >= xsize + ysize);
	  c = mpn_mul (lp, xp, xsize, yp, ysize);
	  lsize = xsize + ysize;
	  lsize -= (c == 0);

	  /* lp = 5*lp */
#if HAVE_NATIVE_mpn_addlsh2_n
	  c = mpn_addlsh2_n (lp, lp, lp, lsize);
#else
	  /* FIXME: Is this faster than mpn_mul_1 ? */
	  c = mpn_lshift (xp, lp, lsize, 2);
	  c += mpn_add_n (lp, lp, xp, lsize);
#endif
	  ASSERT (lalloc >= lsize+1);
	  lp[lsize] = c;
	  lsize += (c != 0);

	  /* lp = lp - 4*(-1)^k */
	  if (n & 2)
	    {
	      /* no overflow, see comments above */
	      ASSERT (lp[0] <= MP_LIMB_T_MAX-4);
	      lp[0] += 4;
	    }
	  else
	    {
	      /* won't go negative */
	      MPN_DECR_U (lp, lsize, CNST_LIMB(4));
	    }

	  TRACE (mpn_trace ("  l",lp, lsize));
	  break;
	}

      MP_PTR_SWAP (xp, lp); /* balance the swaps wanted in the L[2k] below */
      zeros++;
      n /= 2;

      if (n <= FIB_TABLE_LUCNUM_LIMIT)
	{
	  /* L[n] = F[n] + 2F[n-1] */
	  lp[0] = FIB_TABLE (n) + 2 * FIB_TABLE ((int) n - 1);
	  lsize = 1;

	  TRACE (printf ("  initial small n=%lu\n", n);
		 mpn_trace ("  l",lp, lsize));
	  break;
	}
    }

  for ( ; zeros != 0; zeros--)
    {
      /* L[2k] = L[k]^2 + 2*(-1)^k */

      TRACE (printf ("  zeros=%d\n", zeros));

      ASSERT (xalloc >= 2*lsize);
      mpn_sqr (xp, lp, lsize);
      lsize *= 2;
      lsize -= (xp[lsize-1] == 0);

      /* First time around the loop k==n determines (-1)^k, after that k is
	 always even and we set n=0 to indicate that.  */
      if (n & 1)
	{
	  /* L[n]^2 == 0 or 1 mod 4, like all squares, so +2 gives no carry */
	  ASSERT (xp[0] <= MP_LIMB_T_MAX-2);
	  xp[0] += 2;
	  n = 0;
	}
      else
	{
	  /* won't go negative */
	  MPN_DECR_U (xp, lsize, CNST_LIMB(2));
	}

      MP_PTR_SWAP (xp, lp);
      ASSERT (lp[lsize-1] != 0);
    }

  /* should end up in the right spot after all the xp/lp swaps */
  ASSERT (lp == PTR(ln));
  SIZ(ln) = lsize;

  TMP_FREE;
}
