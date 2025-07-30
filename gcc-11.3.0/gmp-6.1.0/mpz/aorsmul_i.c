/* mpz_addmul_ui, mpz_submul_ui - add or subtract small multiple.

   THE mpz_aorsmul_1 FUNCTION IN THIS FILE IS FOR INTERNAL USE ONLY AND IS
   ALMOST CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR
   COMPLETELY IN FUTURE GNU MP RELEASES.

Copyright 2001, 2002, 2004, 2005, 2012 Free Software Foundation, Inc.

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


#if HAVE_NATIVE_mpn_mul_1c
#define MPN_MUL_1C(cout, dst, src, size, n, cin)        \
  do {                                                  \
    (cout) = mpn_mul_1c (dst, src, size, n, cin);       \
  } while (0)
#else
#define MPN_MUL_1C(cout, dst, src, size, n, cin)        \
  do {                                                  \
    mp_limb_t __cy;                                     \
    __cy = mpn_mul_1 (dst, src, size, n);               \
    (cout) = __cy + mpn_add_1 (dst, dst, size, cin);    \
  } while (0)
#endif


/* sub>=0 means an addmul w += x*y, sub<0 means a submul w -= x*y.

   All that's needed to account for negative w or x is to flip "sub".

   The final w will retain its sign, unless an underflow occurs in a submul
   of absolute values, in which case it's flipped.

   If x has more limbs than w, then mpn_submul_1 followed by mpn_com is
   used.  The alternative would be mpn_mul_1 into temporary space followed
   by mpn_sub_n.  Avoiding temporary space seem good, and submul+com stands
   a chance of being faster since it involves only one set of carry
   propagations, not two.  Note that doing an addmul_1 with a
   twos-complement negative y doesn't work, because it effectively adds an
   extra x * 2^GMP_LIMB_BITS.  */

REGPARM_ATTR(1) void
mpz_aorsmul_1 (mpz_ptr w, mpz_srcptr x, mp_limb_t y, mp_size_t sub)
{
  mp_size_t  xsize, wsize, wsize_signed, new_wsize, min_size, dsize;
  mp_srcptr  xp;
  mp_ptr     wp;
  mp_limb_t  cy;

  /* w unaffected if x==0 or y==0 */
  xsize = SIZ (x);
  if (xsize == 0 || y == 0)
    return;

  sub ^= xsize;
  xsize = ABS (xsize);

  wsize_signed = SIZ (w);
  if (wsize_signed == 0)
    {
      /* nothing to add to, just set x*y, "sub" gives the sign */
      wp = MPZ_REALLOC (w, xsize+1);
      cy = mpn_mul_1 (wp, PTR(x), xsize, y);
      wp[xsize] = cy;
      xsize += (cy != 0);
      SIZ (w) = (sub >= 0 ? xsize : -xsize);
      return;
    }

  sub ^= wsize_signed;
  wsize = ABS (wsize_signed);

  new_wsize = MAX (wsize, xsize);
  wp = MPZ_REALLOC (w, new_wsize+1);
  xp = PTR (x);
  min_size = MIN (wsize, xsize);

  if (sub >= 0)
    {
      /* addmul of absolute values */

      cy = mpn_addmul_1 (wp, xp, min_size, y);
      wp += min_size;
      xp += min_size;

      dsize = xsize - wsize;
#if HAVE_NATIVE_mpn_mul_1c
      if (dsize > 0)
	cy = mpn_mul_1c (wp, xp, dsize, y, cy);
      else if (dsize < 0)
	{
	  dsize = -dsize;
	  cy = mpn_add_1 (wp, wp, dsize, cy);
	}
#else
      if (dsize != 0)
	{
	  mp_limb_t  cy2;
	  if (dsize > 0)
	    cy2 = mpn_mul_1 (wp, xp, dsize, y);
	  else
	    {
	      dsize = -dsize;
	      cy2 = 0;
	    }
	  cy = cy2 + mpn_add_1 (wp, wp, dsize, cy);
	}
#endif

      wp[dsize] = cy;
      new_wsize += (cy != 0);
    }
  else
    {
      /* submul of absolute values */

      cy = mpn_submul_1 (wp, xp, min_size, y);
      if (wsize >= xsize)
	{
	  /* if w bigger than x, then propagate borrow through it */
	  if (wsize != xsize)
	    cy = mpn_sub_1 (wp+xsize, wp+xsize, wsize-xsize, cy);

	  if (cy != 0)
	    {
	      /* Borrow out of w, take twos complement negative to get
		 absolute value, flip sign of w.  */
	      wp[new_wsize] = ~-cy;  /* extra limb is 0-cy */
	      mpn_com (wp, wp, new_wsize);
	      new_wsize++;
	      MPN_INCR_U (wp, new_wsize, CNST_LIMB(1));
	      wsize_signed = -wsize_signed;
	    }
	}
      else /* wsize < xsize */
	{
	  /* x bigger than w, so want x*y-w.  Submul has given w-x*y, so
	     take twos complement and use an mpn_mul_1 for the rest.  */

	  mp_limb_t  cy2;

	  /* -(-cy*b^n + w-x*y) = (cy-1)*b^n + ~(w-x*y) + 1 */
	  mpn_com (wp, wp, wsize);
	  cy += mpn_add_1 (wp, wp, wsize, CNST_LIMB(1));
	  cy -= 1;

	  /* If cy-1 == -1 then hold that -1 for latter.  mpn_submul_1 never
	     returns cy==MP_LIMB_T_MAX so that value always indicates a -1. */
	  cy2 = (cy == MP_LIMB_T_MAX);
	  cy += cy2;
	  MPN_MUL_1C (cy, wp+wsize, xp+wsize, xsize-wsize, y, cy);
	  wp[new_wsize] = cy;
	  new_wsize += (cy != 0);

	  /* Apply any -1 from above.  The value at wp+wsize is non-zero
	     because y!=0 and the high limb of x will be non-zero.  */
	  if (cy2)
	    MPN_DECR_U (wp+wsize, new_wsize-wsize, CNST_LIMB(1));

	  wsize_signed = -wsize_signed;
	}

      /* submul can produce high zero limbs due to cancellation, both when w
	 has more limbs or x has more  */
      MPN_NORMALIZE (wp, new_wsize);
    }

  SIZ (w) = (wsize_signed >= 0 ? new_wsize : -new_wsize);

  ASSERT (new_wsize == 0 || PTR(w)[new_wsize-1] != 0);
}


void
mpz_addmul_ui (mpz_ptr w, mpz_srcptr x, unsigned long y)
{
#if BITS_PER_ULONG > GMP_NUMB_BITS
  if (UNLIKELY (y > GMP_NUMB_MAX))
    {
      mpz_t t;
      mp_ptr tp;
      mp_size_t xn;
      TMP_DECL;
      TMP_MARK;
      xn = SIZ (x);
      if (xn == 0) return;
      MPZ_TMP_INIT (t, ABS (xn) + 1);
      tp = PTR (t);
      tp[0] = 0;
      MPN_COPY (tp + 1, PTR(x), ABS (xn));
      SIZ(t) = xn >= 0 ? xn + 1 : xn - 1;
      mpz_aorsmul_1 (w, t, (mp_limb_t) y >> GMP_NUMB_BITS, (mp_size_t) 0);
      PTR(t) = tp + 1;
      SIZ(t) = xn;
      mpz_aorsmul_1 (w, t, (mp_limb_t) y & GMP_NUMB_MASK, (mp_size_t) 0);
      TMP_FREE;
      return;
    }
#endif
  mpz_aorsmul_1 (w, x, (mp_limb_t) y, (mp_size_t) 0);
}

void
mpz_submul_ui (mpz_ptr w, mpz_srcptr x, unsigned long y)
{
#if BITS_PER_ULONG > GMP_NUMB_BITS
  if (y > GMP_NUMB_MAX)
    {
      mpz_t t;
      mp_ptr tp;
      mp_size_t xn;
      TMP_DECL;
      TMP_MARK;
      xn = SIZ (x);
      if (xn == 0) return;
      MPZ_TMP_INIT (t, ABS (xn) + 1);
      tp = PTR (t);
      tp[0] = 0;
      MPN_COPY (tp + 1, PTR(x), ABS (xn));
      SIZ(t) = xn >= 0 ? xn + 1 : xn - 1;
      mpz_aorsmul_1 (w, t, (mp_limb_t) y >> GMP_NUMB_BITS, (mp_size_t) -1);
      PTR(t) = tp + 1;
      SIZ(t) = xn;
      mpz_aorsmul_1 (w, t, (mp_limb_t) y & GMP_NUMB_MASK, (mp_size_t) -1);
      TMP_FREE;
      return;
    }
#endif
  mpz_aorsmul_1 (w, x, (mp_limb_t) y & GMP_NUMB_MASK, (mp_size_t) -1);
}
