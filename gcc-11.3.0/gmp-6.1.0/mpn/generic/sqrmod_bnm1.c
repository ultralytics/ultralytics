/* sqrmod_bnm1.c -- squaring mod B^n-1.

   Contributed to the GNU project by Niels Möller, Torbjorn Granlund and
   Marco Bodrato.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2009, 2010, 2012 Free Software Foundation, Inc.

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

/* Input is {ap,rn}; output is {rp,rn}, computation is
   mod B^rn - 1, and values are semi-normalised; zero is represented
   as either 0 or B^n - 1.  Needs a scratch of 2rn limbs at tp.
   tp==rp is allowed. */
static void
mpn_bc_sqrmod_bnm1 (mp_ptr rp, mp_srcptr ap, mp_size_t rn, mp_ptr tp)
{
  mp_limb_t cy;

  ASSERT (0 < rn);

  mpn_sqr (tp, ap, rn);
  cy = mpn_add_n (rp, tp, tp + rn, rn);
  /* If cy == 1, then the value of rp is at most B^rn - 2, so there can
   * be no overflow when adding in the carry. */
  MPN_INCR_U (rp, rn, cy);
}


/* Input is {ap,rn+1}; output is {rp,rn+1}, in
   semi-normalised representation, computation is mod B^rn + 1. Needs
   a scratch area of 2rn + 2 limbs at tp; tp == rp is allowed.
   Output is normalised. */
static void
mpn_bc_sqrmod_bnp1 (mp_ptr rp, mp_srcptr ap, mp_size_t rn, mp_ptr tp)
{
  mp_limb_t cy;

  ASSERT (0 < rn);

  mpn_sqr (tp, ap, rn + 1);
  ASSERT (tp[2*rn+1] == 0);
  ASSERT (tp[2*rn] < GMP_NUMB_MAX);
  cy = tp[2*rn] + mpn_sub_n (rp, tp, tp+rn, rn);
  rp[rn] = 0;
  MPN_INCR_U (rp, rn+1, cy );
}


/* Computes {rp,MIN(rn,2an)} <- {ap,an}^2 Mod(B^rn-1)
 *
 * The result is expected to be ZERO if and only if the operand
 * already is. Otherwise the class [0] Mod(B^rn-1) is represented by
 * B^rn-1.
 * It should not be a problem if sqrmod_bnm1 is used to
 * compute the full square with an <= 2*rn, because this condition
 * implies (B^an-1)^2 < (B^rn-1) .
 *
 * Requires rn/4 < an <= rn
 * Scratch need: rn/2 + (need for recursive call OR rn + 3). This gives
 *
 * S(n) <= rn/2 + MAX (rn + 4, S(n/2)) <= 3/2 rn + 4
 */
void
mpn_sqrmod_bnm1 (mp_ptr rp, mp_size_t rn, mp_srcptr ap, mp_size_t an, mp_ptr tp)
{
  ASSERT (0 < an);
  ASSERT (an <= rn);

  if ((rn & 1) != 0 || BELOW_THRESHOLD (rn, SQRMOD_BNM1_THRESHOLD))
    {
      if (UNLIKELY (an < rn))
	{
	  if (UNLIKELY (2*an <= rn))
	    {
	      mpn_sqr (rp, ap, an);
	    }
	  else
	    {
	      mp_limb_t cy;
	      mpn_sqr (tp, ap, an);
	      cy = mpn_add (rp, tp, rn, tp + rn, 2*an - rn);
	      MPN_INCR_U (rp, rn, cy);
	    }
	}
      else
	mpn_bc_sqrmod_bnm1 (rp, ap, rn, tp);
    }
  else
    {
      mp_size_t n;
      mp_limb_t cy;
      mp_limb_t hi;

      n = rn >> 1;

      ASSERT (2*an > n);

      /* Compute xm = a^2 mod (B^n - 1), xp = a^2 mod (B^n + 1)
	 and crt together as

	 x = -xp * B^n + (B^n + 1) * [ (xp + xm)/2 mod (B^n-1)]
      */

#define a0 ap
#define a1 (ap + n)

#define xp  tp	/* 2n + 2 */
      /* am1  maybe in {xp, n} */
#define sp1 (tp + 2*n + 2)
      /* ap1  maybe in {sp1, n + 1} */

      {
	mp_srcptr am1;
	mp_size_t anm;
	mp_ptr so;

	if (LIKELY (an > n))
	  {
	    so = xp + n;
	    am1 = xp;
	    cy = mpn_add (xp, a0, n, a1, an - n);
	    MPN_INCR_U (xp, n, cy);
	    anm = n;
	  }
	else
	  {
	    so = xp;
	    am1 = a0;
	    anm = an;
	  }

	mpn_sqrmod_bnm1 (rp, n, am1, anm, so);
      }

      {
	int       k;
	mp_srcptr ap1;
	mp_size_t anp;

	if (LIKELY (an > n)) {
	  ap1 = sp1;
	  cy = mpn_sub (sp1, a0, n, a1, an - n);
	  sp1[n] = 0;
	  MPN_INCR_U (sp1, n + 1, cy);
	  anp = n + ap1[n];
	} else {
	  ap1 = a0;
	  anp = an;
	}

	if (BELOW_THRESHOLD (n, MUL_FFT_MODF_THRESHOLD))
	  k=0;
	else
	  {
	    int mask;
	    k = mpn_fft_best_k (n, 1);
	    mask = (1<<k) -1;
	    while (n & mask) {k--; mask >>=1;};
	  }
	if (k >= FFT_FIRST_K)
	  xp[n] = mpn_mul_fft (xp, n, ap1, anp, ap1, anp, k);
	else if (UNLIKELY (ap1 == a0))
	  {
	    ASSERT (anp <= n);
	    ASSERT (2*anp > n);
	    mpn_sqr (xp, a0, an);
	    anp = 2*an - n;
	    cy = mpn_sub (xp, xp, n, xp + n, anp);
	    xp[n] = 0;
	    MPN_INCR_U (xp, n+1, cy);
	  }
	else
	  mpn_bc_sqrmod_bnp1 (xp, ap1, n, xp);
      }

      /* Here the CRT recomposition begins.

	 xm <- (xp + xm)/2 = (xp + xm)B^n/2 mod (B^n-1)
	 Division by 2 is a bitwise rotation.

	 Assumes xp normalised mod (B^n+1).

	 The residue class [0] is represented by [B^n-1]; except when
	 both input are ZERO.
      */

#if HAVE_NATIVE_mpn_rsh1add_n || HAVE_NATIVE_mpn_rsh1add_nc
#if HAVE_NATIVE_mpn_rsh1add_nc
      cy = mpn_rsh1add_nc(rp, rp, xp, n, xp[n]); /* B^n = 1 */
      hi = cy << (GMP_NUMB_BITS - 1);
      cy = 0;
      /* next update of rp[n-1] will set cy = 1 only if rp[n-1]+=hi
	 overflows, i.e. a further increment will not overflow again. */
#else /* ! _nc */
      cy = xp[n] + mpn_rsh1add_n(rp, rp, xp, n); /* B^n = 1 */
      hi = (cy<<(GMP_NUMB_BITS-1))&GMP_NUMB_MASK; /* (cy&1) << ... */
      cy >>= 1;
      /* cy = 1 only if xp[n] = 1 i.e. {xp,n} = ZERO, this implies that
	 the rsh1add was a simple rshift: the top bit is 0. cy=1 => hi=0. */
#endif
#if GMP_NAIL_BITS == 0
      add_ssaaaa(cy, rp[n-1], cy, rp[n-1], CNST_LIMB(0), hi);
#else
      cy += (hi & rp[n-1]) >> (GMP_NUMB_BITS-1);
      rp[n-1] ^= hi;
#endif
#else /* ! HAVE_NATIVE_mpn_rsh1add_n */
#if HAVE_NATIVE_mpn_add_nc
      cy = mpn_add_nc(rp, rp, xp, n, xp[n]);
#else /* ! _nc */
      cy = xp[n] + mpn_add_n(rp, rp, xp, n); /* xp[n] == 1 implies {xp,n} == ZERO */
#endif
      cy += (rp[0]&1);
      mpn_rshift(rp, rp, n, 1);
      ASSERT (cy <= 2);
      hi = (cy<<(GMP_NUMB_BITS-1))&GMP_NUMB_MASK; /* (cy&1) << ... */
      cy >>= 1;
      /* We can have cy != 0 only if hi = 0... */
      ASSERT ((rp[n-1] & GMP_NUMB_HIGHBIT) == 0);
      rp[n-1] |= hi;
      /* ... rp[n-1] + cy can not overflow, the following INCR is correct. */
#endif
      ASSERT (cy <= 1);
      /* Next increment can not overflow, read the previous comments about cy. */
      ASSERT ((cy == 0) || ((rp[n-1] & GMP_NUMB_HIGHBIT) == 0));
      MPN_INCR_U(rp, n, cy);

      /* Compute the highest half:
	 ([(xp + xm)/2 mod (B^n-1)] - xp ) * B^n
       */
      if (UNLIKELY (2*an < rn))
	{
	  /* Note that in this case, the only way the result can equal
	     zero mod B^{rn} - 1 is if the input is zero, and
	     then the output of both the recursive calls and this CRT
	     reconstruction is zero, not B^{rn} - 1. */
	  cy = mpn_sub_n (rp + n, rp, xp, 2*an - n);

	  /* FIXME: This subtraction of the high parts is not really
	     necessary, we do it to get the carry out, and for sanity
	     checking. */
	  cy = xp[n] + mpn_sub_nc (xp + 2*an - n, rp + 2*an - n,
				   xp + 2*an - n, rn - 2*an, cy);
	  ASSERT (mpn_zero_p (xp + 2*an - n+1, rn - 1 - 2*an));
	  cy = mpn_sub_1 (rp, rp, 2*an, cy);
	  ASSERT (cy == (xp + 2*an - n)[0]);
	}
      else
	{
	  cy = xp[n] + mpn_sub_n (rp + n, rp, xp, n);
	  /* cy = 1 only if {xp,n+1} is not ZERO, i.e. {rp,n} is not ZERO.
	     DECR will affect _at most_ the lowest n limbs. */
	  MPN_DECR_U (rp, 2*n, cy);
	}
#undef a0
#undef a1
#undef xp
#undef sp1
    }
}

mp_size_t
mpn_sqrmod_bnm1_next_size (mp_size_t n)
{
  mp_size_t nh;

  if (BELOW_THRESHOLD (n,     SQRMOD_BNM1_THRESHOLD))
    return n;
  if (BELOW_THRESHOLD (n, 4 * (SQRMOD_BNM1_THRESHOLD - 1) + 1))
    return (n + (2-1)) & (-2);
  if (BELOW_THRESHOLD (n, 8 * (SQRMOD_BNM1_THRESHOLD - 1) + 1))
    return (n + (4-1)) & (-4);

  nh = (n + 1) >> 1;

  if (BELOW_THRESHOLD (nh, SQR_FFT_MODF_THRESHOLD))
    return (n + (8-1)) & (-8);

  return 2 * mpn_fft_next_size (nh, mpn_fft_best_k (nh, 1));
}
