/* mpn_toom32_mul -- Multiply {ap,an} and {bp,bn} where an is nominally 1.5
   times as large as bn.  Or more accurately, bn < an < 3bn.

   Contributed to the GNU project by Torbjorn Granlund.
   Improvements by Marco Bodrato and Niels Möller.

   The idea of applying toom to unbalanced multiplication is due to Marco
   Bodrato and Alberto Zanoni.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2006-2010 Free Software Foundation, Inc.

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

/* Evaluate in: -1, 0, +1, +inf

  <-s-><--n--><--n-->
   ___ ______ ______
  |a2_|___a1_|___a0_|
	|_b1_|___b0_|
	<-t--><--n-->

  v0  =  a0         * b0      #   A(0)*B(0)
  v1  = (a0+ a1+ a2)*(b0+ b1) #   A(1)*B(1)      ah  <= 2  bh <= 1
  vm1 = (a0- a1+ a2)*(b0- b1) #  A(-1)*B(-1)    |ah| <= 1  bh = 0
  vinf=          a2 *     b1  # A(inf)*B(inf)
*/

#define TOOM32_MUL_N_REC(p, a, b, n, ws)				\
  do {									\
    mpn_mul_n (p, a, b, n);						\
  } while (0)

void
mpn_toom32_mul (mp_ptr pp,
		mp_srcptr ap, mp_size_t an,
		mp_srcptr bp, mp_size_t bn,
		mp_ptr scratch)
{
  mp_size_t n, s, t;
  int vm1_neg;
  mp_limb_t cy;
  mp_limb_signed_t hi;
  mp_limb_t ap1_hi, bp1_hi;

#define a0  ap
#define a1  (ap + n)
#define a2  (ap + 2 * n)
#define b0  bp
#define b1  (bp + n)

  /* Required, to ensure that s + t >= n. */
  ASSERT (bn + 2 <= an && an + 6 <= 3*bn);

  n = 1 + (2 * an >= 3 * bn ? (an - 1) / (size_t) 3 : (bn - 1) >> 1);

  s = an - 2 * n;
  t = bn - n;

  ASSERT (0 < s && s <= n);
  ASSERT (0 < t && t <= n);
  ASSERT (s + t >= n);

  /* Product area of size an + bn = 3*n + s + t >= 4*n + 2. */
#define ap1 (pp)		/* n, most significant limb in ap1_hi */
#define bp1 (pp + n)		/* n, most significant bit in bp1_hi */
#define am1 (pp + 2*n)		/* n, most significant bit in hi */
#define bm1 (pp + 3*n)		/* n */
#define v1 (scratch)		/* 2n + 1 */
#define vm1 (pp)		/* 2n + 1 */
#define scratch_out (scratch + 2*n + 1) /* Currently unused. */

  /* Scratch need: 2*n + 1 + scratch for the recursive multiplications. */

  /* FIXME: Keep v1[2*n] and vm1[2*n] in scalar variables? */

  /* Compute ap1 = a0 + a1 + a3, am1 = a0 - a1 + a3 */
  ap1_hi = mpn_add (ap1, a0, n, a2, s);
#if HAVE_NATIVE_mpn_add_n_sub_n
  if (ap1_hi == 0 && mpn_cmp (ap1, a1, n) < 0)
    {
      ap1_hi = mpn_add_n_sub_n (ap1, am1, a1, ap1, n) >> 1;
      hi = 0;
      vm1_neg = 1;
    }
  else
    {
      cy = mpn_add_n_sub_n (ap1, am1, ap1, a1, n);
      hi = ap1_hi - (cy & 1);
      ap1_hi += (cy >> 1);
      vm1_neg = 0;
    }
#else
  if (ap1_hi == 0 && mpn_cmp (ap1, a1, n) < 0)
    {
      ASSERT_NOCARRY (mpn_sub_n (am1, a1, ap1, n));
      hi = 0;
      vm1_neg = 1;
    }
  else
    {
      hi = ap1_hi - mpn_sub_n (am1, ap1, a1, n);
      vm1_neg = 0;
    }
  ap1_hi += mpn_add_n (ap1, ap1, a1, n);
#endif

  /* Compute bp1 = b0 + b1 and bm1 = b0 - b1. */
  if (t == n)
    {
#if HAVE_NATIVE_mpn_add_n_sub_n
      if (mpn_cmp (b0, b1, n) < 0)
	{
	  cy = mpn_add_n_sub_n (bp1, bm1, b1, b0, n);
	  vm1_neg ^= 1;
	}
      else
	{
	  cy = mpn_add_n_sub_n (bp1, bm1, b0, b1, n);
	}
      bp1_hi = cy >> 1;
#else
      bp1_hi = mpn_add_n (bp1, b0, b1, n);

      if (mpn_cmp (b0, b1, n) < 0)
	{
	  ASSERT_NOCARRY (mpn_sub_n (bm1, b1, b0, n));
	  vm1_neg ^= 1;
	}
      else
	{
	  ASSERT_NOCARRY (mpn_sub_n (bm1, b0, b1, n));
	}
#endif
    }
  else
    {
      /* FIXME: Should still use mpn_add_n_sub_n for the main part. */
      bp1_hi = mpn_add (bp1, b0, n, b1, t);

      if (mpn_zero_p (b0 + t, n - t) && mpn_cmp (b0, b1, t) < 0)
	{
	  ASSERT_NOCARRY (mpn_sub_n (bm1, b1, b0, t));
	  MPN_ZERO (bm1 + t, n - t);
	  vm1_neg ^= 1;
	}
      else
	{
	  ASSERT_NOCARRY (mpn_sub (bm1, b0, n, b1, t));
	}
    }

  TOOM32_MUL_N_REC (v1, ap1, bp1, n, scratch_out);
  if (ap1_hi == 1)
    {
      cy = bp1_hi + mpn_add_n (v1 + n, v1 + n, bp1, n);
    }
  else if (ap1_hi == 2)
    {
#if HAVE_NATIVE_mpn_addlsh1_n
      cy = 2 * bp1_hi + mpn_addlsh1_n (v1 + n, v1 + n, bp1, n);
#else
      cy = 2 * bp1_hi + mpn_addmul_1 (v1 + n, bp1, n, CNST_LIMB(2));
#endif
    }
  else
    cy = 0;
  if (bp1_hi != 0)
    cy += mpn_add_n (v1 + n, v1 + n, ap1, n);
  v1[2 * n] = cy;

  TOOM32_MUL_N_REC (vm1, am1, bm1, n, scratch_out);
  if (hi)
    hi = mpn_add_n (vm1+n, vm1+n, bm1, n);

  vm1[2*n] = hi;

  /* v1 <-- (v1 + vm1) / 2 = x0 + x2 */
  if (vm1_neg)
    {
#if HAVE_NATIVE_mpn_rsh1sub_n
      mpn_rsh1sub_n (v1, v1, vm1, 2*n+1);
#else
      mpn_sub_n (v1, v1, vm1, 2*n+1);
      ASSERT_NOCARRY (mpn_rshift (v1, v1, 2*n+1, 1));
#endif
    }
  else
    {
#if HAVE_NATIVE_mpn_rsh1add_n
      mpn_rsh1add_n (v1, v1, vm1, 2*n+1);
#else
      mpn_add_n (v1, v1, vm1, 2*n+1);
      ASSERT_NOCARRY (mpn_rshift (v1, v1, 2*n+1, 1));
#endif
    }

  /* We get x1 + x3 = (x0 + x2) - (x0 - x1 + x2 - x3), and hence

     y = x1 + x3 + (x0 + x2) * B
       = (x0 + x2) * B + (x0 + x2) - vm1.

     y is 3*n + 1 limbs, y = y0 + y1 B + y2 B^2. We store them as
     follows: y0 at scratch, y1 at pp + 2*n, and y2 at scratch + n
     (already in place, except for carry propagation).

     We thus add

   B^3  B^2   B    1
    |    |    |    |
   +-----+----+
 + |  x0 + x2 |
   +----+-----+----+
 +      |  x0 + x2 |
	+----------+
 -      |  vm1     |
 --+----++----+----+-
   | y2  | y1 | y0 |
   +-----+----+----+

  Since we store y0 at the same location as the low half of x0 + x2, we
  need to do the middle sum first. */

  hi = vm1[2*n];
  cy = mpn_add_n (pp + 2*n, v1, v1 + n, n);
  MPN_INCR_U (v1 + n, n + 1, cy + v1[2*n]);

  /* FIXME: Can we get rid of this second vm1_neg conditional by
     swapping the location of +1 and -1 values? */
  if (vm1_neg)
    {
      cy = mpn_add_n (v1, v1, vm1, n);
      hi += mpn_add_nc (pp + 2*n, pp + 2*n, vm1 + n, n, cy);
      MPN_INCR_U (v1 + n, n+1, hi);
    }
  else
    {
      cy = mpn_sub_n (v1, v1, vm1, n);
      hi += mpn_sub_nc (pp + 2*n, pp + 2*n, vm1 + n, n, cy);
      MPN_DECR_U (v1 + n, n+1, hi);
    }

  TOOM32_MUL_N_REC (pp, a0, b0, n, scratch_out);
  /* vinf, s+t limbs.  Use mpn_mul for now, to handle unbalanced operands */
  if (s > t)  mpn_mul (pp+3*n, a2, s, b1, t);
  else        mpn_mul (pp+3*n, b1, t, a2, s);

  /* Remaining interpolation.

     y * B + x0 + x3 B^3 - x0 B^2 - x3 B
     = (x1 + x3) B + (x0 + x2) B^2 + x0 + x3 B^3 - x0 B^2 - x3 B
     = y0 B + y1 B^2 + y3 B^3 + Lx0 + H x0 B
       + L x3 B^3 + H x3 B^4 - Lx0 B^2 - H x0 B^3 - L x3 B - H x3 B^2
     = L x0 + (y0 + H x0 - L x3) B + (y1 - L x0 - H x3) B^2
       + (y2 - (H x0 - L x3)) B^3 + H x3 B^4

	  B^4       B^3       B^2        B         1
 |         |         |         |         |         |
   +-------+                   +---------+---------+
   |  Hx3  |                   | Hx0-Lx3 |    Lx0  |
   +------+----------+---------+---------+---------+
	  |    y2    |  y1     |   y0    |
	  ++---------+---------+---------+
	  -| Hx0-Lx3 | - Lx0   |
	   +---------+---------+
		      | - Hx3  |
		      +--------+

    We must take into account the carry from Hx0 - Lx3.
  */

  cy = mpn_sub_n (pp + n, pp + n, pp+3*n, n);
  hi = scratch[2*n] + cy;

  cy = mpn_sub_nc (pp + 2*n, pp + 2*n, pp, n, cy);
  hi -= mpn_sub_nc (pp + 3*n, scratch + n, pp + n, n, cy);

  hi += mpn_add (pp + n, pp + n, 3*n, scratch, n);

  /* FIXME: Is support for s + t == n needed? */
  if (LIKELY (s + t > n))
    {
      hi -= mpn_sub (pp + 2*n, pp + 2*n, 2*n, pp + 4*n, s+t-n);

      if (hi < 0)
	MPN_DECR_U (pp + 4*n, s+t-n, -hi);
      else
	MPN_INCR_U (pp + 4*n, s+t-n, hi);
    }
  else
    ASSERT (hi == 0);
}
