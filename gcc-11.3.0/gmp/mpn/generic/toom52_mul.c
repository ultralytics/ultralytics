/* mpn_toom52_mul -- Multiply {ap,an} and {bp,bn} where an is nominally 4/3
   times as large as bn.  Or more accurately, bn < an < 2 bn.

   Contributed to the GNU project by Marco Bodrato.

   The idea of applying toom to unbalanced multiplication is due to Marco
   Bodrato and Alberto Zanoni.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2009 Free Software Foundation, Inc.

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

/* Evaluate in: -2, -1, 0, +1, +2, +inf

  <-s-><--n--><--n--><--n--><--n-->
   ___ ______ ______ ______ ______
  |a4_|___a3_|___a2_|___a1_|___a0_|
			|b1|___b0_|
			<t-><--n-->

  v0  =  a0                  * b0      #   A(0)*B(0)
  v1  = (a0+ a1+ a2+ a3+  a4)*(b0+ b1) #   A(1)*B(1)      ah  <= 4   bh <= 1
  vm1 = (a0- a1+ a2- a3+  a4)*(b0- b1) #  A(-1)*B(-1)    |ah| <= 2   bh  = 0
  v2  = (a0+2a1+4a2+8a3+16a4)*(b0+2b1) #   A(2)*B(2)      ah  <= 30  bh <= 2
  vm2 = (a0-2a1+4a2-8a3+16a4)*(b0-2b1) #  A(-2)*B(-2)    |ah| <= 20 |bh|<= 1
  vinf=                   a4 *     b1  # A(inf)*B(inf)

  Some slight optimization in evaluation are taken from the paper:
  "Towards Optimal Toom-Cook Multiplication for Univariate and
  Multivariate Polynomials in Characteristic 2 and 0."
*/

void
mpn_toom52_mul (mp_ptr pp,
		mp_srcptr ap, mp_size_t an,
		mp_srcptr bp, mp_size_t bn, mp_ptr scratch)
{
  mp_size_t n, s, t;
  enum toom6_flags flags;

#define a0  ap
#define a1  (ap + n)
#define a2  (ap + 2 * n)
#define a3  (ap + 3 * n)
#define a4  (ap + 4 * n)
#define b0  bp
#define b1  (bp + n)

  n = 1 + (2 * an >= 5 * bn ? (an - 1) / (size_t) 5 : (bn - 1) >> 1);

  s = an - 4 * n;
  t = bn - n;

  ASSERT (0 < s && s <= n);
  ASSERT (0 < t && t <= n);

  /* Ensures that 5 values of n+1 limbs each fits in the product area.
     Borderline cases are an = 32, bn = 8, n = 7, and an = 36, bn = 9,
     n = 8. */
  ASSERT (s+t >= 5);

#define v0    pp				/* 2n */
#define vm1   (scratch)				/* 2n+1 */
#define v1    (pp + 2 * n)			/* 2n+1 */
#define vm2   (scratch + 2 * n + 1)		/* 2n+1 */
#define v2    (scratch + 4 * n + 2)		/* 2n+1 */
#define vinf  (pp + 5 * n)			/* s+t */
#define bs1    pp				/* n+1 */
#define bsm1  (scratch + 2 * n + 2)		/* n   */
#define asm1  (scratch + 3 * n + 3)		/* n+1 */
#define asm2  (scratch + 4 * n + 4)		/* n+1 */
#define bsm2  (pp + n + 1)			/* n+1 */
#define bs2   (pp + 2 * n + 2)			/* n+1 */
#define as2   (pp + 3 * n + 3)			/* n+1 */
#define as1   (pp + 4 * n + 4)			/* n+1 */

  /* Scratch need is 6 * n + 3 + 1. We need one extra limb, because
     products will overwrite 2n+2 limbs. */

#define a0a2  scratch
#define a1a3  asm1

  /* Compute as2 and asm2.  */
  flags = (enum toom6_flags) (toom6_vm2_neg & mpn_toom_eval_pm2 (as2, asm2, 4, ap, n, s, a1a3));

  /* Compute bs1 and bsm1.  */
  if (t == n)
    {
#if HAVE_NATIVE_mpn_add_n_sub_n
      mp_limb_t cy;

      if (mpn_cmp (b0, b1, n) < 0)
	{
	  cy = mpn_add_n_sub_n (bs1, bsm1, b1, b0, n);
	  flags = (enum toom6_flags) (flags ^ toom6_vm1_neg);
	}
      else
	{
	  cy = mpn_add_n_sub_n (bs1, bsm1, b0, b1, n);
	}
      bs1[n] = cy >> 1;
#else
      bs1[n] = mpn_add_n (bs1, b0, b1, n);
      if (mpn_cmp (b0, b1, n) < 0)
	{
	  mpn_sub_n (bsm1, b1, b0, n);
	  flags = (enum toom6_flags) (flags ^ toom6_vm1_neg);
	}
      else
	{
	  mpn_sub_n (bsm1, b0, b1, n);
	}
#endif
    }
  else
    {
      bs1[n] = mpn_add (bs1, b0, n, b1, t);
      if (mpn_zero_p (b0 + t, n - t) && mpn_cmp (b0, b1, t) < 0)
	{
	  mpn_sub_n (bsm1, b1, b0, t);
	  MPN_ZERO (bsm1 + t, n - t);
	  flags = (enum toom6_flags) (flags ^ toom6_vm1_neg);
	}
      else
	{
	  mpn_sub (bsm1, b0, n, b1, t);
	}
    }

  /* Compute bs2 and bsm2, recycling bs1 and bsm1. bs2=bs1+b1; bsm2=bsm1-b1  */
  mpn_add (bs2, bs1, n+1, b1, t);
  if (flags & toom6_vm1_neg )
    {
      bsm2[n] = mpn_add (bsm2, bsm1, n, b1, t);
      flags = (enum toom6_flags) (flags ^ toom6_vm2_neg);
    }
  else
    {
      bsm2[n] = 0;
      if (t == n)
	{
	  if (mpn_cmp (bsm1, b1, n) < 0)
	    {
	      mpn_sub_n (bsm2, b1, bsm1, n);
	      flags = (enum toom6_flags) (flags ^ toom6_vm2_neg);
	    }
	  else
	    {
	      mpn_sub_n (bsm2, bsm1, b1, n);
	    }
	}
      else
	{
	  if (mpn_zero_p (bsm1 + t, n - t) && mpn_cmp (bsm1, b1, t) < 0)
	    {
	      mpn_sub_n (bsm2, b1, bsm1, t);
	      MPN_ZERO (bsm2 + t, n - t);
	      flags = (enum toom6_flags) (flags ^ toom6_vm2_neg);
	    }
	  else
	    {
	      mpn_sub (bsm2, bsm1, n, b1, t);
	    }
	}
    }

  /* Compute as1 and asm1.  */
  flags = (enum toom6_flags) (flags ^ (toom6_vm1_neg & mpn_toom_eval_pm1 (as1, asm1, 4, ap, n, s, a0a2)));

  ASSERT (as1[n] <= 4);
  ASSERT (bs1[n] <= 1);
  ASSERT (asm1[n] <= 2);
/*   ASSERT (bsm1[n] <= 1); */
  ASSERT (as2[n] <=30);
  ASSERT (bs2[n] <= 2);
  ASSERT (asm2[n] <= 20);
  ASSERT (bsm2[n] <= 1);

  /* vm1, 2n+1 limbs */
  mpn_mul (vm1, asm1, n+1, bsm1, n);  /* W4 */

  /* vm2, 2n+1 limbs */
  mpn_mul_n (vm2, asm2, bsm2, n+1);  /* W2 */

  /* v2, 2n+1 limbs */
  mpn_mul_n (v2, as2, bs2, n+1);  /* W1 */

  /* v1, 2n+1 limbs */
  mpn_mul_n (v1, as1, bs1, n+1);  /* W3 */

  /* vinf, s+t limbs */   /* W0 */
  if (s > t)  mpn_mul (vinf, a4, s, b1, t);
  else        mpn_mul (vinf, b1, t, a4, s);

  /* v0, 2n limbs */
  mpn_mul_n (v0, ap, bp, n);  /* W5 */

  mpn_toom_interpolate_6pts (pp, n, flags, vm1, vm2, v2, t + s);

#undef v0
#undef vm1
#undef v1
#undef vm2
#undef v2
#undef vinf
#undef bs1
#undef bs2
#undef bsm1
#undef bsm2
#undef asm1
#undef asm2
#undef as1
#undef as2
#undef a0a2
#undef b0b2
#undef a1a3
#undef a0
#undef a1
#undef a2
#undef a3
#undef b0
#undef b1
#undef b2

}
