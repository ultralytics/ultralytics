/* mpn_toom62_mul -- Multiply {ap,an} and {bp,bn} where an is nominally 3 times
   as large as bn.  Or more accurately, (5/2)bn < an < 6bn.

   Contributed to the GNU project by Torbjorn Granlund and Marco Bodrato.

   The idea of applying toom to unbalanced multiplication is due to Marco
   Bodrato and Alberto Zanoni.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2006-2008, 2012 Free Software Foundation, Inc.

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

/* Evaluate in:
   0, +1, -1, +2, -2, 1/2, +inf

  <-s-><--n--><--n--><--n--><--n--><--n-->
   ___ ______ ______ ______ ______ ______
  |a5_|___a4_|___a3_|___a2_|___a1_|___a0_|
			     |_b1_|___b0_|
			     <-t--><--n-->

  v0  =    a0                       *   b0      #    A(0)*B(0)
  v1  = (  a0+  a1+ a2+ a3+  a4+  a5)*( b0+ b1) #    A(1)*B(1)      ah  <= 5   bh <= 1
  vm1 = (  a0-  a1+ a2- a3+  a4-  a5)*( b0- b1) #   A(-1)*B(-1)    |ah| <= 2   bh  = 0
  v2  = (  a0+ 2a1+4a2+8a3+16a4+32a5)*( b0+2b1) #    A(2)*B(2)      ah  <= 62  bh <= 2
  vm2 = (  a0- 2a1+4a2-8a3+16a4-32a5)*( b0-2b1) #   A(-2)*B(-2)    -41<=ah<=20 -1<=bh<=0
  vh  = (32a0+16a1+8a2+4a3+ 2a4+  a5)*(2b0+ b1) #  A(1/2)*B(1/2)    ah  <= 62  bh <= 2
  vinf=                           a5 *      b1  #  A(inf)*B(inf)
*/

void
mpn_toom62_mul (mp_ptr pp,
		mp_srcptr ap, mp_size_t an,
		mp_srcptr bp, mp_size_t bn,
		mp_ptr scratch)
{
  mp_size_t n, s, t;
  mp_limb_t cy;
  mp_ptr as1, asm1, as2, asm2, ash;
  mp_ptr bs1, bsm1, bs2, bsm2, bsh;
  mp_ptr gp;
  enum toom7_flags aflags, bflags;
  TMP_DECL;

#define a0  ap
#define a1  (ap + n)
#define a2  (ap + 2*n)
#define a3  (ap + 3*n)
#define a4  (ap + 4*n)
#define a5  (ap + 5*n)
#define b0  bp
#define b1  (bp + n)

  n = 1 + (an >= 3 * bn ? (an - 1) / (size_t) 6 : (bn - 1) >> 1);

  s = an - 5 * n;
  t = bn - n;

  ASSERT (0 < s && s <= n);
  ASSERT (0 < t && t <= n);

  TMP_MARK;

  as1 = TMP_SALLOC_LIMBS (n + 1);
  asm1 = TMP_SALLOC_LIMBS (n + 1);
  as2 = TMP_SALLOC_LIMBS (n + 1);
  asm2 = TMP_SALLOC_LIMBS (n + 1);
  ash = TMP_SALLOC_LIMBS (n + 1);

  bs1 = TMP_SALLOC_LIMBS (n + 1);
  bsm1 = TMP_SALLOC_LIMBS (n);
  bs2 = TMP_SALLOC_LIMBS (n + 1);
  bsm2 = TMP_SALLOC_LIMBS (n + 1);
  bsh = TMP_SALLOC_LIMBS (n + 1);

  gp = pp;

  /* Compute as1 and asm1.  */
  aflags = (enum toom7_flags) (toom7_w3_neg & mpn_toom_eval_pm1 (as1, asm1, 5, ap, n, s, gp));

  /* Compute as2 and asm2. */
  aflags = (enum toom7_flags) (aflags | (toom7_w1_neg & mpn_toom_eval_pm2 (as2, asm2, 5, ap, n, s, gp)));

  /* Compute ash = 32 a0 + 16 a1 + 8 a2 + 4 a3 + 2 a4 + a5
     = 2*(2*(2*(2*(2*a0 + a1) + a2) + a3) + a4) + a5  */

#if HAVE_NATIVE_mpn_addlsh1_n
  cy = mpn_addlsh1_n (ash, a1, a0, n);
  cy = 2*cy + mpn_addlsh1_n (ash, a2, ash, n);
  cy = 2*cy + mpn_addlsh1_n (ash, a3, ash, n);
  cy = 2*cy + mpn_addlsh1_n (ash, a4, ash, n);
  if (s < n)
    {
      mp_limb_t cy2;
      cy2 = mpn_addlsh1_n (ash, a5, ash, s);
      ash[n] = 2*cy + mpn_lshift (ash + s, ash + s, n - s, 1);
      MPN_INCR_U (ash + s, n+1-s, cy2);
    }
  else
    ash[n] = 2*cy + mpn_addlsh1_n (ash, a5, ash, n);
#else
  cy = mpn_lshift (ash, a0, n, 1);
  cy += mpn_add_n (ash, ash, a1, n);
  cy = 2*cy + mpn_lshift (ash, ash, n, 1);
  cy += mpn_add_n (ash, ash, a2, n);
  cy = 2*cy + mpn_lshift (ash, ash, n, 1);
  cy += mpn_add_n (ash, ash, a3, n);
  cy = 2*cy + mpn_lshift (ash, ash, n, 1);
  cy += mpn_add_n (ash, ash, a4, n);
  cy = 2*cy + mpn_lshift (ash, ash, n, 1);
  ash[n] = cy + mpn_add (ash, ash, n, a5, s);
#endif

  /* Compute bs1 and bsm1.  */
  if (t == n)
    {
#if HAVE_NATIVE_mpn_add_n_sub_n
      if (mpn_cmp (b0, b1, n) < 0)
	{
	  cy = mpn_add_n_sub_n (bs1, bsm1, b1, b0, n);
	  bflags = toom7_w3_neg;
	}
      else
	{
	  cy = mpn_add_n_sub_n (bs1, bsm1, b0, b1, n);
	  bflags = (enum toom7_flags) 0;
	}
      bs1[n] = cy >> 1;
#else
      bs1[n] = mpn_add_n (bs1, b0, b1, n);
      if (mpn_cmp (b0, b1, n) < 0)
	{
	  mpn_sub_n (bsm1, b1, b0, n);
	  bflags = toom7_w3_neg;
	}
      else
	{
	  mpn_sub_n (bsm1, b0, b1, n);
	  bflags = (enum toom7_flags) 0;
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
	  bflags = toom7_w3_neg;
	}
      else
	{
	  mpn_sub (bsm1, b0, n, b1, t);
	  bflags = (enum toom7_flags) 0;
	}
    }

  /* Compute bs2 and bsm2. Recycling bs1 and bsm1; bs2=bs1+b1, bsm2 =
     bsm1 - b1 */
  mpn_add (bs2, bs1, n + 1, b1, t);
  if (bflags & toom7_w3_neg)
    {
      bsm2[n] = mpn_add (bsm2, bsm1, n, b1, t);
      bflags = (enum toom7_flags) (bflags | toom7_w1_neg);
    }
  else
    {
      /* FIXME: Simplify this logic? */
      if (t < n)
	{
	  if (mpn_zero_p (bsm1 + t, n - t) && mpn_cmp (bsm1, b1, t) < 0)
	    {
	      ASSERT_NOCARRY (mpn_sub_n (bsm2, b1, bsm1, t));
	      MPN_ZERO (bsm2 + t, n + 1 - t);
	      bflags = (enum toom7_flags) (bflags | toom7_w1_neg);
	    }
	  else
	    {
	      ASSERT_NOCARRY (mpn_sub (bsm2, bsm1, n, b1, t));
	      bsm2[n] = 0;
	    }
	}
      else
	{
	  if (mpn_cmp (bsm1, b1, n) < 0)
	    {
	      ASSERT_NOCARRY (mpn_sub_n (bsm2, b1, bsm1, n));
	      bflags = (enum toom7_flags) (bflags | toom7_w1_neg);
	    }
	  else
	    {
	      ASSERT_NOCARRY (mpn_sub_n (bsm2, bsm1, b1, n));
	    }
	  bsm2[n] = 0;
	}
    }

  /* Compute bsh, recycling bs1. bsh=bs1+b0;  */
  bsh[n] = bs1[n] + mpn_add_n (bsh, bs1, b0, n);

  ASSERT (as1[n] <= 5);
  ASSERT (bs1[n] <= 1);
  ASSERT (asm1[n] <= 2);
  ASSERT (as2[n] <= 62);
  ASSERT (bs2[n] <= 2);
  ASSERT (asm2[n] <= 41);
  ASSERT (bsm2[n] <= 1);
  ASSERT (ash[n] <= 62);
  ASSERT (bsh[n] <= 2);

#define v0    pp				/* 2n */
#define v1    (pp + 2 * n)			/* 2n+1 */
#define vinf  (pp + 6 * n)			/* s+t */
#define v2    scratch				/* 2n+1 */
#define vm2   (scratch + 2 * n + 1)		/* 2n+1 */
#define vh    (scratch + 4 * n + 2)		/* 2n+1 */
#define vm1   (scratch + 6 * n + 3)		/* 2n+1 */
#define scratch_out (scratch + 8 * n + 4)		/* 2n+1 */
  /* Total scratch need: 10*n+5 */

  /* Must be in allocation order, as they overwrite one limb beyond
   * 2n+1. */
  mpn_mul_n (v2, as2, bs2, n + 1);		/* v2, 2n+1 limbs */
  mpn_mul_n (vm2, asm2, bsm2, n + 1);		/* vm2, 2n+1 limbs */
  mpn_mul_n (vh, ash, bsh, n + 1);		/* vh, 2n+1 limbs */

  /* vm1, 2n+1 limbs */
  mpn_mul_n (vm1, asm1, bsm1, n);
  cy = 0;
  if (asm1[n] == 1)
    {
      cy = mpn_add_n (vm1 + n, vm1 + n, bsm1, n);
    }
  else if (asm1[n] == 2)
    {
#if HAVE_NATIVE_mpn_addlsh1_n
      cy = mpn_addlsh1_n (vm1 + n, vm1 + n, bsm1, n);
#else
      cy = mpn_addmul_1 (vm1 + n, bsm1, n, CNST_LIMB(2));
#endif
    }
  vm1[2 * n] = cy;

  /* v1, 2n+1 limbs */
  mpn_mul_n (v1, as1, bs1, n);
  if (as1[n] == 1)
    {
      cy = bs1[n] + mpn_add_n (v1 + n, v1 + n, bs1, n);
    }
  else if (as1[n] == 2)
    {
#if HAVE_NATIVE_mpn_addlsh1_n
      cy = 2 * bs1[n] + mpn_addlsh1_n (v1 + n, v1 + n, bs1, n);
#else
      cy = 2 * bs1[n] + mpn_addmul_1 (v1 + n, bs1, n, CNST_LIMB(2));
#endif
    }
  else if (as1[n] != 0)
    {
      cy = as1[n] * bs1[n] + mpn_addmul_1 (v1 + n, bs1, n, as1[n]);
    }
  else
    cy = 0;
  if (bs1[n] != 0)
    cy += mpn_add_n (v1 + n, v1 + n, as1, n);
  v1[2 * n] = cy;

  mpn_mul_n (v0, a0, b0, n);			/* v0, 2n limbs */

  /* vinf, s+t limbs */
  if (s > t)  mpn_mul (vinf, a5, s, b1, t);
  else        mpn_mul (vinf, b1, t, a5, s);

  mpn_toom_interpolate_7pts (pp, n, (enum toom7_flags) (aflags ^ bflags),
			     vm2, vm1, v2, vh, s + t, scratch_out);

  TMP_FREE;
}
