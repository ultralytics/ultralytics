/* mpn_toom53_mul -- Multiply {ap,an} and {bp,bn} where an is nominally 5/3
   times as large as bn.  Or more accurately, (4/3)bn < an < (5/2)bn.

   Contributed to the GNU project by Torbjorn Granlund and Marco Bodrato.

   The idea of applying toom to unbalanced multiplication is due to Marco
   Bodrato and Alberto Zanoni.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2006-2008, 2012, 2014 Free Software Foundation, Inc.

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

/* Evaluate in: 0, +1, -1, +2, -2, 1/2, +inf

  <-s-><--n--><--n--><--n--><--n-->
   ___ ______ ______ ______ ______
  |a4_|___a3_|___a2_|___a1_|___a0_|
	       |__b2|___b1_|___b0_|
	       <-t--><--n--><--n-->

  v0  =    a0                  *  b0          #    A(0)*B(0)
  v1  = (  a0+ a1+ a2+ a3+  a4)*( b0+ b1+ b2) #    A(1)*B(1)      ah  <= 4   bh <= 2
  vm1 = (  a0- a1+ a2- a3+  a4)*( b0- b1+ b2) #   A(-1)*B(-1)    |ah| <= 2   bh <= 1
  v2  = (  a0+2a1+4a2+8a3+16a4)*( b0+2b1+4b2) #    A(2)*B(2)      ah  <= 30  bh <= 6
  vm2 = (  a0-2a1+4a2-8a3+16a4)*( b0-2b1+4b2) #    A(2)*B(2)     -9<=ah<=20 -1<=bh<=4
  vh  = (16a0+8a1+4a2+2a3+  a4)*(4b0+2b1+ b2) #  A(1/2)*B(1/2)    ah  <= 30  bh <= 6
  vinf=                     a4 *          b2  #  A(inf)*B(inf)
*/

void
mpn_toom53_mul (mp_ptr pp,
		mp_srcptr ap, mp_size_t an,
		mp_srcptr bp, mp_size_t bn,
		mp_ptr scratch)
{
  mp_size_t n, s, t;
  mp_limb_t cy;
  mp_ptr gp;
  mp_ptr as1, asm1, as2, asm2, ash;
  mp_ptr bs1, bsm1, bs2, bsm2, bsh;
  mp_ptr tmp;
  enum toom7_flags flags;
  TMP_DECL;

#define a0  ap
#define a1  (ap + n)
#define a2  (ap + 2*n)
#define a3  (ap + 3*n)
#define a4  (ap + 4*n)
#define b0  bp
#define b1  (bp + n)
#define b2  (bp + 2*n)

  n = 1 + (3 * an >= 5 * bn ? (an - 1) / (size_t) 5 : (bn - 1) / (size_t) 3);

  s = an - 4 * n;
  t = bn - 2 * n;

  ASSERT (0 < s && s <= n);
  ASSERT (0 < t && t <= n);

  TMP_MARK;

  tmp = TMP_ALLOC_LIMBS (10 * (n + 1));
  as1  = tmp; tmp += n + 1;
  asm1 = tmp; tmp += n + 1;
  as2  = tmp; tmp += n + 1;
  asm2 = tmp; tmp += n + 1;
  ash  = tmp; tmp += n + 1;
  bs1  = tmp; tmp += n + 1;
  bsm1 = tmp; tmp += n + 1;
  bs2  = tmp; tmp += n + 1;
  bsm2 = tmp; tmp += n + 1;
  bsh  = tmp; tmp += n + 1;

  gp = pp;

  /* Compute as1 and asm1.  */
  flags = (enum toom7_flags) (toom7_w3_neg & mpn_toom_eval_pm1 (as1, asm1, 4, ap, n, s, gp));

  /* Compute as2 and asm2. */
  flags = (enum toom7_flags) (flags | (toom7_w1_neg & mpn_toom_eval_pm2 (as2, asm2, 4, ap, n, s, gp)));

  /* Compute ash = 16 a0 + 8 a1 + 4 a2 + 2 a3 + a4
     = 2*(2*(2*(2*a0 + a1) + a2) + a3) + a4  */
#if HAVE_NATIVE_mpn_addlsh1_n
  cy = mpn_addlsh1_n (ash, a1, a0, n);
  cy = 2*cy + mpn_addlsh1_n (ash, a2, ash, n);
  cy = 2*cy + mpn_addlsh1_n (ash, a3, ash, n);
  if (s < n)
    {
      mp_limb_t cy2;
      cy2 = mpn_addlsh1_n (ash, a4, ash, s);
      ash[n] = 2*cy + mpn_lshift (ash + s, ash + s, n - s, 1);
      MPN_INCR_U (ash + s, n+1-s, cy2);
    }
  else
    ash[n] = 2*cy + mpn_addlsh1_n (ash, a4, ash, n);
#else
  cy = mpn_lshift (ash, a0, n, 1);
  cy += mpn_add_n (ash, ash, a1, n);
  cy = 2*cy + mpn_lshift (ash, ash, n, 1);
  cy += mpn_add_n (ash, ash, a2, n);
  cy = 2*cy + mpn_lshift (ash, ash, n, 1);
  cy += mpn_add_n (ash, ash, a3, n);
  cy = 2*cy + mpn_lshift (ash, ash, n, 1);
  ash[n] = cy + mpn_add (ash, ash, n, a4, s);
#endif

  /* Compute bs1 and bsm1.  */
  bs1[n] = mpn_add (bs1, b0, n, b2, t);		/* b0 + b2 */
#if HAVE_NATIVE_mpn_add_n_sub_n
  if (bs1[n] == 0 && mpn_cmp (bs1, b1, n) < 0)
    {
      bs1[n] = mpn_add_n_sub_n (bs1, bsm1, b1, bs1, n) >> 1;
      bsm1[n] = 0;
      flags = (enum toom7_flags) (flags ^ toom7_w3_neg);
    }
  else
    {
      cy = mpn_add_n_sub_n (bs1, bsm1, bs1, b1, n);
      bsm1[n] = bs1[n] - (cy & 1);
      bs1[n] += (cy >> 1);
    }
#else
  if (bs1[n] == 0 && mpn_cmp (bs1, b1, n) < 0)
    {
      mpn_sub_n (bsm1, b1, bs1, n);
      bsm1[n] = 0;
      flags = (enum toom7_flags) (flags ^ toom7_w3_neg);
    }
  else
    {
      bsm1[n] = bs1[n] - mpn_sub_n (bsm1, bs1, b1, n);
    }
  bs1[n] += mpn_add_n (bs1, bs1, b1, n);  /* b0+b1+b2 */
#endif

  /* Compute bs2 and bsm2. */
#if HAVE_NATIVE_mpn_addlsh_n || HAVE_NATIVE_mpn_addlsh2_n
#if HAVE_NATIVE_mpn_addlsh2_n
  cy = mpn_addlsh2_n (bs2, b0, b2, t);
#else /* HAVE_NATIVE_mpn_addlsh_n */
  cy = mpn_addlsh_n (bs2, b0, b2, t, 2);
#endif
  if (t < n)
    cy = mpn_add_1 (bs2 + t, b0 + t, n - t, cy);
  bs2[n] = cy;
#else
  cy = mpn_lshift (gp, b2, t, 2);
  bs2[n] = mpn_add (bs2, b0, n, gp, t);
  MPN_INCR_U (bs2 + t, n+1-t, cy);
#endif

  gp[n] = mpn_lshift (gp, b1, n, 1);

#if HAVE_NATIVE_mpn_add_n_sub_n
  if (mpn_cmp (bs2, gp, n+1) < 0)
    {
      ASSERT_NOCARRY (mpn_add_n_sub_n (bs2, bsm2, gp, bs2, n+1));
      flags = (enum toom7_flags) (flags ^ toom7_w1_neg);
    }
  else
    {
      ASSERT_NOCARRY (mpn_add_n_sub_n (bs2, bsm2, bs2, gp, n+1));
    }
#else
  if (mpn_cmp (bs2, gp, n+1) < 0)
    {
      ASSERT_NOCARRY (mpn_sub_n (bsm2, gp, bs2, n+1));
      flags = (enum toom7_flags) (flags ^ toom7_w1_neg);
    }
  else
    {
      ASSERT_NOCARRY (mpn_sub_n (bsm2, bs2, gp, n+1));
    }
  mpn_add_n (bs2, bs2, gp, n+1);
#endif

  /* Compute bsh = 4 b0 + 2 b1 + b2 = 2*(2*b0 + b1)+b2.  */
#if HAVE_NATIVE_mpn_addlsh1_n
  cy = mpn_addlsh1_n (bsh, b1, b0, n);
  if (t < n)
    {
      mp_limb_t cy2;
      cy2 = mpn_addlsh1_n (bsh, b2, bsh, t);
      bsh[n] = 2*cy + mpn_lshift (bsh + t, bsh + t, n - t, 1);
      MPN_INCR_U (bsh + t, n+1-t, cy2);
    }
  else
    bsh[n] = 2*cy + mpn_addlsh1_n (bsh, b2, bsh, n);
#else
  cy = mpn_lshift (bsh, b0, n, 1);
  cy += mpn_add_n (bsh, bsh, b1, n);
  cy = 2*cy + mpn_lshift (bsh, bsh, n, 1);
  bsh[n] = cy + mpn_add (bsh, bsh, n, b2, t);
#endif

  ASSERT (as1[n] <= 4);
  ASSERT (bs1[n] <= 2);
  ASSERT (asm1[n] <= 2);
  ASSERT (bsm1[n] <= 1);
  ASSERT (as2[n] <= 30);
  ASSERT (bs2[n] <= 6);
  ASSERT (asm2[n] <= 20);
  ASSERT (bsm2[n] <= 4);
  ASSERT (ash[n] <= 30);
  ASSERT (bsh[n] <= 6);

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
#ifdef SMALLER_RECURSION
  mpn_mul_n (vm1, asm1, bsm1, n);
  if (asm1[n] == 1)
    {
      cy = bsm1[n] + mpn_add_n (vm1 + n, vm1 + n, bsm1, n);
    }
  else if (asm1[n] == 2)
    {
#if HAVE_NATIVE_mpn_addlsh1_n
      cy = 2 * bsm1[n] + mpn_addlsh1_n (vm1 + n, vm1 + n, bsm1, n);
#else
      cy = 2 * bsm1[n] + mpn_addmul_1 (vm1 + n, bsm1, n, CNST_LIMB(2));
#endif
    }
  else
    cy = 0;
  if (bsm1[n] != 0)
    cy += mpn_add_n (vm1 + n, vm1 + n, asm1, n);
  vm1[2 * n] = cy;
#else /* SMALLER_RECURSION */
  vm1[2 * n] = 0;
  mpn_mul_n (vm1, asm1, bsm1, n + ((asm1[n] | bsm1[n]) != 0));
#endif /* SMALLER_RECURSION */

  /* v1, 2n+1 limbs */
#ifdef SMALLER_RECURSION
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
  if (bs1[n] == 1)
    {
      cy += mpn_add_n (v1 + n, v1 + n, as1, n);
    }
  else if (bs1[n] == 2)
    {
#if HAVE_NATIVE_mpn_addlsh1_n
      cy += mpn_addlsh1_n (v1 + n, v1 + n, as1, n);
#else
      cy += mpn_addmul_1 (v1 + n, as1, n, CNST_LIMB(2));
#endif
    }
  v1[2 * n] = cy;
#else /* SMALLER_RECURSION */
  v1[2 * n] = 0;
  mpn_mul_n (v1, as1, bs1, n + ((as1[n] | bs1[n]) != 0));
#endif /* SMALLER_RECURSION */

  mpn_mul_n (v0, a0, b0, n);			/* v0, 2n limbs */

  /* vinf, s+t limbs */
  if (s > t)  mpn_mul (vinf, a4, s, b2, t);
  else        mpn_mul (vinf, b2, t, a4, s);

  mpn_toom_interpolate_7pts (pp, n, flags, vm2, vm1, v2, vh, s + t,
			     scratch_out);

  TMP_FREE;
}
