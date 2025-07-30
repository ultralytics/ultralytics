/* mpn_toom42_mul -- Multiply {ap,an} and {bp,bn} where an is nominally twice
   as large as bn.  Or more accurately, (3/2)bn < an < 4bn.

   Contributed to the GNU project by Torbjorn Granlund.
   Additional improvements by Marco Bodrato.

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

/* Evaluate in: -1, 0, +1, +2, +inf

  <-s-><--n--><--n--><--n-->
   ___ ______ ______ ______
  |a3_|___a2_|___a1_|___a0_|
	       |_b1_|___b0_|
	       <-t--><--n-->

  v0  =  a0             * b0      #   A(0)*B(0)
  v1  = (a0+ a1+ a2+ a3)*(b0+ b1) #   A(1)*B(1)      ah  <= 3  bh <= 1
  vm1 = (a0- a1+ a2- a3)*(b0- b1) #  A(-1)*B(-1)    |ah| <= 1  bh  = 0
  v2  = (a0+2a1+4a2+8a3)*(b0+2b1) #   A(2)*B(2)      ah  <= 14 bh <= 2
  vinf=              a3 *     b1  # A(inf)*B(inf)
*/

#define TOOM42_MUL_N_REC(p, a, b, n, ws)				\
  do {									\
    mpn_mul_n (p, a, b, n);						\
  } while (0)

void
mpn_toom42_mul (mp_ptr pp,
		mp_srcptr ap, mp_size_t an,
		mp_srcptr bp, mp_size_t bn,
		mp_ptr scratch)
{
  mp_size_t n, s, t;
  int vm1_neg;
  mp_limb_t cy, vinf0;
  mp_ptr a0_a2;
  mp_ptr as1, asm1, as2;
  mp_ptr bs1, bsm1, bs2;
  mp_ptr tmp;
  TMP_DECL;

#define a0  ap
#define a1  (ap + n)
#define a2  (ap + 2*n)
#define a3  (ap + 3*n)
#define b0  bp
#define b1  (bp + n)

  n = an >= 2 * bn ? (an + 3) >> 2 : (bn + 1) >> 1;

  s = an - 3 * n;
  t = bn - n;

  ASSERT (0 < s && s <= n);
  ASSERT (0 < t && t <= n);

  TMP_MARK;

  tmp = TMP_ALLOC_LIMBS (6 * n + 5);
  as1  = tmp; tmp += n + 1;
  asm1 = tmp; tmp += n + 1;
  as2  = tmp; tmp += n + 1;
  bs1  = tmp; tmp += n + 1;
  bsm1 = tmp; tmp += n;
  bs2  = tmp; tmp += n + 1;

  a0_a2 = pp;

  /* Compute as1 and asm1.  */
  vm1_neg = mpn_toom_eval_dgr3_pm1 (as1, asm1, ap, n, s, a0_a2) & 1;

  /* Compute as2.  */
#if HAVE_NATIVE_mpn_addlsh1_n
  cy  = mpn_addlsh1_n (as2, a2, a3, s);
  if (s != n)
    cy = mpn_add_1 (as2 + s, a2 + s, n - s, cy);
  cy = 2 * cy + mpn_addlsh1_n (as2, a1, as2, n);
  cy = 2 * cy + mpn_addlsh1_n (as2, a0, as2, n);
#else
  cy  = mpn_lshift (as2, a3, s, 1);
  cy += mpn_add_n (as2, a2, as2, s);
  if (s != n)
    cy = mpn_add_1 (as2 + s, a2 + s, n - s, cy);
  cy = 2 * cy + mpn_lshift (as2, as2, n, 1);
  cy += mpn_add_n (as2, a1, as2, n);
  cy = 2 * cy + mpn_lshift (as2, as2, n, 1);
  cy += mpn_add_n (as2, a0, as2, n);
#endif
  as2[n] = cy;

  /* Compute bs1 and bsm1.  */
  if (t == n)
    {
#if HAVE_NATIVE_mpn_add_n_sub_n
      if (mpn_cmp (b0, b1, n) < 0)
	{
	  cy = mpn_add_n_sub_n (bs1, bsm1, b1, b0, n);
	  vm1_neg ^= 1;
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
	  vm1_neg ^= 1;
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
	  vm1_neg ^= 1;
	}
      else
	{
	  mpn_sub (bsm1, b0, n, b1, t);
	}
    }

  /* Compute bs2, recycling bs1. bs2=bs1+b1  */
  mpn_add (bs2, bs1, n + 1, b1, t);

  ASSERT (as1[n] <= 3);
  ASSERT (bs1[n] <= 1);
  ASSERT (asm1[n] <= 1);
/*ASSERT (bsm1[n] == 0);*/
  ASSERT (as2[n] <= 14);
  ASSERT (bs2[n] <= 2);

#define v0    pp				/* 2n */
#define v1    (pp + 2 * n)			/* 2n+1 */
#define vinf  (pp + 4 * n)			/* s+t */
#define vm1   scratch				/* 2n+1 */
#define v2    (scratch + 2 * n + 1)		/* 2n+2 */
#define scratch_out	scratch + 4 * n + 4	/* Currently unused. */

  /* vm1, 2n+1 limbs */
  TOOM42_MUL_N_REC (vm1, asm1, bsm1, n, scratch_out);
  cy = 0;
  if (asm1[n] != 0)
    cy = mpn_add_n (vm1 + n, vm1 + n, bsm1, n);
  vm1[2 * n] = cy;

  TOOM42_MUL_N_REC (v2, as2, bs2, n + 1, scratch_out);	/* v2, 2n+1 limbs */

  /* vinf, s+t limbs */
  if (s > t)  mpn_mul (vinf, a3, s, b1, t);
  else        mpn_mul (vinf, b1, t, a3, s);

  vinf0 = vinf[0];				/* v1 overlaps with this */

  /* v1, 2n+1 limbs */
  TOOM42_MUL_N_REC (v1, as1, bs1, n, scratch_out);
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
  else if (as1[n] == 3)
    {
      cy = 3 * bs1[n] + mpn_addmul_1 (v1 + n, bs1, n, CNST_LIMB(3));
    }
  else
    cy = 0;
  if (bs1[n] != 0)
    cy += mpn_add_n (v1 + n, v1 + n, as1, n);
  v1[2 * n] = cy;

  TOOM42_MUL_N_REC (v0, ap, bp, n, scratch_out);	/* v0, 2n limbs */

  mpn_toom_interpolate_5pts (pp, v2, vm1, n, s + t, vm1_neg, vinf0);

  TMP_FREE;
}
