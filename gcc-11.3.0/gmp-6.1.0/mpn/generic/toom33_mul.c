/* mpn_toom33_mul -- Multiply {ap,an} and {p,bn} where an and bn are close in
   size.  Or more accurately, bn <= an < (3/2)bn.

   Contributed to the GNU project by Torbjorn Granlund.
   Additional improvements by Marco Bodrato.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2006-2008, 2010, 2012 Free Software Foundation, Inc.

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

  <-s--><--n--><--n-->
   ____ ______ ______
  |_a2_|___a1_|___a0_|
   |b2_|___b1_|___b0_|
   <-t-><--n--><--n-->

  v0  =  a0         * b0          #   A(0)*B(0)
  v1  = (a0+ a1+ a2)*(b0+ b1+ b2) #   A(1)*B(1)      ah  <= 2  bh <= 2
  vm1 = (a0- a1+ a2)*(b0- b1+ b2) #  A(-1)*B(-1)    |ah| <= 1  bh <= 1
  v2  = (a0+2a1+4a2)*(b0+2b1+4b2) #   A(2)*B(2)      ah  <= 6  bh <= 6
  vinf=          a2 *         b2  # A(inf)*B(inf)
*/

#if TUNE_PROGRAM_BUILD || WANT_FAT_BINARY
#define MAYBE_mul_basecase 1
#define MAYBE_mul_toom33   1
#else
#define MAYBE_mul_basecase						\
  (MUL_TOOM33_THRESHOLD < 3 * MUL_TOOM22_THRESHOLD)
#define MAYBE_mul_toom33						\
  (MUL_TOOM44_THRESHOLD >= 3 * MUL_TOOM33_THRESHOLD)
#endif

/* FIXME: TOOM33_MUL_N_REC is not quite right for a balanced
   multiplication at the infinity point. We may have
   MAYBE_mul_basecase == 0, and still get s just below
   MUL_TOOM22_THRESHOLD. If MUL_TOOM33_THRESHOLD == 7, we can even get
   s == 1 and mpn_toom22_mul will crash.
*/

#define TOOM33_MUL_N_REC(p, a, b, n, ws)				\
  do {									\
    if (MAYBE_mul_basecase						\
	&& BELOW_THRESHOLD (n, MUL_TOOM22_THRESHOLD))			\
      mpn_mul_basecase (p, a, n, b, n);					\
    else if (! MAYBE_mul_toom33						\
	     || BELOW_THRESHOLD (n, MUL_TOOM33_THRESHOLD))		\
      mpn_toom22_mul (p, a, n, b, n, ws);				\
    else								\
      mpn_toom33_mul (p, a, n, b, n, ws);				\
  } while (0)

void
mpn_toom33_mul (mp_ptr pp,
		mp_srcptr ap, mp_size_t an,
		mp_srcptr bp, mp_size_t bn,
		mp_ptr scratch)
{
  const int __gmpn_cpuvec_initialized = 1;
  mp_size_t n, s, t;
  int vm1_neg;
  mp_limb_t cy, vinf0;
  mp_ptr gp;
  mp_ptr as1, asm1, as2;
  mp_ptr bs1, bsm1, bs2;

#define a0  ap
#define a1  (ap + n)
#define a2  (ap + 2*n)
#define b0  bp
#define b1  (bp + n)
#define b2  (bp + 2*n)

  n = (an + 2) / (size_t) 3;

  s = an - 2 * n;
  t = bn - 2 * n;

  ASSERT (an >= bn);

  ASSERT (0 < s && s <= n);
  ASSERT (0 < t && t <= n);

  as1  = scratch + 4 * n + 4;
  asm1 = scratch + 2 * n + 2;
  as2 = pp + n + 1;

  bs1 = pp;
  bsm1 = scratch + 3 * n + 3; /* we need 4n+4 <= 4n+s+t */
  bs2 = pp + 2 * n + 2;

  gp = scratch;

  vm1_neg = 0;

  /* Compute as1 and asm1.  */
  cy = mpn_add (gp, a0, n, a2, s);
#if HAVE_NATIVE_mpn_add_n_sub_n
  if (cy == 0 && mpn_cmp (gp, a1, n) < 0)
    {
      cy = mpn_add_n_sub_n (as1, asm1, a1, gp, n);
      as1[n] = cy >> 1;
      asm1[n] = 0;
      vm1_neg = 1;
    }
  else
    {
      mp_limb_t cy2;
      cy2 = mpn_add_n_sub_n (as1, asm1, gp, a1, n);
      as1[n] = cy + (cy2 >> 1);
      asm1[n] = cy - (cy2 & 1);
    }
#else
  as1[n] = cy + mpn_add_n (as1, gp, a1, n);
  if (cy == 0 && mpn_cmp (gp, a1, n) < 0)
    {
      mpn_sub_n (asm1, a1, gp, n);
      asm1[n] = 0;
      vm1_neg = 1;
    }
  else
    {
      cy -= mpn_sub_n (asm1, gp, a1, n);
      asm1[n] = cy;
    }
#endif

  /* Compute as2.  */
#if HAVE_NATIVE_mpn_rsblsh1_n
  cy = mpn_add_n (as2, a2, as1, s);
  if (s != n)
    cy = mpn_add_1 (as2 + s, as1 + s, n - s, cy);
  cy += as1[n];
  cy = 2 * cy + mpn_rsblsh1_n (as2, a0, as2, n);
#else
#if HAVE_NATIVE_mpn_addlsh1_n
  cy  = mpn_addlsh1_n (as2, a1, a2, s);
  if (s != n)
    cy = mpn_add_1 (as2 + s, a1 + s, n - s, cy);
  cy = 2 * cy + mpn_addlsh1_n (as2, a0, as2, n);
#else
  cy = mpn_add_n (as2, a2, as1, s);
  if (s != n)
    cy = mpn_add_1 (as2 + s, as1 + s, n - s, cy);
  cy += as1[n];
  cy = 2 * cy + mpn_lshift (as2, as2, n, 1);
  cy -= mpn_sub_n (as2, as2, a0, n);
#endif
#endif
  as2[n] = cy;

  /* Compute bs1 and bsm1.  */
  cy = mpn_add (gp, b0, n, b2, t);
#if HAVE_NATIVE_mpn_add_n_sub_n
  if (cy == 0 && mpn_cmp (gp, b1, n) < 0)
    {
      cy = mpn_add_n_sub_n (bs1, bsm1, b1, gp, n);
      bs1[n] = cy >> 1;
      bsm1[n] = 0;
      vm1_neg ^= 1;
    }
  else
    {
      mp_limb_t cy2;
      cy2 = mpn_add_n_sub_n (bs1, bsm1, gp, b1, n);
      bs1[n] = cy + (cy2 >> 1);
      bsm1[n] = cy - (cy2 & 1);
    }
#else
  bs1[n] = cy + mpn_add_n (bs1, gp, b1, n);
  if (cy == 0 && mpn_cmp (gp, b1, n) < 0)
    {
      mpn_sub_n (bsm1, b1, gp, n);
      bsm1[n] = 0;
      vm1_neg ^= 1;
    }
  else
    {
      cy -= mpn_sub_n (bsm1, gp, b1, n);
      bsm1[n] = cy;
    }
#endif

  /* Compute bs2.  */
#if HAVE_NATIVE_mpn_rsblsh1_n
  cy = mpn_add_n (bs2, b2, bs1, t);
  if (t != n)
    cy = mpn_add_1 (bs2 + t, bs1 + t, n - t, cy);
  cy += bs1[n];
  cy = 2 * cy + mpn_rsblsh1_n (bs2, b0, bs2, n);
#else
#if HAVE_NATIVE_mpn_addlsh1_n
  cy  = mpn_addlsh1_n (bs2, b1, b2, t);
  if (t != n)
    cy = mpn_add_1 (bs2 + t, b1 + t, n - t, cy);
  cy = 2 * cy + mpn_addlsh1_n (bs2, b0, bs2, n);
#else
  cy  = mpn_add_n (bs2, bs1, b2, t);
  if (t != n)
    cy = mpn_add_1 (bs2 + t, bs1 + t, n - t, cy);
  cy += bs1[n];
  cy = 2 * cy + mpn_lshift (bs2, bs2, n, 1);
  cy -= mpn_sub_n (bs2, bs2, b0, n);
#endif
#endif
  bs2[n] = cy;

  ASSERT (as1[n] <= 2);
  ASSERT (bs1[n] <= 2);
  ASSERT (asm1[n] <= 1);
  ASSERT (bsm1[n] <= 1);
  ASSERT (as2[n] <= 6);
  ASSERT (bs2[n] <= 6);

#define v0    pp				/* 2n */
#define v1    (pp + 2 * n)			/* 2n+1 */
#define vinf  (pp + 4 * n)			/* s+t */
#define vm1   scratch				/* 2n+1 */
#define v2    (scratch + 2 * n + 1)		/* 2n+2 */
#define scratch_out  (scratch + 5 * n + 5)

  /* vm1, 2n+1 limbs */
#ifdef SMALLER_RECURSION
  TOOM33_MUL_N_REC (vm1, asm1, bsm1, n, scratch_out);
  cy = 0;
  if (asm1[n] != 0)
    cy = bsm1[n] + mpn_add_n (vm1 + n, vm1 + n, bsm1, n);
  if (bsm1[n] != 0)
    cy += mpn_add_n (vm1 + n, vm1 + n, asm1, n);
  vm1[2 * n] = cy;
#else
  TOOM33_MUL_N_REC (vm1, asm1, bsm1, n + 1, scratch_out);
#endif

  TOOM33_MUL_N_REC (v2, as2, bs2, n + 1, scratch_out);	/* v2, 2n+1 limbs */

  /* vinf, s+t limbs */
  if (s > t)  mpn_mul (vinf, a2, s, b2, t);
  else        TOOM33_MUL_N_REC (vinf, a2, b2, s, scratch_out);

  vinf0 = vinf[0];				/* v1 overlaps with this */

#ifdef SMALLER_RECURSION
  /* v1, 2n+1 limbs */
  TOOM33_MUL_N_REC (v1, as1, bs1, n, scratch_out);
  if (as1[n] == 1)
    {
      cy = bs1[n] + mpn_add_n (v1 + n, v1 + n, bs1, n);
    }
  else if (as1[n] != 0)
    {
#if HAVE_NATIVE_mpn_addlsh1_n
      cy = 2 * bs1[n] + mpn_addlsh1_n (v1 + n, v1 + n, bs1, n);
#else
      cy = 2 * bs1[n] + mpn_addmul_1 (v1 + n, bs1, n, CNST_LIMB(2));
#endif
    }
  else
    cy = 0;
  if (bs1[n] == 1)
    {
      cy += mpn_add_n (v1 + n, v1 + n, as1, n);
    }
  else if (bs1[n] != 0)
    {
#if HAVE_NATIVE_mpn_addlsh1_n
      cy += mpn_addlsh1_n (v1 + n, v1 + n, as1, n);
#else
      cy += mpn_addmul_1 (v1 + n, as1, n, CNST_LIMB(2));
#endif
    }
  v1[2 * n] = cy;
#else
  cy = vinf[1];
  TOOM33_MUL_N_REC (v1, as1, bs1, n + 1, scratch_out);
  vinf[1] = cy;
#endif

  TOOM33_MUL_N_REC (v0, ap, bp, n, scratch_out);	/* v0, 2n limbs */

  mpn_toom_interpolate_5pts (pp, v2, vm1, n, s + t, vm1_neg, vinf0);
}
