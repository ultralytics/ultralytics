/* mpn_invertappr and helper functions.  Compute I such that
   floor((B^{2n}-1)/U - 1 <= I + B^n <= floor((B^{2n}-1)/U.

   Contributed to the GNU project by Marco Bodrato.

   The algorithm used here was inspired by ApproximateReciprocal from "Modern
   Computer Arithmetic", by Richard P. Brent and Paul Zimmermann.  Special
   thanks to Paul Zimmermann for his very valuable suggestions on all the
   theoretical aspects during the work on this code.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright (C) 2007, 2009, 2010, 2012, 2015 Free Software Foundation, Inc.

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

/* FIXME: The iterative version splits the operand in two slightly unbalanced
   parts, the use of log_2 (or counting the bits) underestimate the maximum
   number of iterations.  */

#if TUNE_PROGRAM_BUILD
#define NPOWS \
 ((sizeof(mp_size_t) > 6 ? 48 : 8*sizeof(mp_size_t)))
#define MAYBE_dcpi1_divappr   1
#else
#define NPOWS \
 ((sizeof(mp_size_t) > 6 ? 48 : 8*sizeof(mp_size_t)) - LOG2C (INV_NEWTON_THRESHOLD))
#define MAYBE_dcpi1_divappr \
  (INV_NEWTON_THRESHOLD < DC_DIVAPPR_Q_THRESHOLD)
#if (INV_NEWTON_THRESHOLD > INV_MULMOD_BNM1_THRESHOLD) && \
    (INV_APPR_THRESHOLD > INV_MULMOD_BNM1_THRESHOLD)
#undef  INV_MULMOD_BNM1_THRESHOLD
#define INV_MULMOD_BNM1_THRESHOLD 0 /* always when Newton */
#endif
#endif

/* All the three functions mpn{,_bc,_ni}_invertappr (ip, dp, n, scratch), take
   the strictly normalised value {dp,n} (i.e., most significant bit must be set)
   as an input, and compute {ip,n}: the approximate reciprocal of {dp,n}.

   Let e = mpn*_invertappr (ip, dp, n, scratch) be the returned value; the
   following conditions are satisfied by the output:
     0 <= e <= 1;
     {dp,n}*(B^n+{ip,n}) < B^{2n} <= {dp,n}*(B^n+{ip,n}+1+e) .
   I.e. e=0 means that the result {ip,n} equals the one given by mpn_invert.
	e=1 means that the result _may_ be one less than expected.

   The _bc version returns e=1 most of the time.
   The _ni version should return e=0 most of the time; only about 1% of
   possible random input should give e=1.

   When the strict result is needed, i.e., e=0 in the relation above:
     {dp,n}*(B^n+{ip,n}) < B^{2n} <= {dp,n}*(B^n+{ip,n}+1) ;
   the function mpn_invert (ip, dp, n, scratch) should be used instead.  */

/* Maximum scratch needed by this branch (at xp): 2*n */
static mp_limb_t
mpn_bc_invertappr (mp_ptr ip, mp_srcptr dp, mp_size_t n, mp_ptr xp)
{
  ASSERT (n > 0);
  ASSERT (dp[n-1] & GMP_NUMB_HIGHBIT);
  ASSERT (! MPN_OVERLAP_P (ip, n, dp, n));
  ASSERT (! MPN_OVERLAP_P (ip, n, xp, mpn_invertappr_itch(n)));
  ASSERT (! MPN_OVERLAP_P (dp, n, xp, mpn_invertappr_itch(n)));

  /* Compute a base value of r limbs. */
  if (n == 1)
    invert_limb (*ip, *dp);
  else {
    mp_size_t i;

    /* n > 1 here */
    i = n;
    do
      xp[--i] = GMP_NUMB_MAX;
    while (i);
    mpn_com (xp + n, dp, n);

    /* Now xp contains B^2n - {dp,n}*B^n - 1 */

    /* FIXME: if mpn_*pi1_divappr_q handles n==2, use it! */
    if (n == 2) {
      mpn_divrem_2 (ip, 0, xp, 4, dp);
    } else {
      gmp_pi1_t inv;
      invert_pi1 (inv, dp[n-1], dp[n-2]);
      if (! MAYBE_dcpi1_divappr
	  || BELOW_THRESHOLD (n, DC_DIVAPPR_Q_THRESHOLD))
	mpn_sbpi1_divappr_q (ip, xp, 2 * n, dp, n, inv.inv32);
      else
	mpn_dcpi1_divappr_q (ip, xp, 2 * n, dp, n, &inv);
      MPN_DECR_U(ip, n, CNST_LIMB (1));
      return 1;
    }
  }
  return 0;
}

/* mpn_ni_invertappr: computes the approximate reciprocal using Newton's
   iterations (at least one).

   Inspired by Algorithm "ApproximateReciprocal", published in "Modern Computer
   Arithmetic" by Richard P. Brent and Paul Zimmermann, algorithm 3.5, page 121
   in version 0.4 of the book.

   Some adaptations were introduced, to allow product mod B^m-1 and return the
   value e.

   We introduced a correction in such a way that "the value of
   B^{n+h}-T computed at step 8 cannot exceed B^n-1" (the book reads
   "2B^n-1").

   Maximum scratch needed by this branch <= 2*n, but have to fit 3*rn
   in the scratch, i.e. 3*rn <= 2*n: we require n>4.

   We use a wrapped product modulo B^m-1.  NOTE: is there any normalisation
   problem for the [0] class?  It shouldn't: we compute 2*|A*X_h - B^{n+h}| <
   B^m-1.  We may get [0] if and only if we get AX_h = B^{n+h}.  This can
   happen only if A=B^{n}/2, but this implies X_h = B^{h}*2-1 i.e., AX_h =
   B^{n+h} - A, then we get into the "negative" branch, where X_h is not
   incremented (because A < B^n).

   FIXME: the scratch for mulmod_bnm1 does not currently fit in the scratch, it
   is allocated apart.
 */

mp_limb_t
mpn_ni_invertappr (mp_ptr ip, mp_srcptr dp, mp_size_t n, mp_ptr scratch)
{
  mp_limb_t cy;
  mp_size_t rn, mn;
  mp_size_t sizes[NPOWS], *sizp;
  mp_ptr tp;
  TMP_DECL;
#define xp scratch

  ASSERT (n > 4);
  ASSERT (dp[n-1] & GMP_NUMB_HIGHBIT);
  ASSERT (! MPN_OVERLAP_P (ip, n, dp, n));
  ASSERT (! MPN_OVERLAP_P (ip, n, scratch, mpn_invertappr_itch(n)));
  ASSERT (! MPN_OVERLAP_P (dp, n, scratch, mpn_invertappr_itch(n)));

  /* Compute the computation precisions from highest to lowest, leaving the
     base case size in 'rn'.  */
  sizp = sizes;
  rn = n;
  do {
    *sizp = rn;
    rn = (rn >> 1) + 1;
    ++sizp;
  } while (ABOVE_THRESHOLD (rn, INV_NEWTON_THRESHOLD));

  /* We search the inverse of 0.{dp,n}, we compute it as 1.{ip,n} */
  dp += n;
  ip += n;

  /* Compute a base value of rn limbs. */
  mpn_bc_invertappr (ip - rn, dp - rn, rn, scratch);

  TMP_MARK;

  if (ABOVE_THRESHOLD (n, INV_MULMOD_BNM1_THRESHOLD))
    {
      mn = mpn_mulmod_bnm1_next_size (n + 1);
      tp = TMP_ALLOC_LIMBS (mpn_mulmod_bnm1_itch (mn, n, (n >> 1) + 1));
    }
  /* Use Newton's iterations to get the desired precision.*/

  while (1) {
    n = *--sizp;
    /*
      v    n  v
      +----+--+
      ^ rn ^
    */

    /* Compute i_jd . */
    if (BELOW_THRESHOLD (n, INV_MULMOD_BNM1_THRESHOLD)
	|| ((mn = mpn_mulmod_bnm1_next_size (n + 1)) > (n + rn))) {
      /* FIXME: We do only need {xp,n+1}*/
      mpn_mul (xp, dp - n, n, ip - rn, rn);
      mpn_add_n (xp + rn, xp + rn, dp - n, n - rn + 1);
      cy = CNST_LIMB(1); /* Remember we truncated, Mod B^(n+1) */
      /* We computed (truncated) {xp,n+1} <- 1.{ip,rn} * 0.{dp,n} */
    } else { /* Use B^mn-1 wraparound */
      mpn_mulmod_bnm1 (xp, mn, dp - n, n, ip - rn, rn, tp);
      /* We computed {xp,mn} <- {ip,rn} * {dp,n} mod (B^mn-1) */
      /* We know that 2*|ip*dp + dp*B^rn - B^{rn+n}| < B^mn-1 */
      /* Add dp*B^rn mod (B^mn-1) */
      ASSERT (n >= mn - rn);
      cy = mpn_add_n (xp + rn, xp + rn, dp - n, mn - rn);
      cy = mpn_add_nc (xp, xp, dp - (n - (mn - rn)), n - (mn - rn), cy);
      /* Subtract B^{rn+n}, maybe only compensate the carry*/
      xp[mn] = CNST_LIMB (1); /* set a limit for DECR_U */
      MPN_DECR_U (xp + rn + n - mn, 2 * mn + 1 - rn - n, CNST_LIMB (1) - cy);
      MPN_DECR_U (xp, mn, CNST_LIMB (1) - xp[mn]); /* if DECR_U eroded xp[mn] */
      cy = CNST_LIMB(0); /* Remember we are working Mod B^mn-1 */
    }

    if (xp[n] < CNST_LIMB (2)) { /* "positive" residue class */
      cy = xp[n]; /* 0 <= cy <= 1 here. */
#if HAVE_NATIVE_mpn_sublsh1_n
      if (cy++) {
	if (mpn_cmp (xp, dp - n, n) > 0) {
	  mp_limb_t chk;
	  chk = mpn_sublsh1_n (xp, xp, dp - n, n);
	  ASSERT (chk == xp[n]);
	  ++ cy;
	} else
	  ASSERT_CARRY (mpn_sub_n (xp, xp, dp - n, n));
      }
#else /* no mpn_sublsh1_n*/
      if (cy++ && !mpn_sub_n (xp, xp, dp - n, n)) {
	ASSERT_CARRY (mpn_sub_n (xp, xp, dp - n, n));
	++cy;
      }
#endif
      /* 1 <= cy <= 3 here. */
#if HAVE_NATIVE_mpn_rsblsh1_n
      if (mpn_cmp (xp, dp - n, n) > 0) {
	ASSERT_NOCARRY (mpn_rsblsh1_n (xp + n, xp, dp - n, n));
	++cy;
      } else
	ASSERT_NOCARRY (mpn_sub_nc (xp + 2 * n - rn, dp - rn, xp + n - rn, rn, mpn_cmp (xp, dp - n, n - rn) > 0));
#else /* no mpn_rsblsh1_n*/
      if (mpn_cmp (xp, dp - n, n) > 0) {
	ASSERT_NOCARRY (mpn_sub_n (xp, xp, dp - n, n));
	++cy;
      }
      ASSERT_NOCARRY (mpn_sub_nc (xp + 2 * n - rn, dp - rn, xp + n - rn, rn, mpn_cmp (xp, dp - n, n - rn) > 0));
#endif
      MPN_DECR_U(ip - rn, rn, cy); /* 1 <= cy <= 4 here. */
    } else { /* "negative" residue class */
      ASSERT (xp[n] >= GMP_NUMB_MAX - CNST_LIMB(1));
      MPN_DECR_U(xp, n + 1, cy);
      if (xp[n] != GMP_NUMB_MAX) {
	MPN_INCR_U(ip - rn, rn, CNST_LIMB (1));
	ASSERT_CARRY (mpn_add_n (xp, xp, dp - n, n));
      }
      mpn_com (xp + 2 * n - rn, xp + n - rn, rn);
    }

    /* Compute x_ju_j. FIXME:We need {xp+rn,rn}, mulhi? */
    mpn_mul_n (xp, xp + 2 * n - rn, ip - rn, rn);
    cy = mpn_add_n (xp + rn, xp + rn, xp + 2 * n - rn, 2 * rn - n);
    cy = mpn_add_nc (ip - n, xp + 3 * rn - n, xp + n + rn, n - rn, cy);
    MPN_INCR_U (ip - rn, rn, cy);
    if (sizp == sizes) { /* Get out of the cycle */
      /* Check for possible carry propagation from below. */
      cy = xp[3 * rn - n - 1] > GMP_NUMB_MAX - CNST_LIMB (7); /* Be conservative. */
      /*    cy = mpn_add_1 (xp + rn, xp + rn, 2*rn - n, 4); */
      break;
    }
    rn = n;
  }
  TMP_FREE;

  return cy;
#undef xp
}

mp_limb_t
mpn_invertappr (mp_ptr ip, mp_srcptr dp, mp_size_t n, mp_ptr scratch)
{
  ASSERT (n > 0);
  ASSERT (dp[n-1] & GMP_NUMB_HIGHBIT);
  ASSERT (! MPN_OVERLAP_P (ip, n, dp, n));
  ASSERT (! MPN_OVERLAP_P (ip, n, scratch, mpn_invertappr_itch(n)));
  ASSERT (! MPN_OVERLAP_P (dp, n, scratch, mpn_invertappr_itch(n)));

  if (BELOW_THRESHOLD (n, INV_NEWTON_THRESHOLD))
    return mpn_bc_invertappr (ip, dp, n, scratch);
  else
    return mpn_ni_invertappr (ip, dp, n, scratch);
}
