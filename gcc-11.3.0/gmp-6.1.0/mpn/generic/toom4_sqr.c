/* mpn_toom4_sqr -- Square {ap,an}.

   Contributed to the GNU project by Torbjorn Granlund and Marco Bodrato.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2006-2010, 2013 Free Software Foundation, Inc.

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

/* Evaluate in: -1, -1/2, 0, +1/2, +1, +2, +inf

  <-s--><--n--><--n--><--n-->
   ____ ______ ______ ______
  |_a3_|___a2_|___a1_|___a0_|

  v0  =   a0             ^2 #    A(0)^2
  v1  = ( a0+ a1+ a2+ a3)^2 #    A(1)^2   ah  <= 3
  vm1 = ( a0- a1+ a2- a3)^2 #   A(-1)^2  |ah| <= 1
  v2  = ( a0+2a1+4a2+8a3)^2 #    A(2)^2   ah  <= 14
  vh  = (8a0+4a1+2a2+ a3)^2 #  A(1/2)^2   ah  <= 14
  vmh = (8a0-4a1+2a2- a3)^2 # A(-1/2)^2  -4<=ah<=9
  vinf=               a3 ^2 #  A(inf)^2
*/

#if TUNE_PROGRAM_BUILD
#define MAYBE_sqr_basecase 1
#define MAYBE_sqr_toom2   1
#define MAYBE_sqr_toom4   1
#else
#define MAYBE_sqr_basecase						\
  (SQR_TOOM4_THRESHOLD < 4 * SQR_TOOM2_THRESHOLD)
#define MAYBE_sqr_toom2							\
  (SQR_TOOM4_THRESHOLD < 4 * SQR_TOOM3_THRESHOLD)
#define MAYBE_sqr_toom4							\
  (SQR_TOOM6_THRESHOLD >= 4 * SQR_TOOM4_THRESHOLD)
#endif

#define TOOM4_SQR_REC(p, a, n, ws)					\
  do {									\
    if (MAYBE_sqr_basecase						\
	&& BELOW_THRESHOLD (n, SQR_TOOM2_THRESHOLD))			\
      mpn_sqr_basecase (p, a, n);					\
    else if (MAYBE_sqr_toom2						\
	     && BELOW_THRESHOLD (n, SQR_TOOM3_THRESHOLD))		\
      mpn_toom2_sqr (p, a, n, ws);					\
    else if (! MAYBE_sqr_toom4						\
	     || BELOW_THRESHOLD (n, SQR_TOOM4_THRESHOLD))		\
      mpn_toom3_sqr (p, a, n, ws);					\
    else								\
      mpn_toom4_sqr (p, a, n, ws);					\
  } while (0)

void
mpn_toom4_sqr (mp_ptr pp,
	       mp_srcptr ap, mp_size_t an,
	       mp_ptr scratch)
{
  mp_size_t n, s;
  mp_limb_t cy;

#define a0  ap
#define a1  (ap + n)
#define a2  (ap + 2*n)
#define a3  (ap + 3*n)

  n = (an + 3) >> 2;

  s = an - 3 * n;

  ASSERT (0 < s && s <= n);

  /* NOTE: The multiplications to v2, vm2, vh and vm1 overwrites the
   * following limb, so these must be computed in order, and we need a
   * one limb gap to tp. */
#define v0    pp				/* 2n */
#define v1    (pp + 2 * n)			/* 2n+1 */
#define vinf  (pp + 6 * n)			/* s+t */
#define v2    scratch				/* 2n+1 */
#define vm2   (scratch + 2 * n + 1)		/* 2n+1 */
#define vh    (scratch + 4 * n + 2)		/* 2n+1 */
#define vm1   (scratch + 6 * n + 3)		/* 2n+1 */
#define tp (scratch + 8*n + 5)

  /* No overlap with v1 */
#define apx   pp				/* n+1 */
#define amx   (pp + 4*n + 2)			/* n+1 */

  /* Total scratch need: 8*n + 5 + scratch for recursive calls. This
     gives roughly 32 n/3 + log term. */

  /* Compute apx = a0 + 2 a1 + 4 a2 + 8 a3 and amx = a0 - 2 a1 + 4 a2 - 8 a3.  */
  mpn_toom_eval_dgr3_pm2 (apx, amx, ap, n, s, tp);

  TOOM4_SQR_REC (v2, apx, n + 1, tp);	/* v2,  2n+1 limbs */
  TOOM4_SQR_REC (vm2, amx, n + 1, tp);	/* vm2,  2n+1 limbs */

  /* Compute apx = 8 a0 + 4 a1 + 2 a2 + a3 = (((2*a0 + a1) * 2 + a2) * 2 + a3 */
#if HAVE_NATIVE_mpn_addlsh1_n
  cy = mpn_addlsh1_n (apx, a1, a0, n);
  cy = 2*cy + mpn_addlsh1_n (apx, a2, apx, n);
  if (s < n)
    {
      mp_limb_t cy2;
      cy2 = mpn_addlsh1_n (apx, a3, apx, s);
      apx[n] = 2*cy + mpn_lshift (apx + s, apx + s, n - s, 1);
      MPN_INCR_U (apx + s, n+1-s, cy2);
    }
  else
    apx[n] = 2*cy + mpn_addlsh1_n (apx, a3, apx, n);
#else
  cy = mpn_lshift (apx, a0, n, 1);
  cy += mpn_add_n (apx, apx, a1, n);
  cy = 2*cy + mpn_lshift (apx, apx, n, 1);
  cy += mpn_add_n (apx, apx, a2, n);
  cy = 2*cy + mpn_lshift (apx, apx, n, 1);
  apx[n] = cy + mpn_add (apx, apx, n, a3, s);
#endif

  ASSERT (apx[n] < 15);

  TOOM4_SQR_REC (vh, apx, n + 1, tp);	/* vh,  2n+1 limbs */

  /* Compute apx = a0 + a1 + a2 + a3 and amx = a0 - a1 + a2 - a3.  */
  mpn_toom_eval_dgr3_pm1 (apx, amx, ap, n, s, tp);

  TOOM4_SQR_REC (v1, apx, n + 1, tp);	/* v1,  2n+1 limbs */
  TOOM4_SQR_REC (vm1, amx, n + 1, tp);	/* vm1,  2n+1 limbs */

  TOOM4_SQR_REC (v0, a0, n, tp);
  TOOM4_SQR_REC (vinf, a3, s, tp);	/* vinf, 2s limbs */

  mpn_toom_interpolate_7pts (pp, n, (enum toom7_flags) 0, vm2, vm1, v2, vh, 2*s, tp);
}
