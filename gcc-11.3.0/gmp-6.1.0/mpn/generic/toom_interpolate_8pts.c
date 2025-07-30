/* mpn_toom_interpolate_8pts -- Interpolate for toom54, 63, 72.

   Contributed to the GNU project by Marco Bodrato.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2009, 2011, 2012 Free Software Foundation, Inc.

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

#define BINVERT_3 MODLIMB_INVERSE_3

#define BINVERT_15 \
  ((((GMP_NUMB_MAX >> (GMP_NUMB_BITS % 4)) / 15) * 14 * 16 & GMP_NUMB_MAX) + 15)

#define BINVERT_45 ((BINVERT_15 * BINVERT_3) & GMP_NUMB_MASK)

#ifndef mpn_divexact_by3
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1
#define mpn_divexact_by3(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,3,BINVERT_3,0)
#else
#define mpn_divexact_by3(dst,src,size) mpn_divexact_1(dst,src,size,3)
#endif
#endif

#ifndef mpn_divexact_by45
#if GMP_NUMB_BITS % 12 == 0
#define mpn_divexact_by45(dst,src,size) \
  (63 & 19 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 45)))
#else
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1
#define mpn_divexact_by45(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,45,BINVERT_45,0)
#else
#define mpn_divexact_by45(dst,src,size) mpn_divexact_1(dst,src,size,45)
#endif
#endif
#endif

#if HAVE_NATIVE_mpn_sublsh2_n_ip1
#define DO_mpn_sublsh2_n(dst,src,n,ws) mpn_sublsh2_n_ip1(dst,src,n)
#else
#define DO_mpn_sublsh2_n(dst,src,n,ws) DO_mpn_sublsh_n(dst,src,n,2,ws)
#endif

#if HAVE_NATIVE_mpn_sublsh_n
#define DO_mpn_sublsh_n(dst,src,n,s,ws) mpn_sublsh_n (dst,dst,src,n,s)
#else
static mp_limb_t
DO_mpn_sublsh_n (mp_ptr dst, mp_srcptr src, mp_size_t n, unsigned int s, mp_ptr ws)
{
#if USE_MUL_1 && 0
  return mpn_submul_1(dst,src,n,CNST_LIMB(1) <<(s));
#else
  mp_limb_t __cy;
  __cy = mpn_lshift (ws,src,n,s);
  return __cy + mpn_sub_n (dst,dst,ws,n);
#endif
}
#endif


#if HAVE_NATIVE_mpn_subrsh
#define DO_mpn_subrsh(dst,nd,src,ns,s,ws) mpn_subrsh (dst,nd,src,ns,s)
#else
/* This is not a correct definition, it assumes no carry */
#define DO_mpn_subrsh(dst,nd,src,ns,s,ws)				\
do {									\
  mp_limb_t __cy;							\
  MPN_DECR_U (dst, nd, src[0] >> s);					\
  __cy = DO_mpn_sublsh_n (dst, src + 1, ns - 1, GMP_NUMB_BITS - s, ws);	\
  MPN_DECR_U (dst + ns - 1, nd - ns + 1, __cy);				\
} while (0)
#endif

/* Interpolation for Toom-4.5 (or Toom-4), using the evaluation
   points: infinity(4.5 only), 4, -4, 2, -2, 1, -1, 0. More precisely,
   we want to compute f(2^(GMP_NUMB_BITS * n)) for a polynomial f of
   degree 7 (or 6), given the 8 (rsp. 7) values:

     r1 = limit at infinity of f(x) / x^7,
     r2 = f(4),
     r3 = f(-4),
     r4 = f(2),
     r5 = f(-2),
     r6 = f(1),
     r7 = f(-1),
     r8 = f(0).

   All couples of the form f(n),f(-n) must be already mixed with
   toom_couple_handling(f(n),...,f(-n),...)

   The result is stored in {pp, spt + 7*n (or 6*n)}.
   At entry, r8 is stored at {pp, 2n},
   r5 is stored at {pp + 3n, 3n + 1}.

   The other values are 2n+... limbs each (with most significant limbs small).

   All intermediate results are positive.
   Inputs are destroyed.
*/

void
mpn_toom_interpolate_8pts (mp_ptr pp, mp_size_t n,
			   mp_ptr r3, mp_ptr r7,
			   mp_size_t spt, mp_ptr ws)
{
  mp_limb_signed_t cy;
  mp_ptr r5, r1;
  r5 = (pp + 3 * n);			/* 3n+1 */
  r1 = (pp + 7 * n);			/* spt */

  /******************************* interpolation *****************************/

  DO_mpn_subrsh(r3+n, 2 * n + 1, pp, 2 * n, 4, ws);
  cy = DO_mpn_sublsh_n (r3, r1, spt, 12, ws);
  MPN_DECR_U (r3 + spt, 3 * n + 1 - spt, cy);

  DO_mpn_subrsh(r5+n, 2 * n + 1, pp, 2 * n, 2, ws);
  cy = DO_mpn_sublsh_n (r5, r1, spt, 6, ws);
  MPN_DECR_U (r5 + spt, 3 * n + 1 - spt, cy);

  r7[3*n] -= mpn_sub_n (r7+n, r7+n, pp, 2 * n);
  cy = mpn_sub_n (r7, r7, r1, spt);
  MPN_DECR_U (r7 + spt, 3 * n + 1 - spt, cy);

  ASSERT_NOCARRY(mpn_sub_n (r3, r3, r5, 3 * n + 1));
  ASSERT_NOCARRY(mpn_rshift(r3, r3, 3 * n + 1, 2));

  ASSERT_NOCARRY(mpn_sub_n (r5, r5, r7, 3 * n + 1));

  ASSERT_NOCARRY(mpn_sub_n (r3, r3, r5, 3 * n + 1));

  mpn_divexact_by45 (r3, r3, 3 * n + 1);

  ASSERT_NOCARRY(mpn_divexact_by3 (r5, r5, 3 * n + 1));

  ASSERT_NOCARRY(DO_mpn_sublsh2_n (r5, r3, 3 * n + 1, ws));

  /* last interpolation steps... */
  /* ... are mixed with recomposition */

  /***************************** recomposition *******************************/
  /*
    pp[] prior to operations:
     |_H r1|_L r1|____||_H r5|_M_r5|_L r5|_____|_H r8|_L r8|pp

    summation scheme for remaining operations:
     |____8|n___7|n___6|n___5|n___4|n___3|n___2|n____|n____|pp
     |_H r1|_L r1|____||_H*r5|_M r5|_L r5|_____|_H_r8|_L r8|pp
	  ||_H r3|_M r3|_L*r3|
				  ||_H_r7|_M_r7|_L_r7|
		      ||-H r3|-M r3|-L*r3|
				  ||-H*r5|-M_r5|-L_r5|
  */

  cy = mpn_add_n (pp + n, pp + n, r7, n); /* Hr8+Lr7-Lr5 */
  cy-= mpn_sub_n (pp + n, pp + n, r5, n);
  if (0 > cy)
    MPN_DECR_U (r7 + n, 2*n + 1, 1);
  else
    MPN_INCR_U (r7 + n, 2*n + 1, cy);

  cy = mpn_sub_n (pp + 2*n, r7 + n, r5 + n, n); /* Mr7-Mr5 */
  MPN_DECR_U (r7 + 2*n, n + 1, cy);

  cy = mpn_add_n (pp + 3*n, r5, r7+ 2*n, n+1); /* Hr7+Lr5 */
  r5[3*n]+= mpn_add_n (r5 + 2*n, r5 + 2*n, r3, n); /* Hr5+Lr3 */
  cy-= mpn_sub_n (pp + 3*n, pp + 3*n, r5 + 2*n, n+1); /* Hr7-Hr5+Lr5-Lr3 */
  if (UNLIKELY(0 > cy))
    MPN_DECR_U (r5 + n + 1, 2*n, 1);
  else
    MPN_INCR_U (r5 + n + 1, 2*n, cy);

  ASSERT_NOCARRY(mpn_sub_n(pp + 4*n, r5 + n, r3 + n, 2*n +1)); /* Mr5-Mr3,Hr5-Hr3 */

  cy = mpn_add_1 (pp + 6*n, r3 + n, n, pp[6*n]);
  MPN_INCR_U (r3 + 2*n, n + 1, cy);
  cy = mpn_add_n (pp + 7*n, pp + 7*n, r3 + 2*n, n);
  if (LIKELY(spt != n))
    MPN_INCR_U (pp + 8*n, spt - n, cy + r3[3*n]);
  else
    ASSERT (r3[3*n] + cy == 0);
}
