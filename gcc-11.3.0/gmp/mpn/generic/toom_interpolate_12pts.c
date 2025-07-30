/* Interpolation for the algorithm Toom-Cook 6.5-way.

   Contributed to the GNU project by Marco Bodrato.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

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


#if HAVE_NATIVE_mpn_sublsh_n
#define DO_mpn_sublsh_n(dst,src,n,s,ws) mpn_sublsh_n(dst,dst,src,n,s)
#else
static mp_limb_t
DO_mpn_sublsh_n(mp_ptr dst, mp_srcptr src, mp_size_t n, unsigned int s, mp_ptr ws)
{
#if USE_MUL_1 && 0
  return mpn_submul_1(dst,src,n,CNST_LIMB(1) <<(s));
#else
  mp_limb_t __cy;
  __cy = mpn_lshift(ws,src,n,s);
  return    __cy + mpn_sub_n(dst,dst,ws,n);
#endif
}
#endif

#if HAVE_NATIVE_mpn_addlsh_n
#define DO_mpn_addlsh_n(dst,src,n,s,ws) mpn_addlsh_n(dst,dst,src,n,s)
#else
static mp_limb_t
DO_mpn_addlsh_n(mp_ptr dst, mp_srcptr src, mp_size_t n, unsigned int s, mp_ptr ws)
{
#if USE_MUL_1 && 0
  return mpn_addmul_1(dst,src,n,CNST_LIMB(1) <<(s));
#else
  mp_limb_t __cy;
  __cy = mpn_lshift(ws,src,n,s);
  return    __cy + mpn_add_n(dst,dst,ws,n);
#endif
}
#endif

#if HAVE_NATIVE_mpn_subrsh
#define DO_mpn_subrsh(dst,nd,src,ns,s,ws) mpn_subrsh(dst,nd,src,ns,s)
#else
/* FIXME: This is not a correct definition, it assumes no carry */
#define DO_mpn_subrsh(dst,nd,src,ns,s,ws)				\
do {									\
  mp_limb_t __cy;							\
  MPN_DECR_U (dst, nd, src[0] >> s);					\
  __cy = DO_mpn_sublsh_n (dst, src + 1, ns - 1, GMP_NUMB_BITS - s, ws);	\
  MPN_DECR_U (dst + ns - 1, nd - ns + 1, __cy);				\
} while (0)
#endif


#if GMP_NUMB_BITS < 21
#error Not implemented: Both sublsh_n(,,,20) should be corrected.
#endif

#if GMP_NUMB_BITS < 16
#error Not implemented: divexact_by42525 needs splitting.
#endif

#if GMP_NUMB_BITS < 12
#error Not implemented: Hard to adapt...
#endif

/* FIXME: tuneup should decide the best variant */
#ifndef AORSMUL_FASTER_AORS_AORSLSH
#define AORSMUL_FASTER_AORS_AORSLSH 1
#endif
#ifndef AORSMUL_FASTER_AORS_2AORSLSH
#define AORSMUL_FASTER_AORS_2AORSLSH 1
#endif
#ifndef AORSMUL_FASTER_2AORSLSH
#define AORSMUL_FASTER_2AORSLSH 1
#endif
#ifndef AORSMUL_FASTER_3AORSLSH
#define AORSMUL_FASTER_3AORSLSH 1
#endif

#define BINVERT_9 \
  ((((GMP_NUMB_MAX / 9) << (6 - GMP_NUMB_BITS % 6)) * 8 & GMP_NUMB_MAX) | 0x39)

#define BINVERT_255 \
  (GMP_NUMB_MAX - ((GMP_NUMB_MAX / 255) << (8 - GMP_NUMB_BITS % 8)))

  /* FIXME: find some more general expressions for 2835^-1, 42525^-1 */
#if GMP_LIMB_BITS == 32
#define BINVERT_2835  (GMP_NUMB_MASK &		CNST_LIMB(0x53E3771B))
#define BINVERT_42525 (GMP_NUMB_MASK &		CNST_LIMB(0x9F314C35))
#else
#if GMP_LIMB_BITS == 64
#define BINVERT_2835  (GMP_NUMB_MASK &	CNST_LIMB(0x938CC70553E3771B))
#define BINVERT_42525 (GMP_NUMB_MASK &	CNST_LIMB(0xE7B40D449F314C35))
#endif
#endif

#ifndef mpn_divexact_by255
#if GMP_NUMB_BITS % 8 == 0
#define mpn_divexact_by255(dst,src,size) \
  (255 & 1 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 255)))
#else
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1
#define mpn_divexact_by255(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,CNST_LIMB(255),BINVERT_255,0)
#else
#define mpn_divexact_by255(dst,src,size) mpn_divexact_1(dst,src,size,CNST_LIMB(255))
#endif
#endif
#endif

#ifndef mpn_divexact_by9x4
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1
#define mpn_divexact_by9x4(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,CNST_LIMB(9),BINVERT_9,2)
#else
#define mpn_divexact_by9x4(dst,src,size) mpn_divexact_1(dst,src,size,CNST_LIMB(9)<<2)
#endif
#endif

#ifndef mpn_divexact_by42525
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1 && defined(BINVERT_42525)
#define mpn_divexact_by42525(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,CNST_LIMB(42525),BINVERT_42525,0)
#else
#define mpn_divexact_by42525(dst,src,size) mpn_divexact_1(dst,src,size,CNST_LIMB(42525))
#endif
#endif

#ifndef mpn_divexact_by2835x4
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1 && defined(BINVERT_2835)
#define mpn_divexact_by2835x4(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,CNST_LIMB(2835),BINVERT_2835,2)
#else
#define mpn_divexact_by2835x4(dst,src,size) mpn_divexact_1(dst,src,size,CNST_LIMB(2835)<<2)
#endif
#endif

/* Interpolation for Toom-6.5 (or Toom-6), using the evaluation
   points: infinity(6.5 only), +-4, +-2, +-1, +-1/4, +-1/2, 0. More precisely,
   we want to compute f(2^(GMP_NUMB_BITS * n)) for a polynomial f of
   degree 11 (or 10), given the 12 (rsp. 11) values:

     r0 = limit at infinity of f(x) / x^7,
     r1 = f(4),f(-4),
     r2 = f(2),f(-2),
     r3 = f(1),f(-1),
     r4 = f(1/4),f(-1/4),
     r5 = f(1/2),f(-1/2),
     r6 = f(0).

   All couples of the form f(n),f(-n) must be already mixed with
   toom_couple_handling(f(n),...,f(-n),...)

   The result is stored in {pp, spt + 7*n (or 6*n)}.
   At entry, r6 is stored at {pp, 2n},
   r4 is stored at {pp + 3n, 3n + 1}.
   r2 is stored at {pp + 7n, 3n + 1}.
   r0 is stored at {pp +11n, spt}.

   The other values are 3n+1 limbs each (with most significant limbs small).

   Negative intermediate results are stored two-complemented.
   Inputs are destroyed.
*/

void
mpn_toom_interpolate_12pts (mp_ptr pp, mp_ptr r1, mp_ptr r3, mp_ptr r5,
			mp_size_t n, mp_size_t spt, int half, mp_ptr wsi)
{
  mp_limb_t cy;
  mp_size_t n3;
  mp_size_t n3p1;
  n3 = 3 * n;
  n3p1 = n3 + 1;

#define   r4    (pp + n3)			/* 3n+1 */
#define   r2    (pp + 7 * n)			/* 3n+1 */
#define   r0    (pp +11 * n)			/* s+t <= 2*n */

  /******************************* interpolation *****************************/
  if (half != 0) {
    cy = mpn_sub_n (r3, r3, r0, spt);
    MPN_DECR_U (r3 + spt, n3p1 - spt, cy);

    cy = DO_mpn_sublsh_n (r2, r0, spt, 10, wsi);
    MPN_DECR_U (r2 + spt, n3p1 - spt, cy);
    DO_mpn_subrsh(r5, n3p1, r0, spt, 2, wsi);

    cy = DO_mpn_sublsh_n (r1, r0, spt, 20, wsi);
    MPN_DECR_U (r1 + spt, n3p1 - spt, cy);
    DO_mpn_subrsh(r4, n3p1, r0, spt, 4, wsi);
  };

  r4[n3] -= DO_mpn_sublsh_n (r4 + n, pp, 2 * n, 20, wsi);
  DO_mpn_subrsh(r1 + n, 2 * n + 1, pp, 2 * n, 4, wsi);

#if HAVE_NATIVE_mpn_add_n_sub_n
  mpn_add_n_sub_n (r1, r4, r4, r1, n3p1);
#else
  ASSERT_NOCARRY(mpn_add_n (wsi, r1, r4, n3p1));
  mpn_sub_n (r4, r4, r1, n3p1); /* can be negative */
  MP_PTR_SWAP(r1, wsi);
#endif

  r5[n3] -= DO_mpn_sublsh_n (r5 + n, pp, 2 * n, 10, wsi);
  DO_mpn_subrsh(r2 + n, 2 * n + 1, pp, 2 * n, 2, wsi);

#if HAVE_NATIVE_mpn_add_n_sub_n
  mpn_add_n_sub_n (r2, r5, r5, r2, n3p1);
#else
  mpn_sub_n (wsi, r5, r2, n3p1); /* can be negative */
  ASSERT_NOCARRY(mpn_add_n (r2, r2, r5, n3p1));
  MP_PTR_SWAP(r5, wsi);
#endif

  r3[n3] -= mpn_sub_n (r3+n, r3+n, pp, 2 * n);

#if AORSMUL_FASTER_AORS_AORSLSH
  mpn_submul_1 (r4, r5, n3p1, 257); /* can be negative */
#else
  mpn_sub_n (r4, r4, r5, n3p1); /* can be negative */
  DO_mpn_sublsh_n (r4, r5, n3p1, 8, wsi); /* can be negative */
#endif
  /* A division by 2835x4 follows. Warning: the operand can be negative! */
  mpn_divexact_by2835x4(r4, r4, n3p1);
  if ((r4[n3] & (GMP_NUMB_MAX << (GMP_NUMB_BITS-3))) != 0)
    r4[n3] |= (GMP_NUMB_MAX << (GMP_NUMB_BITS-2));

#if AORSMUL_FASTER_2AORSLSH
  mpn_addmul_1 (r5, r4, n3p1, 60); /* can be negative */
#else
  DO_mpn_sublsh_n (r5, r4, n3p1, 2, wsi); /* can be negative */
  DO_mpn_addlsh_n (r5, r4, n3p1, 6, wsi); /* can give a carry */
#endif
  mpn_divexact_by255(r5, r5, n3p1);

  ASSERT_NOCARRY(DO_mpn_sublsh_n (r2, r3, n3p1, 5, wsi));

#if AORSMUL_FASTER_3AORSLSH
  ASSERT_NOCARRY(mpn_submul_1 (r1, r2, n3p1, 100));
#else
  ASSERT_NOCARRY(DO_mpn_sublsh_n (r1, r2, n3p1, 6, wsi));
  ASSERT_NOCARRY(DO_mpn_sublsh_n (r1, r2, n3p1, 5, wsi));
  ASSERT_NOCARRY(DO_mpn_sublsh_n (r1, r2, n3p1, 2, wsi));
#endif
  ASSERT_NOCARRY(DO_mpn_sublsh_n (r1, r3, n3p1, 9, wsi));
  mpn_divexact_by42525(r1, r1, n3p1);

#if AORSMUL_FASTER_AORS_2AORSLSH
  ASSERT_NOCARRY(mpn_submul_1 (r2, r1, n3p1, 225));
#else
  ASSERT_NOCARRY(mpn_sub_n (r2, r2, r1, n3p1));
  ASSERT_NOCARRY(DO_mpn_addlsh_n (r2, r1, n3p1, 5, wsi));
  ASSERT_NOCARRY(DO_mpn_sublsh_n (r2, r1, n3p1, 8, wsi));
#endif
  mpn_divexact_by9x4(r2, r2, n3p1);

  ASSERT_NOCARRY(mpn_sub_n (r3, r3, r2, n3p1));

  mpn_sub_n (r4, r2, r4, n3p1);
  ASSERT_NOCARRY(mpn_rshift(r4, r4, n3p1, 1));
  ASSERT_NOCARRY(mpn_sub_n (r2, r2, r4, n3p1));

  mpn_add_n (r5, r5, r1, n3p1);
  ASSERT_NOCARRY(mpn_rshift(r5, r5, n3p1, 1));

  /* last interpolation steps... */
  ASSERT_NOCARRY(mpn_sub_n (r3, r3, r1, n3p1));
  ASSERT_NOCARRY(mpn_sub_n (r1, r1, r5, n3p1));
  /* ... could be mixed with recomposition
	||H-r5|M-r5|L-r5|   ||H-r1|M-r1|L-r1|
  */

  /***************************** recomposition *******************************/
  /*
    pp[] prior to operations:
    |M r0|L r0|___||H r2|M r2|L r2|___||H r4|M r4|L r4|____|H_r6|L r6|pp

    summation scheme for remaining operations:
    |__12|n_11|n_10|n__9|n__8|n__7|n__6|n__5|n__4|n__3|n__2|n___|n___|pp
    |M r0|L r0|___||H r2|M r2|L r2|___||H r4|M r4|L r4|____|H_r6|L r6|pp
	||H r1|M r1|L r1|   ||H r3|M r3|L r3|   ||H_r5|M_r5|L_r5|
  */

  cy = mpn_add_n (pp + n, pp + n, r5, n);
  cy = mpn_add_1 (pp + 2 * n, r5 + n, n, cy);
#if HAVE_NATIVE_mpn_add_nc
  cy = r5[n3] + mpn_add_nc(pp + n3, pp + n3, r5 + 2 * n, n, cy);
#else
  MPN_INCR_U (r5 + 2 * n, n + 1, cy);
  cy = r5[n3] + mpn_add_n (pp + n3, pp + n3, r5 + 2 * n, n);
#endif
  MPN_INCR_U (pp + n3 + n, 2 * n + 1, cy);

  pp[2 * n3]+= mpn_add_n (pp + 5 * n, pp + 5 * n, r3, n);
  cy = mpn_add_1 (pp + 2 * n3, r3 + n, n, pp[2 * n3]);
#if HAVE_NATIVE_mpn_add_nc
  cy = r3[n3] + mpn_add_nc(pp + 7 * n, pp + 7 * n, r3 + 2 * n, n, cy);
#else
  MPN_INCR_U (r3 + 2 * n, n + 1, cy);
  cy = r3[n3] + mpn_add_n (pp + 7 * n, pp + 7 * n, r3 + 2 * n, n);
#endif
  MPN_INCR_U (pp + 8 * n, 2 * n + 1, cy);

  pp[10*n]+=mpn_add_n (pp + 9 * n, pp + 9 * n, r1, n);
  if (half) {
    cy = mpn_add_1 (pp + 10 * n, r1 + n, n, pp[10 * n]);
#if HAVE_NATIVE_mpn_add_nc
    if (LIKELY (spt > n)) {
      cy = r1[n3] + mpn_add_nc(pp + 11 * n, pp + 11 * n, r1 + 2 * n, n, cy);
      MPN_INCR_U (pp + 4 * n3, spt - n, cy);
    } else {
      ASSERT_NOCARRY(mpn_add_nc(pp + 11 * n, pp + 11 * n, r1 + 2 * n, spt, cy));
    }
#else
    MPN_INCR_U (r1 + 2 * n, n + 1, cy);
    if (LIKELY (spt > n)) {
      cy = r1[n3] + mpn_add_n (pp + 11 * n, pp + 11 * n, r1 + 2 * n, n);
      MPN_INCR_U (pp + 4 * n3, spt - n, cy);
    } else {
      ASSERT_NOCARRY(mpn_add_n (pp + 11 * n, pp + 11 * n, r1 + 2 * n, spt));
    }
#endif
  } else {
    ASSERT_NOCARRY(mpn_add_1 (pp + 10 * n, r1 + n, spt, pp[10 * n]));
  }

#undef   r0
#undef   r2
#undef   r4
}
