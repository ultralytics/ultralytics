/* Interpolation for the algorithm Toom-Cook 8.5-way.

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

#if GMP_NUMB_BITS < 29
#error Not implemented: Both sublsh_n(,,,28) should be corrected; r2 and r5 need one more LIMB.
#endif

#if GMP_NUMB_BITS < 28
#error Not implemented: divexact_by188513325 and _by182712915 will not work.
#endif


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

#if GMP_NUMB_BITS < 43
#define BIT_CORRECTION 1
#define CORRECTION_BITS GMP_NUMB_BITS
#else
#define BIT_CORRECTION 0
#define CORRECTION_BITS 0
#endif

#define BINVERT_9 \
  ((((GMP_NUMB_MAX / 9) << (6 - GMP_NUMB_BITS % 6)) * 8 & GMP_NUMB_MAX) | 0x39)

#define BINVERT_255 \
  (GMP_NUMB_MAX - ((GMP_NUMB_MAX / 255) << (8 - GMP_NUMB_BITS % 8)))

  /* FIXME: find some more general expressions for inverses */
#if GMP_LIMB_BITS == 32
#define BINVERT_2835  (GMP_NUMB_MASK &		CNST_LIMB(0x53E3771B))
#define BINVERT_42525 (GMP_NUMB_MASK &		CNST_LIMB(0x9F314C35))
#define BINVERT_182712915 (GMP_NUMB_MASK &	CNST_LIMB(0x550659DB))
#define BINVERT_188513325 (GMP_NUMB_MASK &	CNST_LIMB(0xFBC333A5))
#define BINVERT_255x182712915L (GMP_NUMB_MASK &	CNST_LIMB(0x6FC4CB25))
#define BINVERT_255x188513325L (GMP_NUMB_MASK &	CNST_LIMB(0x6864275B))
#if GMP_NAIL_BITS == 0
#define BINVERT_255x182712915H CNST_LIMB(0x1B649A07)
#define BINVERT_255x188513325H CNST_LIMB(0x06DB993A)
#else /* GMP_NAIL_BITS != 0 */
#define BINVERT_255x182712915H \
  (GMP_NUMB_MASK & CNST_LIMB((0x1B649A07<<GMP_NAIL_BITS) | (0x6FC4CB25>>GMP_NUMB_BITS)))
#define BINVERT_255x188513325H \
  (GMP_NUMB_MASK & CNST_LIMB((0x06DB993A<<GMP_NAIL_BITS) | (0x6864275B>>GMP_NUMB_BITS)))
#endif
#else
#if GMP_LIMB_BITS == 64
#define BINVERT_2835  (GMP_NUMB_MASK &	CNST_LIMB(0x938CC70553E3771B))
#define BINVERT_42525 (GMP_NUMB_MASK &	CNST_LIMB(0xE7B40D449F314C35))
#define BINVERT_255x182712915  (GMP_NUMB_MASK &	CNST_LIMB(0x1B649A076FC4CB25))
#define BINVERT_255x188513325  (GMP_NUMB_MASK &	CNST_LIMB(0x06DB993A6864275B))
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

#ifndef mpn_divexact_by255x4
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1
#define mpn_divexact_by255x4(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,CNST_LIMB(255),BINVERT_255,2)
#else
#define mpn_divexact_by255x4(dst,src,size) mpn_divexact_1(dst,src,size,CNST_LIMB(255)<<2)
#endif
#endif

#ifndef mpn_divexact_by9x16
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1
#define mpn_divexact_by9x16(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,CNST_LIMB(9),BINVERT_9,4)
#else
#define mpn_divexact_by9x16(dst,src,size) mpn_divexact_1(dst,src,size,CNST_LIMB(9)<<4)
#endif
#endif

#ifndef mpn_divexact_by42525x16
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1 && defined(BINVERT_42525)
#define mpn_divexact_by42525x16(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,CNST_LIMB(42525),BINVERT_42525,4)
#else
#define mpn_divexact_by42525x16(dst,src,size) mpn_divexact_1(dst,src,size,CNST_LIMB(42525)<<4)
#endif
#endif

#ifndef mpn_divexact_by2835x64
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1 && defined(BINVERT_2835)
#define mpn_divexact_by2835x64(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,CNST_LIMB(2835),BINVERT_2835,6)
#else
#define mpn_divexact_by2835x64(dst,src,size) mpn_divexact_1(dst,src,size,CNST_LIMB(2835)<<6)
#endif
#endif

#ifndef  mpn_divexact_by255x182712915
#if GMP_NUMB_BITS < 36
#if HAVE_NATIVE_mpn_bdiv_q_2_pi2 && defined(BINVERT_255x182712915H)
/* FIXME: use mpn_bdiv_q_2_pi2 */
#endif
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1 && defined(BINVERT_182712915)
#define mpn_divexact_by255x182712915(dst,src,size)				\
  do {										\
    mpn_pi1_bdiv_q_1(dst,src,size,CNST_LIMB(182712915),BINVERT_182712915,0);	\
    mpn_divexact_by255(dst,dst,size);						\
  } while(0)
#else
#define mpn_divexact_by255x182712915(dst,src,size)	\
  do {							\
    mpn_divexact_1(dst,src,size,CNST_LIMB(182712915));	\
    mpn_divexact_by255(dst,dst,size);			\
  } while(0)
#endif
#else /* GMP_NUMB_BITS > 35 */
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1 && defined(BINVERT_255x182712915)
#define mpn_divexact_by255x182712915(dst,src,size) \
  mpn_pi1_bdiv_q_1(dst,src,size,255*CNST_LIMB(182712915),BINVERT_255x182712915,0)
#else
#define mpn_divexact_by255x182712915(dst,src,size) mpn_divexact_1(dst,src,size,255*CNST_LIMB(182712915))
#endif
#endif /* GMP_NUMB_BITS >?< 36 */
#endif

#ifndef  mpn_divexact_by255x188513325
#if GMP_NUMB_BITS < 36
#if HAVE_NATIVE_mpn_bdiv_q_1_pi2 && defined(BINVERT_255x188513325H)
/* FIXME: use mpn_bdiv_q_1_pi2 */
#endif
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1 && defined(BINVERT_188513325)
#define mpn_divexact_by255x188513325(dst,src,size)			\
  do {									\
    mpn_pi1_bdiv_q_1(dst,src,size,CNST_LIMB(188513325),BINVERT_188513325,0);	\
    mpn_divexact_by255(dst,dst,size);					\
  } while(0)
#else
#define mpn_divexact_by255x188513325(dst,src,size)	\
  do {							\
    mpn_divexact_1(dst,src,size,CNST_LIMB(188513325));	\
    mpn_divexact_by255(dst,dst,size);			\
  } while(0)
#endif
#else /* GMP_NUMB_BITS > 35 */
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1 && defined(BINVERT_255x188513325)
#define mpn_divexact_by255x188513325(dst,src,size) \
  mpn_pi1_bdiv_q_1(dst,src,size,255*CNST_LIMB(188513325),BINVERT_255x188513325,0)
#else
#define mpn_divexact_by255x188513325(dst,src,size) mpn_divexact_1(dst,src,size,255*CNST_LIMB(188513325))
#endif
#endif /* GMP_NUMB_BITS >?< 36 */
#endif

/* Interpolation for Toom-8.5 (or Toom-8), using the evaluation
   points: infinity(8.5 only), +-8, +-4, +-2, +-1, +-1/4, +-1/2,
   +-1/8, 0. More precisely, we want to compute
   f(2^(GMP_NUMB_BITS * n)) for a polynomial f of degree 15 (or
   14), given the 16 (rsp. 15) values:

     r0 = limit at infinity of f(x) / x^7,
     r1 = f(8),f(-8),
     r2 = f(4),f(-4),
     r3 = f(2),f(-2),
     r4 = f(1),f(-1),
     r5 = f(1/4),f(-1/4),
     r6 = f(1/2),f(-1/2),
     r7 = f(1/8),f(-1/8),
     r8 = f(0).

   All couples of the form f(n),f(-n) must be already mixed with
   toom_couple_handling(f(n),...,f(-n),...)

   The result is stored in {pp, spt + 7*n (or 8*n)}.
   At entry, r8 is stored at {pp, 2n},
   r6 is stored at {pp + 3n, 3n + 1}.
   r4 is stored at {pp + 7n, 3n + 1}.
   r2 is stored at {pp +11n, 3n + 1}.
   r0 is stored at {pp +15n, spt}.

   The other values are 3n+1 limbs each (with most significant limbs small).

   Negative intermediate results are stored two-complemented.
   Inputs are destroyed.
*/

void
mpn_toom_interpolate_16pts (mp_ptr pp, mp_ptr r1, mp_ptr r3, mp_ptr r5, mp_ptr r7,
			mp_size_t n, mp_size_t spt, int half, mp_ptr wsi)
{
  mp_limb_t cy;
  mp_size_t n3;
  mp_size_t n3p1;
  n3 = 3 * n;
  n3p1 = n3 + 1;

#define   r6    (pp + n3)			/* 3n+1 */
#define   r4    (pp + 7 * n)			/* 3n+1 */
#define   r2    (pp +11 * n)			/* 3n+1 */
#define   r0    (pp +15 * n)			/* s+t <= 2*n */

  ASSERT( spt <= 2 * n );
  /******************************* interpolation *****************************/
  if( half != 0) {
    cy = mpn_sub_n (r4, r4, r0, spt);
    MPN_DECR_U (r4 + spt, n3p1 - spt, cy);

    cy = DO_mpn_sublsh_n (r3, r0, spt, 14, wsi);
    MPN_DECR_U (r3 + spt, n3p1 - spt, cy);
    DO_mpn_subrsh(r6, n3p1, r0, spt, 2, wsi);

    cy = DO_mpn_sublsh_n (r2, r0, spt, 28, wsi);
    MPN_DECR_U (r2 + spt, n3p1 - spt, cy);
    DO_mpn_subrsh(r5, n3p1, r0, spt, 4, wsi);

    cy = DO_mpn_sublsh_n (r1 + BIT_CORRECTION, r0, spt, 42 - CORRECTION_BITS, wsi);
#if BIT_CORRECTION
    cy = mpn_sub_1 (r1 + spt + BIT_CORRECTION, r1 + spt + BIT_CORRECTION,
		    n3p1 - spt - BIT_CORRECTION, cy);
    ASSERT (BIT_CORRECTION > 0 || cy == 0);
    /* FIXME: assumes r7[n3p1] is writable (it is if r5 follows). */
    cy = r7[n3p1];
    r7[n3p1] = 0x80;
#else
    MPN_DECR_U (r1 + spt + BIT_CORRECTION, n3p1 - spt - BIT_CORRECTION, cy);
#endif
    DO_mpn_subrsh(r7, n3p1 + BIT_CORRECTION, r0, spt, 6, wsi);
#if BIT_CORRECTION
    /* FIXME: assumes r7[n3p1] is writable. */
    ASSERT ( BIT_CORRECTION > 0 || r7[n3p1] == 0x80 );
    r7[n3p1] = cy;
#endif
  };

  r5[n3] -= DO_mpn_sublsh_n (r5 + n, pp, 2 * n, 28, wsi);
  DO_mpn_subrsh(r2 + n, 2 * n + 1, pp, 2 * n, 4, wsi);

#if HAVE_NATIVE_mpn_add_n_sub_n
  mpn_add_n_sub_n (r2, r5, r5, r2, n3p1);
#else
  mpn_sub_n (wsi, r5, r2, n3p1); /* can be negative */
  ASSERT_NOCARRY(mpn_add_n (r2, r2, r5, n3p1));
  MP_PTR_SWAP(r5, wsi);
#endif

  r6[n3] -= DO_mpn_sublsh_n (r6 + n, pp, 2 * n, 14, wsi);
  DO_mpn_subrsh(r3 + n, 2 * n + 1, pp, 2 * n, 2, wsi);

#if HAVE_NATIVE_mpn_add_n_sub_n
  mpn_add_n_sub_n (r3, r6, r6, r3, n3p1);
#else
  ASSERT_NOCARRY(mpn_add_n (wsi, r3, r6, n3p1));
  mpn_sub_n (r6, r6, r3, n3p1); /* can be negative */
  MP_PTR_SWAP(r3, wsi);
#endif

  cy = DO_mpn_sublsh_n (r7 + n + BIT_CORRECTION, pp, 2 * n, 42 - CORRECTION_BITS, wsi);
#if BIT_CORRECTION
  MPN_DECR_U (r1 + n, 2 * n + 1, pp[0] >> 6);
  cy = DO_mpn_sublsh_n (r1 + n, pp + 1, 2 * n - 1, GMP_NUMB_BITS - 6, wsi);
  cy = mpn_sub_1(r1 + 3 * n - 1, r1 + 3 * n - 1, 2, cy);
  ASSERT ( BIT_CORRECTION > 0 || cy != 0 );
#else
  r7[n3] -= cy;
  DO_mpn_subrsh(r1 + n, 2 * n + 1, pp, 2 * n, 6, wsi);
#endif

#if HAVE_NATIVE_mpn_add_n_sub_n
  mpn_add_n_sub_n (r1, r7, r7, r1, n3p1);
#else
  mpn_sub_n (wsi, r7, r1, n3p1); /* can be negative */
  mpn_add_n (r1, r1, r7, n3p1);  /* if BIT_CORRECTION != 0, can give a carry. */
  MP_PTR_SWAP(r7, wsi);
#endif

  r4[n3] -= mpn_sub_n (r4+n, r4+n, pp, 2 * n);

#if AORSMUL_FASTER_2AORSLSH
  mpn_submul_1 (r5, r6, n3p1, 1028); /* can be negative */
#else
  DO_mpn_sublsh_n (r5, r6, n3p1, 2, wsi); /* can be negative */
  DO_mpn_sublsh_n (r5, r6, n3p1,10, wsi); /* can be negative */
#endif

  mpn_submul_1 (r7, r5, n3p1, 1300); /* can be negative */
#if AORSMUL_FASTER_3AORSLSH
  mpn_submul_1 (r7, r6, n3p1, 1052688); /* can be negative */
#else
  DO_mpn_sublsh_n (r7, r6, n3p1, 4, wsi); /* can be negative */
  DO_mpn_sublsh_n (r7, r6, n3p1,12, wsi); /* can be negative */
  DO_mpn_sublsh_n (r7, r6, n3p1,20, wsi); /* can be negative */
#endif
  mpn_divexact_by255x188513325(r7, r7, n3p1);

  mpn_submul_1 (r5, r7, n3p1, 12567555); /* can be negative */
  /* A division by 2835x64 follows. Warning: the operand can be negative! */
  mpn_divexact_by2835x64(r5, r5, n3p1);
  if ((r5[n3] & (GMP_NUMB_MAX << (GMP_NUMB_BITS-7))) != 0)
    r5[n3] |= (GMP_NUMB_MAX << (GMP_NUMB_BITS-6));

#if AORSMUL_FASTER_AORS_AORSLSH
  mpn_submul_1 (r6, r7, n3p1, 4095); /* can be negative */
#else
  mpn_add_n (r6, r6, r7, n3p1); /* can give a carry */
  DO_mpn_sublsh_n (r6, r7, n3p1, 12, wsi); /* can be negative */
#endif
#if AORSMUL_FASTER_2AORSLSH
  mpn_addmul_1 (r6, r5, n3p1, 240); /* can be negative */
#else
  DO_mpn_addlsh_n (r6, r5, n3p1, 8, wsi); /* can give a carry */
  DO_mpn_sublsh_n (r6, r5, n3p1, 4, wsi); /* can be negative */
#endif
  /* A division by 255x4 follows. Warning: the operand can be negative! */
  mpn_divexact_by255x4(r6, r6, n3p1);
  if ((r6[n3] & (GMP_NUMB_MAX << (GMP_NUMB_BITS-3))) != 0)
    r6[n3] |= (GMP_NUMB_MAX << (GMP_NUMB_BITS-2));

  ASSERT_NOCARRY(DO_mpn_sublsh_n (r3, r4, n3p1, 7, wsi));

  ASSERT_NOCARRY(DO_mpn_sublsh_n (r2, r4, n3p1, 13, wsi));
  ASSERT_NOCARRY(mpn_submul_1 (r2, r3, n3p1, 400));

  /* If GMP_NUMB_BITS < 42 next operations on r1 can give a carry!*/
  DO_mpn_sublsh_n (r1, r4, n3p1, 19, wsi);
  mpn_submul_1 (r1, r2, n3p1, 1428);
  mpn_submul_1 (r1, r3, n3p1, 112896);
  mpn_divexact_by255x182712915(r1, r1, n3p1);

  ASSERT_NOCARRY(mpn_submul_1 (r2, r1, n3p1, 15181425));
  mpn_divexact_by42525x16(r2, r2, n3p1);

#if AORSMUL_FASTER_AORS_2AORSLSH
  ASSERT_NOCARRY(mpn_submul_1 (r3, r1, n3p1, 3969));
#else
  ASSERT_NOCARRY(mpn_sub_n (r3, r3, r1, n3p1));
  ASSERT_NOCARRY(DO_mpn_addlsh_n (r3, r1, n3p1, 7, wsi));
  ASSERT_NOCARRY(DO_mpn_sublsh_n (r3, r1, n3p1, 12, wsi));
#endif
  ASSERT_NOCARRY(mpn_submul_1 (r3, r2, n3p1, 900));
  mpn_divexact_by9x16(r3, r3, n3p1);

  ASSERT_NOCARRY(mpn_sub_n (r4, r4, r1, n3p1));
  ASSERT_NOCARRY(mpn_sub_n (r4, r4, r3, n3p1));
  ASSERT_NOCARRY(mpn_sub_n (r4, r4, r2, n3p1));

  mpn_add_n (r6, r2, r6, n3p1);
  ASSERT_NOCARRY(mpn_rshift(r6, r6, n3p1, 1));
  ASSERT_NOCARRY(mpn_sub_n (r2, r2, r6, n3p1));

  mpn_sub_n (r5, r3, r5, n3p1);
  ASSERT_NOCARRY(mpn_rshift(r5, r5, n3p1, 1));
  ASSERT_NOCARRY(mpn_sub_n (r3, r3, r5, n3p1));

  mpn_add_n (r7, r1, r7, n3p1);
  ASSERT_NOCARRY(mpn_rshift(r7, r7, n3p1, 1));
  ASSERT_NOCARRY(mpn_sub_n (r1, r1, r7, n3p1));

  /* last interpolation steps... */
  /* ... could be mixed with recomposition
	||H-r7|M-r7|L-r7|   ||H-r5|M-r5|L-r5|
  */

  /***************************** recomposition *******************************/
  /*
    pp[] prior to operations:
    |M r0|L r0|___||H r2|M r2|L r2|___||H r4|M r4|L r4|___||H r6|M r6|L r6|____|H_r8|L r8|pp

    summation scheme for remaining operations:
    |__16|n_15|n_14|n_13|n_12|n_11|n_10|n__9|n__8|n__7|n__6|n__5|n__4|n__3|n__2|n___|n___|pp
    |M r0|L r0|___||H r2|M r2|L r2|___||H r4|M r4|L r4|___||H r6|M r6|L r6|____|H_r8|L r8|pp
	||H r1|M r1|L r1|   ||H r3|M r3|L r3|   ||H_r5|M_r5|L_r5|   ||H r7|M r7|L r7|
  */

  cy = mpn_add_n (pp + n, pp + n, r7, n);
  cy = mpn_add_1 (pp + 2 * n, r7 + n, n, cy);
#if HAVE_NATIVE_mpn_add_nc
  cy = r7[n3] + mpn_add_nc(pp + n3, pp + n3, r7 + 2 * n, n, cy);
#else
  MPN_INCR_U (r7 + 2 * n, n + 1, cy);
  cy = r7[n3] + mpn_add_n (pp + n3, pp + n3, r7 + 2 * n, n);
#endif
  MPN_INCR_U (pp + 4 * n, 2 * n + 1, cy);

  pp[2 * n3]+= mpn_add_n (pp + 5 * n, pp + 5 * n, r5, n);
  cy = mpn_add_1 (pp + 2 * n3, r5 + n, n, pp[2 * n3]);
#if HAVE_NATIVE_mpn_add_nc
  cy = r5[n3] + mpn_add_nc(pp + 7 * n, pp + 7 * n, r5 + 2 * n, n, cy);
#else
  MPN_INCR_U (r5 + 2 * n, n + 1, cy);
  cy = r5[n3] + mpn_add_n (pp + 7 * n, pp + 7 * n, r5 + 2 * n, n);
#endif
  MPN_INCR_U (pp + 8 * n, 2 * n + 1, cy);

  pp[10 * n]+= mpn_add_n (pp + 9 * n, pp + 9 * n, r3, n);
  cy = mpn_add_1 (pp + 10 * n, r3 + n, n, pp[10 * n]);
#if HAVE_NATIVE_mpn_add_nc
  cy = r3[n3] + mpn_add_nc(pp +11 * n, pp +11 * n, r3 + 2 * n, n, cy);
#else
  MPN_INCR_U (r3 + 2 * n, n + 1, cy);
  cy = r3[n3] + mpn_add_n (pp +11 * n, pp +11 * n, r3 + 2 * n, n);
#endif
  MPN_INCR_U (pp +12 * n, 2 * n + 1, cy);

  pp[14 * n]+=mpn_add_n (pp +13 * n, pp +13 * n, r1, n);
  if ( half ) {
    cy = mpn_add_1 (pp + 14 * n, r1 + n, n, pp[14 * n]);
#if HAVE_NATIVE_mpn_add_nc
    if(LIKELY(spt > n)) {
      cy = r1[n3] + mpn_add_nc(pp + 15 * n, pp + 15 * n, r1 + 2 * n, n, cy);
      MPN_INCR_U (pp + 16 * n, spt - n, cy);
    } else {
      ASSERT_NOCARRY(mpn_add_nc(pp + 15 * n, pp + 15 * n, r1 + 2 * n, spt, cy));
    }
#else
    MPN_INCR_U (r1 + 2 * n, n + 1, cy);
    if(LIKELY(spt > n)) {
      cy = r1[n3] + mpn_add_n (pp + 15 * n, pp + 15 * n, r1 + 2 * n, n);
      MPN_INCR_U (pp + 16 * n, spt - n, cy);
    } else {
      ASSERT_NOCARRY(mpn_add_n (pp + 15 * n, pp + 15 * n, r1 + 2 * n, spt));
    }
#endif
  } else {
    ASSERT_NOCARRY(mpn_add_1 (pp + 14 * n, r1 + n, spt, pp[14 * n]));
  }

#undef   r0
#undef   r2
#undef   r4
#undef   r6
}
