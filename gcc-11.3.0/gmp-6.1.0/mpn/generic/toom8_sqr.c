/* Implementation of the squaring algorithm with Toom-Cook 8.5-way.

   Contributed to the GNU project by Marco Bodrato.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2009, 2012 Free Software Foundation, Inc.

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
#error Not implemented.
#endif

#if GMP_NUMB_BITS < 43
#define BIT_CORRECTION 1
#define CORRECTION_BITS GMP_NUMB_BITS
#else
#define BIT_CORRECTION 0
#define CORRECTION_BITS 0
#endif

#ifndef SQR_TOOM8_THRESHOLD
#define SQR_TOOM8_THRESHOLD MUL_TOOM8H_THRESHOLD
#endif

#ifndef SQR_TOOM6_THRESHOLD
#define SQR_TOOM6_THRESHOLD MUL_TOOM6H_THRESHOLD
#endif

#if TUNE_PROGRAM_BUILD
#define MAYBE_sqr_basecase 1
#define MAYBE_sqr_above_basecase   1
#define MAYBE_sqr_toom2   1
#define MAYBE_sqr_above_toom2   1
#define MAYBE_sqr_toom3   1
#define MAYBE_sqr_above_toom3   1
#define MAYBE_sqr_toom4   1
#define MAYBE_sqr_above_toom4   1
#define MAYBE_sqr_above_toom6   1
#else
#define SQR_TOOM8_MAX					\
  ((SQR_FFT_THRESHOLD <= MP_SIZE_T_MAX - (8*2-1+7)) ?	\
   ((SQR_FFT_THRESHOLD+8*2-1+7)/8)			\
   : MP_SIZE_T_MAX )
#define MAYBE_sqr_basecase					\
  (SQR_TOOM8_THRESHOLD < 8 * SQR_TOOM2_THRESHOLD)
#define MAYBE_sqr_above_basecase				\
  (SQR_TOOM8_MAX >= SQR_TOOM2_THRESHOLD)
#define MAYBE_sqr_toom2						\
  (SQR_TOOM8_THRESHOLD < 8 * SQR_TOOM3_THRESHOLD)
#define MAYBE_sqr_above_toom2					\
  (SQR_TOOM8_MAX >= SQR_TOOM3_THRESHOLD)
#define MAYBE_sqr_toom3						\
  (SQR_TOOM8_THRESHOLD < 8 * SQR_TOOM4_THRESHOLD)
#define MAYBE_sqr_above_toom3					\
  (SQR_TOOM8_MAX >= SQR_TOOM4_THRESHOLD)
#define MAYBE_sqr_toom4						\
  (SQR_TOOM8_THRESHOLD < 8 * SQR_TOOM6_THRESHOLD)
#define MAYBE_sqr_above_toom4					\
  (SQR_TOOM8_MAX >= SQR_TOOM6_THRESHOLD)
#define MAYBE_sqr_above_toom6					\
  (SQR_TOOM8_MAX >= SQR_TOOM8_THRESHOLD)
#endif

#define TOOM8_SQR_REC(p, a, f, p2, a2, n, ws)				\
  do {									\
    if (MAYBE_sqr_basecase && ( !MAYBE_sqr_above_basecase		\
	|| BELOW_THRESHOLD (n, SQR_TOOM2_THRESHOLD))) {			\
      mpn_sqr_basecase (p, a, n);					\
      if (f) mpn_sqr_basecase (p2, a2, n);				\
    } else if (MAYBE_sqr_toom2 && ( !MAYBE_sqr_above_toom2		\
	     || BELOW_THRESHOLD (n, SQR_TOOM3_THRESHOLD))) {		\
      mpn_toom2_sqr (p, a, n, ws);					\
      if (f) mpn_toom2_sqr (p2, a2, n, ws);				\
    } else if (MAYBE_sqr_toom3 && ( !MAYBE_sqr_above_toom3		\
	     || BELOW_THRESHOLD (n, SQR_TOOM4_THRESHOLD))) {		\
      mpn_toom3_sqr (p, a, n, ws);					\
      if (f) mpn_toom3_sqr (p2, a2, n, ws);				\
    } else if (MAYBE_sqr_toom4 && ( !MAYBE_sqr_above_toom4		\
	     || BELOW_THRESHOLD (n, SQR_TOOM6_THRESHOLD))) {		\
      mpn_toom4_sqr (p, a, n, ws);					\
      if (f) mpn_toom4_sqr (p2, a2, n, ws);				\
    } else if (! MAYBE_sqr_above_toom6					\
	     || BELOW_THRESHOLD (n, SQR_TOOM8_THRESHOLD)) {		\
      mpn_toom6_sqr (p, a, n, ws);					\
      if (f) mpn_toom6_sqr (p2, a2, n, ws);				\
    } else {								\
      mpn_toom8_sqr (p, a, n, ws);					\
      if (f) mpn_toom8_sqr (p2, a2, n, ws);				\
    }									\
  } while (0)

void
mpn_toom8_sqr  (mp_ptr pp, mp_srcptr ap, mp_size_t an, mp_ptr scratch)
{
  mp_size_t n, s;

  /***************************** decomposition *******************************/

  ASSERT ( an >= 40 );

  n = 1 + ((an - 1)>>3);

  s = an - 7 * n;

  ASSERT (0 < s && s <= n);
  ASSERT ( s + s > 3 );

#define   r6    (pp + 3 * n)			/* 3n+1 */
#define   r4    (pp + 7 * n)			/* 3n+1 */
#define   r2    (pp +11 * n)			/* 3n+1 */
#define   r0    (pp +15 * n)			/* s+t <= 2*n */
#define   r7    (scratch)			/* 3n+1 */
#define   r5    (scratch + 3 * n + 1)		/* 3n+1 */
#define   r3    (scratch + 6 * n + 2)		/* 3n+1 */
#define   r1    (scratch + 9 * n + 3)		/* 3n+1 */
#define   v0    (pp +11 * n)			/* n+1 */
#define   v2    (pp +13 * n+2)			/* n+1 */
#define   wse   (scratch +12 * n + 4)		/* 3n+1 */

  /* Alloc also 3n+1 limbs for ws... toom_interpolate_16pts may
     need all of them, when DO_mpn_sublsh_n usea a scratch  */
/*   if (scratch == NULL) */
/*     scratch = TMP_SALLOC_LIMBS (30 * n + 6); */

  /********************** evaluation and recursive calls *********************/
  /* $\pm1/8$ */
  mpn_toom_eval_pm2rexp (v2, v0, 7, ap, n, s, 3, pp);
  /* A(-1/8)*B(-1/8)*8^. */ /* A(+1/8)*B(+1/8)*8^. */
  TOOM8_SQR_REC(pp, v0, 2, r7, v2, n + 1, wse);
  mpn_toom_couple_handling (r7, 2 * n + 1 + BIT_CORRECTION, pp, 0, n, 3, 0);

  /* $\pm1/4$ */
  mpn_toom_eval_pm2rexp (v2, v0, 7, ap, n, s, 2, pp);
  /* A(-1/4)*B(-1/4)*4^. */ /* A(+1/4)*B(+1/4)*4^. */
  TOOM8_SQR_REC(pp, v0, 2, r5, v2, n + 1, wse);
  mpn_toom_couple_handling (r5, 2 * n + 1, pp, 0, n, 2, 0);

  /* $\pm2$ */
  mpn_toom_eval_pm2 (v2, v0, 7, ap, n, s, pp);
  /* A(-2)*B(-2) */ /* A(+2)*B(+2) */
  TOOM8_SQR_REC(pp, v0, 2, r3, v2, n + 1, wse);
  mpn_toom_couple_handling (r3, 2 * n + 1, pp, 0, n, 1, 2);

  /* $\pm8$ */
  mpn_toom_eval_pm2exp (v2, v0, 7, ap, n, s, 3, pp);
  /* A(-8)*B(-8) */ /* A(+8)*B(+8) */
  TOOM8_SQR_REC(pp, v0, 2, r1, v2, n + 1, wse);
  mpn_toom_couple_handling (r1, 2 * n + 1 + BIT_CORRECTION, pp, 0, n, 3, 6);

  /* $\pm1/2$ */
  mpn_toom_eval_pm2rexp (v2, v0, 7, ap, n, s, 1, pp);
  /* A(-1/2)*B(-1/2)*2^. */ /* A(+1/2)*B(+1/2)*2^. */
  TOOM8_SQR_REC(pp, v0, 2, r6, v2, n + 1, wse);
  mpn_toom_couple_handling (r6, 2 * n + 1, pp, 0, n, 1, 0);

  /* $\pm1$ */
  mpn_toom_eval_pm1 (v2, v0, 7, ap, n, s,    pp);
  /* A(-1)*B(-1) */ /* A(1)*B(1) */
  TOOM8_SQR_REC(pp, v0, 2, r4, v2, n + 1, wse);
  mpn_toom_couple_handling (r4, 2 * n + 1, pp, 0, n, 0, 0);

  /* $\pm4$ */
  mpn_toom_eval_pm2exp (v2, v0, 7, ap, n, s, 2, pp);
  /* A(-4)*B(-4) */ /* A(+4)*B(+4) */
  TOOM8_SQR_REC(pp, v0, 2, r2, v2, n + 1, wse);
  mpn_toom_couple_handling (r2, 2 * n + 1, pp, 0, n, 2, 4);

#undef v0
#undef v2

  /* A(0)*B(0) */
  TOOM8_SQR_REC(pp, ap, 0, pp, ap, n, wse);

  mpn_toom_interpolate_16pts (pp, r1, r3, r5, r7, n, 2 * s, 0, wse);

#undef r0
#undef r1
#undef r2
#undef r3
#undef r4
#undef r5
#undef r6
#undef wse

}

#undef TOOM8_SQR_REC
#undef MAYBE_sqr_basecase
#undef MAYBE_sqr_above_basecase
#undef MAYBE_sqr_toom2
#undef MAYBE_sqr_above_toom2
#undef MAYBE_sqr_toom3
#undef MAYBE_sqr_above_toom3
#undef MAYBE_sqr_above_toom4
