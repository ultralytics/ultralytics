/* Implementation of the squaring algorithm with Toom-Cook 6.5-way.

   Contributed to the GNU project by Marco Bodrato.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2009 Free Software Foundation, Inc.

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


#if GMP_NUMB_BITS < 21
#error Not implemented.
#endif


#if TUNE_PROGRAM_BUILD
#define MAYBE_sqr_basecase 1
#define MAYBE_sqr_above_basecase   1
#define MAYBE_sqr_toom2   1
#define MAYBE_sqr_above_toom2   1
#define MAYBE_sqr_toom3   1
#define MAYBE_sqr_above_toom3   1
#define MAYBE_sqr_above_toom4   1
#else
#ifdef  SQR_TOOM8_THRESHOLD
#define SQR_TOOM6_MAX ((SQR_TOOM8_THRESHOLD+6*2-1+5)/6)
#else
#define SQR_TOOM6_MAX					\
  ((SQR_FFT_THRESHOLD <= MP_SIZE_T_MAX - (6*2-1+5)) ?	\
   ((SQR_FFT_THRESHOLD+6*2-1+5)/6)			\
   : MP_SIZE_T_MAX )
#endif
#define MAYBE_sqr_basecase					\
  (SQR_TOOM6_THRESHOLD < 6 * SQR_TOOM2_THRESHOLD)
#define MAYBE_sqr_above_basecase				\
  (SQR_TOOM6_MAX >=  SQR_TOOM2_THRESHOLD)
#define MAYBE_sqr_toom2						\
  (SQR_TOOM6_THRESHOLD < 6 * SQR_TOOM3_THRESHOLD)
#define MAYBE_sqr_above_toom2					\
  (SQR_TOOM6_MAX >= SQR_TOOM3_THRESHOLD)
#define MAYBE_sqr_toom3						\
  (SQR_TOOM6_THRESHOLD < 6 * SQR_TOOM4_THRESHOLD)
#define MAYBE_sqr_above_toom3					\
  (SQR_TOOM6_MAX >= SQR_TOOM4_THRESHOLD)
#define MAYBE_sqr_above_toom4					\
  (SQR_TOOM6_MAX >= SQR_TOOM6_THRESHOLD)
#endif

#define TOOM6_SQR_REC(p, a, n, ws)					\
  do {									\
    if (MAYBE_sqr_basecase && ( !MAYBE_sqr_above_basecase		\
	|| BELOW_THRESHOLD (n, SQR_TOOM2_THRESHOLD)))			\
      mpn_sqr_basecase (p, a, n);					\
    else if (MAYBE_sqr_toom2 && ( !MAYBE_sqr_above_toom2		\
	     || BELOW_THRESHOLD (n, SQR_TOOM3_THRESHOLD)))		\
      mpn_toom2_sqr (p, a, n, ws);					\
    else if (MAYBE_sqr_toom3 && ( !MAYBE_sqr_above_toom3		\
	     || BELOW_THRESHOLD (n, SQR_TOOM4_THRESHOLD)))		\
      mpn_toom3_sqr (p, a, n, ws);					\
    else if (! MAYBE_sqr_above_toom4					\
	     || BELOW_THRESHOLD (n, SQR_TOOM6_THRESHOLD))		\
      mpn_toom4_sqr (p, a, n, ws);					\
    else								\
      mpn_toom6_sqr (p, a, n, ws);					\
  } while (0)

void
mpn_toom6_sqr  (mp_ptr pp, mp_srcptr ap, mp_size_t an, mp_ptr scratch)
{
  mp_size_t n, s;

  /***************************** decomposition *******************************/

  ASSERT( an >= 18 );

  n = 1 + (an - 1) / (size_t) 6;

  s = an - 5 * n;

  ASSERT (0 < s && s <= n);

#define   r4    (pp + 3 * n)			/* 3n+1 */
#define   r2    (pp + 7 * n)			/* 3n+1 */
#define   r0    (pp +11 * n)			/* s+t <= 2*n */
#define   r5    (scratch)			/* 3n+1 */
#define   r3    (scratch + 3 * n + 1)		/* 3n+1 */
#define   r1    (scratch + 6 * n + 2)		/* 3n+1 */
#define   v0    (pp + 7 * n)			/* n+1 */
#define   v2    (pp + 9 * n+2)			/* n+1 */
#define   wse   (scratch + 9 * n + 3)		/* 3n+1 */

  /* Alloc also 3n+1 limbs for ws... toom_interpolate_12pts may
     need all of them, when DO_mpn_sublsh_n usea a scratch  */
/*   if (scratch== NULL) */
/*     scratch = TMP_SALLOC_LIMBS (12 * n + 6); */

  /********************** evaluation and recursive calls *********************/
  /* $\pm1/2$ */
  mpn_toom_eval_pm2rexp (v2, v0, 5, ap, n, s, 1, pp);
  TOOM6_SQR_REC(pp, v0, n + 1, wse); /* A(-1/2)*B(-1/2)*2^. */
  TOOM6_SQR_REC(r5, v2, n + 1, wse); /* A(+1/2)*B(+1/2)*2^. */
  mpn_toom_couple_handling (r5, 2 * n + 1, pp, 0, n, 1, 0);

  /* $\pm1$ */
  mpn_toom_eval_pm1 (v2, v0, 5, ap, n, s,    pp);
  TOOM6_SQR_REC(pp, v0, n + 1, wse); /* A(-1)*B(-1) */
  TOOM6_SQR_REC(r3, v2, n + 1, wse); /* A(1)*B(1) */
  mpn_toom_couple_handling (r3, 2 * n + 1, pp, 0, n, 0, 0);

  /* $\pm4$ */
  mpn_toom_eval_pm2exp (v2, v0, 5, ap, n, s, 2, pp);
  TOOM6_SQR_REC(pp, v0, n + 1, wse); /* A(-4)*B(-4) */
  TOOM6_SQR_REC(r1, v2, n + 1, wse); /* A(+4)*B(+4) */
  mpn_toom_couple_handling (r1, 2 * n + 1, pp, 0, n, 2, 4);

  /* $\pm1/4$ */
  mpn_toom_eval_pm2rexp (v2, v0, 5, ap, n, s, 2, pp);
  TOOM6_SQR_REC(pp, v0, n + 1, wse); /* A(-1/4)*B(-1/4)*4^. */
  TOOM6_SQR_REC(r4, v2, n + 1, wse); /* A(+1/4)*B(+1/4)*4^. */
  mpn_toom_couple_handling (r4, 2 * n + 1, pp, 0, n, 2, 0);

  /* $\pm2$ */
  mpn_toom_eval_pm2 (v2, v0, 5, ap, n, s, pp);
  TOOM6_SQR_REC(pp, v0, n + 1, wse); /* A(-2)*B(-2) */
  TOOM6_SQR_REC(r2, v2, n + 1, wse); /* A(+2)*B(+2) */
  mpn_toom_couple_handling (r2, 2 * n + 1, pp, 0, n, 1, 2);

#undef v0
#undef v2

  /* A(0)*B(0) */
  TOOM6_SQR_REC(pp, ap, n, wse);

  mpn_toom_interpolate_12pts (pp, r1, r3, r5, n, 2 * s, 0, wse);

#undef r0
#undef r1
#undef r2
#undef r3
#undef r4
#undef r5

}
#undef TOOM6_SQR_REC
#undef MAYBE_sqr_basecase
#undef MAYBE_sqr_above_basecase
#undef MAYBE_sqr_toom2
#undef MAYBE_sqr_above_toom2
#undef MAYBE_sqr_toom3
#undef MAYBE_sqr_above_toom3
#undef MAYBE_sqr_above_toom4
