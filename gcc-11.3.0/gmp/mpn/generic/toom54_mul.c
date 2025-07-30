/* Implementation of the algorithm for Toom-Cook 4.5-way.

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


/* Toom-4.5, the splitting 5x4 unbalanced version.
   Evaluate in: infinity, +4, -4, +2, -2, +1, -1, 0.

  <--s-><--n--><--n--><--n--><--n-->
   ____ ______ ______ ______ ______
  |_a4_|__a3__|__a2__|__a1__|__a0__|
	  |b3_|__b2__|__b1__|__b0__|
	  <-t-><--n--><--n--><--n-->

*/
#define TOOM_54_MUL_N_REC(p, a, b, n, ws)		\
  do {	mpn_mul_n (p, a, b, n);				\
  } while (0)

#define TOOM_54_MUL_REC(p, a, na, b, nb, ws)		\
  do {	mpn_mul (p, a, na, b, nb);			\
  } while (0)

void
mpn_toom54_mul (mp_ptr pp,
		mp_srcptr ap, mp_size_t an,
		mp_srcptr bp, mp_size_t bn, mp_ptr scratch)
{
  mp_size_t n, s, t;
  int sign;

  /***************************** decomposition *******************************/
#define a4  (ap + 4 * n)
#define b3  (bp + 3 * n)

  ASSERT (an >= bn);
  n = 1 + (4 * an >= 5 * bn ? (an - 1) / (size_t) 5 : (bn - 1) / (size_t) 4);

  s = an - 4 * n;
  t = bn - 3 * n;

  ASSERT (0 < s && s <= n);
  ASSERT (0 < t && t <= n);
  /* Required by mpn_toom_interpolate_8pts. */
  ASSERT ( s + t >= n );
  ASSERT ( s + t > 4);
  ASSERT ( n > 2);

#define   r8    pp				/* 2n   */
#define   r7    scratch				/* 3n+1 */
#define   r5    (pp + 3*n)			/* 3n+1 */
#define   v0    (pp + 3*n)			/* n+1 */
#define   v1    (pp + 4*n+1)			/* n+1 */
#define   v2    (pp + 5*n+2)			/* n+1 */
#define   v3    (pp + 6*n+3)			/* n+1 */
#define   r3    (scratch + 3 * n + 1)		/* 3n+1 */
#define   r1    (pp + 7*n)			/* s+t <= 2*n */
#define   ws    (scratch + 6 * n + 2)		/* ??? */

  /* Alloc also 3n+1 limbs for ws... mpn_toom_interpolate_8pts may
     need all of them, when DO_mpn_sublsh_n usea a scratch  */
  /********************** evaluation and recursive calls *********************/
  /* $\pm4$ */
  sign = mpn_toom_eval_pm2exp (v2, v0, 4, ap, n, s, 2, pp)
       ^ mpn_toom_eval_pm2exp (v3, v1, 3, bp, n, t, 2, pp);
  TOOM_54_MUL_N_REC(pp, v0, v1, n + 1, ws); /* A(-4)*B(-4) */
  TOOM_54_MUL_N_REC(r3, v2, v3, n + 1, ws); /* A(+4)*B(+4) */
  mpn_toom_couple_handling (r3, 2*n+1, pp, sign, n, 2, 4);

  /* $\pm1$ */
  sign = mpn_toom_eval_pm1 (v2, v0, 4, ap, n, s,    pp)
       ^ mpn_toom_eval_dgr3_pm1 (v3, v1, bp, n, t,    pp);
  TOOM_54_MUL_N_REC(pp, v0, v1, n + 1, ws); /* A(-1)*B(-1) */
  TOOM_54_MUL_N_REC(r7, v2, v3, n + 1, ws); /* A(1)*B(1) */
  mpn_toom_couple_handling (r7, 2*n+1, pp, sign, n, 0, 0);

  /* $\pm2$ */
  sign = mpn_toom_eval_pm2 (v2, v0, 4, ap, n, s, pp)
       ^ mpn_toom_eval_dgr3_pm2 (v3, v1, bp, n, t, pp);
  TOOM_54_MUL_N_REC(pp, v0, v1, n + 1, ws); /* A(-2)*B(-2) */
  TOOM_54_MUL_N_REC(r5, v2, v3, n + 1, ws); /* A(+2)*B(+2) */
  mpn_toom_couple_handling (r5, 2*n+1, pp, sign, n, 1, 2);

  /* A(0)*B(0) */
  TOOM_54_MUL_N_REC(pp, ap, bp, n, ws);

  /* Infinity */
  if (s > t) {
    TOOM_54_MUL_REC(r1, a4, s, b3, t, ws);
  } else {
    TOOM_54_MUL_REC(r1, b3, t, a4, s, ws);
  };

  mpn_toom_interpolate_8pts (pp, n, r3, r7, s + t, ws);

#undef a4
#undef b3
#undef r1
#undef r3
#undef r5
#undef v0
#undef v1
#undef v2
#undef v3
#undef r7
#undef r8
#undef ws
}
