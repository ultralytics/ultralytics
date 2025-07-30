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

/* Stores |{ap,n}-{bp,n}| in {rp,n}, returns the sign. */
static int
abs_sub_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  mp_limb_t  x, y;
  while (--n >= 0)
    {
      x = ap[n];
      y = bp[n];
      if (x != y)
	{
	  n++;
	  if (x > y)
	    {
	      mpn_sub_n (rp, ap, bp, n);
	      return 0;
	    }
	  else
	    {
	      mpn_sub_n (rp, bp, ap, n);
	      return ~0;
	    }
	}
      rp[n] = 0;
    }
  return 0;
}

static int
abs_sub_add_n (mp_ptr rm, mp_ptr rp, mp_srcptr rs, mp_size_t n) {
  int result;
  result = abs_sub_n (rm, rp, rs, n);
  ASSERT_NOCARRY(mpn_add_n (rp, rp, rs, n));
  return result;
}


/* Toom-4.5, the splitting 6x3 unbalanced version.
   Evaluate in: infinity, +4, -4, +2, -2, +1, -1, 0.

  <--s-><--n--><--n--><--n--><--n--><--n-->
   ____ ______ ______ ______ ______ ______
  |_a5_|__a4__|__a3__|__a2__|__a1__|__a0__|
			|b2_|__b1__|__b0__|
			<-t-><--n--><--n-->

*/
#define TOOM_63_MUL_N_REC(p, a, b, n, ws)		\
  do {	mpn_mul_n (p, a, b, n);				\
  } while (0)

#define TOOM_63_MUL_REC(p, a, na, b, nb, ws)		\
  do {	mpn_mul (p, a, na, b, nb);			\
  } while (0)

void
mpn_toom63_mul (mp_ptr pp,
		mp_srcptr ap, mp_size_t an,
		mp_srcptr bp, mp_size_t bn, mp_ptr scratch)
{
  mp_size_t n, s, t;
  mp_limb_t cy;
  int sign;

  /***************************** decomposition *******************************/
#define a5  (ap + 5 * n)
#define b0  (bp + 0 * n)
#define b1  (bp + 1 * n)
#define b2  (bp + 2 * n)

  ASSERT (an >= bn);
  n = 1 + (an >= 2 * bn ? (an - 1) / (size_t) 6 : (bn - 1) / (size_t) 3);

  s = an - 5 * n;
  t = bn - 2 * n;

  ASSERT (0 < s && s <= n);
  ASSERT (0 < t && t <= n);
  /* WARNING! it assumes s+t>=n */
  ASSERT ( s + t >= n );
  ASSERT ( s + t > 4);
  /* WARNING! it assumes n>1 */
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
/*   if (scratch == NULL) scratch = TMP_SALLOC_LIMBS (9 * n + 3); */

  /********************** evaluation and recursive calls *********************/
  /* $\pm4$ */
  sign = mpn_toom_eval_pm2exp (v2, v0, 5, ap, n, s, 2, pp);
  pp[n] = mpn_lshift (pp, b1, n, 2); /* 4b1 */
  /* FIXME: use addlsh */
  v3[t] = mpn_lshift (v3, b2, t, 4);/* 16b2 */
  if ( n == t )
    v3[n]+= mpn_add_n (v3, v3, b0, n); /* 16b2+b0 */
  else
    v3[n] = mpn_add (v3, b0, n, v3, t+1); /* 16b2+b0 */
  sign ^= abs_sub_add_n (v1, v3, pp, n + 1);
  TOOM_63_MUL_N_REC(pp, v0, v1, n + 1, ws); /* A(-4)*B(-4) */
  TOOM_63_MUL_N_REC(r3, v2, v3, n + 1, ws); /* A(+4)*B(+4) */
  mpn_toom_couple_handling (r3, 2*n+1, pp, sign, n, 2, 4);

  /* $\pm1$ */
  sign = mpn_toom_eval_pm1 (v2, v0, 5, ap, n, s,    pp);
  /* Compute bs1 and bsm1. Code taken from toom33 */
  cy = mpn_add (ws, b0, n, b2, t);
#if HAVE_NATIVE_mpn_add_n_sub_n
  if (cy == 0 && mpn_cmp (ws, b1, n) < 0)
    {
      cy = mpn_add_n_sub_n (v3, v1, b1, ws, n);
      v3[n] = cy >> 1;
      v1[n] = 0;
      sign = ~sign;
    }
  else
    {
      mp_limb_t cy2;
      cy2 = mpn_add_n_sub_n (v3, v1, ws, b1, n);
      v3[n] = cy + (cy2 >> 1);
      v1[n] = cy - (cy2 & 1);
    }
#else
  v3[n] = cy + mpn_add_n (v3, ws, b1, n);
  if (cy == 0 && mpn_cmp (ws, b1, n) < 0)
    {
      mpn_sub_n (v1, b1, ws, n);
      v1[n] = 0;
      sign = ~sign;
    }
  else
    {
      cy -= mpn_sub_n (v1, ws, b1, n);
      v1[n] = cy;
    }
#endif
  TOOM_63_MUL_N_REC(pp, v0, v1, n + 1, ws); /* A(-1)*B(-1) */
  TOOM_63_MUL_N_REC(r7, v2, v3, n + 1, ws); /* A(1)*B(1) */
  mpn_toom_couple_handling (r7, 2*n+1, pp, sign, n, 0, 0);

  /* $\pm2$ */
  sign = mpn_toom_eval_pm2 (v2, v0, 5, ap, n, s, pp);
  pp[n] = mpn_lshift (pp, b1, n, 1); /* 2b1 */
  /* FIXME: use addlsh or addlsh2 */
  v3[t] = mpn_lshift (v3, b2, t, 2);/* 4b2 */
  if ( n == t )
    v3[n]+= mpn_add_n (v3, v3, b0, n); /* 4b2+b0 */
  else
    v3[n] = mpn_add (v3, b0, n, v3, t+1); /* 4b2+b0 */
  sign ^= abs_sub_add_n (v1, v3, pp, n + 1);
  TOOM_63_MUL_N_REC(pp, v0, v1, n + 1, ws); /* A(-2)*B(-2) */
  TOOM_63_MUL_N_REC(r5, v2, v3, n + 1, ws); /* A(+2)*B(+2) */
  mpn_toom_couple_handling (r5, 2*n+1, pp, sign, n, 1, 2);

  /* A(0)*B(0) */
  TOOM_63_MUL_N_REC(pp, ap, bp, n, ws);

  /* Infinity */
  if (s > t) {
    TOOM_63_MUL_REC(r1, a5, s, b2, t, ws);
  } else {
    TOOM_63_MUL_REC(r1, b2, t, a5, s, ws);
  };

  mpn_toom_interpolate_8pts (pp, n, r3, r7, s + t, ws);

#undef a5
#undef b0
#undef b1
#undef b2
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
