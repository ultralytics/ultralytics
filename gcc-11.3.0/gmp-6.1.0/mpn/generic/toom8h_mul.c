/* Implementation of the multiplication algorithm for Toom-Cook 8.5-way.

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
#error Not implemented.
#endif

#if GMP_NUMB_BITS < 43
#define BIT_CORRECTION 1
#define CORRECTION_BITS GMP_NUMB_BITS
#else
#define BIT_CORRECTION 0
#define CORRECTION_BITS 0
#endif


#if TUNE_PROGRAM_BUILD
#define MAYBE_mul_basecase 1
#define MAYBE_mul_toom22   1
#define MAYBE_mul_toom33   1
#define MAYBE_mul_toom44   1
#define MAYBE_mul_toom8h   1
#else
#define MAYBE_mul_basecase						\
  (MUL_TOOM8H_THRESHOLD < 8 * MUL_TOOM22_THRESHOLD)
#define MAYBE_mul_toom22						\
  (MUL_TOOM8H_THRESHOLD < 8 * MUL_TOOM33_THRESHOLD)
#define MAYBE_mul_toom33						\
  (MUL_TOOM8H_THRESHOLD < 8 * MUL_TOOM44_THRESHOLD)
#define MAYBE_mul_toom44						\
  (MUL_TOOM8H_THRESHOLD < 8 * MUL_TOOM6H_THRESHOLD)
#define MAYBE_mul_toom8h						\
  (MUL_FFT_THRESHOLD >= 8 * MUL_TOOM8H_THRESHOLD)
#endif

#define TOOM8H_MUL_N_REC(p, a, b, f, p2, a2, b2, n, ws)			\
  do {									\
    if (MAYBE_mul_basecase						\
	&& BELOW_THRESHOLD (n, MUL_TOOM22_THRESHOLD)) {			\
      mpn_mul_basecase (p, a, n, b, n);					\
      if (f) mpn_mul_basecase (p2, a2, n, b2, n);			\
    } else if (MAYBE_mul_toom22						\
	     && BELOW_THRESHOLD (n, MUL_TOOM33_THRESHOLD)) {		\
      mpn_toom22_mul (p, a, n, b, n, ws);				\
      if (f) mpn_toom22_mul (p2, a2, n, b2, n, ws);			\
    } else if (MAYBE_mul_toom33						\
	     && BELOW_THRESHOLD (n, MUL_TOOM44_THRESHOLD)) {		\
      mpn_toom33_mul (p, a, n, b, n, ws);				\
      if (f) mpn_toom33_mul (p2, a2, n, b2, n, ws);			\
    } else if (MAYBE_mul_toom44						\
	     && BELOW_THRESHOLD (n, MUL_TOOM6H_THRESHOLD)) {		\
      mpn_toom44_mul (p, a, n, b, n, ws);				\
      if (f) mpn_toom44_mul (p2, a2, n, b2, n, ws);			\
    } else if (! MAYBE_mul_toom8h					\
	     || BELOW_THRESHOLD (n, MUL_TOOM8H_THRESHOLD)) {		\
      mpn_toom6h_mul (p, a, n, b, n, ws);				\
      if (f) mpn_toom6h_mul (p2, a2, n, b2, n, ws);			\
    } else {								\
      mpn_toom8h_mul (p, a, n, b, n, ws);				\
      if (f) mpn_toom8h_mul (p2, a2, n, b2, n, ws);			\
    }									\
  } while (0)

#define TOOM8H_MUL_REC(p, a, na, b, nb, ws)		\
  do { mpn_mul (p, a, na, b, nb); } while (0)

/* Toom-8.5 , compute the product {pp,an+bn} <- {ap,an} * {bp,bn}
   With: an >= bn >= 86, an*5 <  bn * 11.
   It _may_ work with bn<=?? and bn*?? < an*? < bn*??

   Evaluate in: infinity, +8,-8,+4,-4,+2,-2,+1,-1,+1/2,-1/2,+1/4,-1/4,+1/8,-1/8,0.
*/
/* Estimate on needed scratch:
   S(n) <= (n+7)\8*13+5+MAX(S((n+7)\8),1+2*(n+7)\8),
   since n>80; S(n) <= ceil(log(n/10)/log(8))*(13+5)+n*15\8 < n*15\8 + lg2(n)*6
 */

void
mpn_toom8h_mul   (mp_ptr pp,
		  mp_srcptr ap, mp_size_t an,
		  mp_srcptr bp, mp_size_t bn, mp_ptr scratch)
{
  mp_size_t n, s, t;
  int p, q, half;
  int sign;

  /***************************** decomposition *******************************/

  ASSERT (an >= bn);
  /* Can not handle too small operands */
  ASSERT (bn >= 86);
  /* Can not handle too much unbalancement */
  ASSERT (an <= bn*4);
  ASSERT (GMP_NUMB_BITS > 11*3 || an*4 <= bn*11);
  ASSERT (GMP_NUMB_BITS > 10*3 || an*1 <= bn* 2);
  ASSERT (GMP_NUMB_BITS >  9*3 || an*2 <= bn* 3);

  /* Limit num/den is a rational number between
     (16/15)^(log(6)/log(2*6-1)) and (16/15)^(log(8)/log(2*8-1))             */
#define LIMIT_numerator (21)
#define LIMIT_denominat (20)

  if (LIKELY (an == bn) || an * (LIMIT_denominat>>1) < LIMIT_numerator * (bn>>1) ) /* is 8*... < 8*... */
    {
      half = 0;
      n = 1 + ((an - 1)>>3);
      p = q = 7;
      s = an - 7 * n;
      t = bn - 7 * n;
    }
  else
    {
      if (an * 13 < 16 * bn) /* (an*7*LIMIT_numerator<LIMIT_denominat*9*bn) */
	{ p = 9; q = 8; }
      else if (GMP_NUMB_BITS <= 9*3 ||
	       an *(LIMIT_denominat>>1) < (LIMIT_numerator/7*9) * (bn>>1))
	{ p = 9; q = 7; }
      else if (an * 10 < 33 * (bn>>1)) /* (an*3*LIMIT_numerator<LIMIT_denominat*5*bn) */
	{ p =10; q = 7; }
      else if (GMP_NUMB_BITS <= 10*3 ||
	       an * (LIMIT_denominat/5) < (LIMIT_numerator/3) * bn)
	{ p =10; q = 6; }
      else if (an * 6 < 13 * bn) /*(an * 5 * LIMIT_numerator < LIMIT_denominat *11 * bn)*/
	{ p =11; q = 6; }
      else if (GMP_NUMB_BITS <= 11*3 ||
	       an * 4 < 9 * bn)
	{ p =11; q = 5; }
      else if (an *(LIMIT_numerator/3) < LIMIT_denominat * bn)  /* is 4*... <12*... */
	{ p =12; q = 5; }
      else if (GMP_NUMB_BITS <= 12*3 ||
	       an * 9 < 28 * bn )  /* is 4*... <12*... */
	{ p =12; q = 4; }
      else
	{ p =13; q = 4; }

      half = (p+q)&1;
      n = 1 + (q * an >= p * bn ? (an - 1) / (size_t) p : (bn - 1) / (size_t) q);
      p--; q--;

      s = an - p * n;
      t = bn - q * n;

      if(half) { /* Recover from badly chosen splitting */
	if (UNLIKELY (s<1)) {p--; s+=n; half=0;}
	else if (UNLIKELY (t<1)) {q--; t+=n; half=0;}
      }
    }
#undef LIMIT_numerator
#undef LIMIT_denominat

  ASSERT (0 < s && s <= n);
  ASSERT (0 < t && t <= n);
  ASSERT (half || s + t > 3);
  ASSERT (n > 2);

#define   r6    (pp + 3 * n)			/* 3n+1 */
#define   r4    (pp + 7 * n)			/* 3n+1 */
#define   r2    (pp +11 * n)			/* 3n+1 */
#define   r0    (pp +15 * n)			/* s+t <= 2*n */
#define   r7    (scratch)			/* 3n+1 */
#define   r5    (scratch + 3 * n + 1)		/* 3n+1 */
#define   r3    (scratch + 6 * n + 2)		/* 3n+1 */
#define   r1    (scratch + 9 * n + 3)		/* 3n+1 */
#define   v0    (pp +11 * n)			/* n+1 */
#define   v1    (pp +12 * n+1)			/* n+1 */
#define   v2    (pp +13 * n+2)			/* n+1 */
#define   v3    (scratch +12 * n + 4)		/* n+1 */
#define   wsi   (scratch +12 * n + 4)		/* 3n+1 */
#define   wse   (scratch +13 * n + 5)		/* 2n+1 */

  /* Alloc also 3n+1 limbs for wsi... toom_interpolate_16pts may
     need all of them  */
/*   if (scratch == NULL) */
/*     scratch = TMP_SALLOC_LIMBS(mpn_toom8_sqr_itch(n * 8)); */
  ASSERT (15 * n + 6 <= mpn_toom8h_mul_itch (an, bn));
  ASSERT (15 * n + 6 <= mpn_toom8_sqr_itch (n * 8));

  /********************** evaluation and recursive calls *********************/

  /* $\pm1/8$ */
  sign = mpn_toom_eval_pm2rexp (v2, v0, p, ap, n, s, 3, pp) ^
	 mpn_toom_eval_pm2rexp (v3, v1, q, bp, n, t, 3, pp);
  /* A(-1/8)*B(-1/8)*8^. */ /* A(+1/8)*B(+1/8)*8^. */
  TOOM8H_MUL_N_REC(pp, v0, v1, 2, r7, v2, v3, n + 1, wse);
  mpn_toom_couple_handling (r7, 2 * n + 1 + BIT_CORRECTION, pp, sign, n, 3*(1+half), 3*(half));

  /* $\pm1/4$ */
  sign = mpn_toom_eval_pm2rexp (v2, v0, p, ap, n, s, 2, pp) ^
	 mpn_toom_eval_pm2rexp (v3, v1, q, bp, n, t, 2, pp);
  /* A(-1/4)*B(-1/4)*4^. */ /* A(+1/4)*B(+1/4)*4^. */
  TOOM8H_MUL_N_REC(pp, v0, v1, 2, r5, v2, v3, n + 1, wse);
  mpn_toom_couple_handling (r5, 2 * n + 1, pp, sign, n, 2*(1+half), 2*(half));

  /* $\pm2$ */
  sign = mpn_toom_eval_pm2 (v2, v0, p, ap, n, s, pp) ^
	 mpn_toom_eval_pm2 (v3, v1, q, bp, n, t, pp);
  /* A(-2)*B(-2) */ /* A(+2)*B(+2) */
  TOOM8H_MUL_N_REC(pp, v0, v1, 2, r3, v2, v3, n + 1, wse);
  mpn_toom_couple_handling (r3, 2 * n + 1, pp, sign, n, 1, 2);

  /* $\pm8$ */
  sign = mpn_toom_eval_pm2exp (v2, v0, p, ap, n, s, 3, pp) ^
	 mpn_toom_eval_pm2exp (v3, v1, q, bp, n, t, 3, pp);
  /* A(-8)*B(-8) */ /* A(+8)*B(+8) */
  TOOM8H_MUL_N_REC(pp, v0, v1, 2, r1, v2, v3, n + 1, wse);
  mpn_toom_couple_handling (r1, 2 * n + 1 + BIT_CORRECTION, pp, sign, n, 3, 6);

  /* $\pm1/2$ */
  sign = mpn_toom_eval_pm2rexp (v2, v0, p, ap, n, s, 1, pp) ^
	 mpn_toom_eval_pm2rexp (v3, v1, q, bp, n, t, 1, pp);
  /* A(-1/2)*B(-1/2)*2^. */ /* A(+1/2)*B(+1/2)*2^. */
  TOOM8H_MUL_N_REC(pp, v0, v1, 2, r6, v2, v3, n + 1, wse);
  mpn_toom_couple_handling (r6, 2 * n + 1, pp, sign, n, 1+half, half);

  /* $\pm1$ */
  sign = mpn_toom_eval_pm1 (v2, v0, p, ap, n, s,    pp);
  if (GMP_NUMB_BITS > 12*3 && UNLIKELY (q == 3))
    sign ^= mpn_toom_eval_dgr3_pm1 (v3, v1, bp, n, t,    pp);
  else
    sign ^= mpn_toom_eval_pm1 (v3, v1, q, bp, n, t,    pp);
  /* A(-1)*B(-1) */ /* A(1)*B(1) */
  TOOM8H_MUL_N_REC(pp, v0, v1, 2, r4, v2, v3, n + 1, wse);
  mpn_toom_couple_handling (r4, 2 * n + 1, pp, sign, n, 0, 0);

  /* $\pm4$ */
  sign = mpn_toom_eval_pm2exp (v2, v0, p, ap, n, s, 2, pp) ^
	 mpn_toom_eval_pm2exp (v3, v1, q, bp, n, t, 2, pp);
  /* A(-4)*B(-4) */ /* A(+4)*B(+4) */
  TOOM8H_MUL_N_REC(pp, v0, v1, 2, r2, v2, v3, n + 1, wse);
  mpn_toom_couple_handling (r2, 2 * n + 1, pp, sign, n, 2, 4);

#undef v0
#undef v1
#undef v2
#undef v3
#undef wse

  /* A(0)*B(0) */
  TOOM8H_MUL_N_REC(pp, ap, bp, 0, pp, ap, bp, n, wsi);

  /* Infinity */
  if (UNLIKELY (half != 0)) {
    if (s > t) {
      TOOM8H_MUL_REC(r0, ap + p * n, s, bp + q * n, t, wsi);
    } else {
      TOOM8H_MUL_REC(r0, bp + q * n, t, ap + p * n, s, wsi);
    };
  };

  mpn_toom_interpolate_16pts (pp, r1, r3, r5, r7, n, s+t, half, wsi);

#undef r0
#undef r1
#undef r2
#undef r3
#undef r4
#undef r5
#undef r6
#undef wsi
}

#undef TOOM8H_MUL_N_REC
#undef TOOM8H_MUL_REC
#undef MAYBE_mul_basecase
#undef MAYBE_mul_toom22
#undef MAYBE_mul_toom33
#undef MAYBE_mul_toom44
#undef MAYBE_mul_toom8h
