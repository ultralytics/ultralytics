/* mpn_div_qr_2 -- Divide natural numbers, producing both remainder and
   quotient.  The divisor is two limbs.

   Contributed to the GNU project by Torbjorn Granlund and Niels Möller

   THIS FILE CONTAINS INTERNAL FUNCTIONS WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.


Copyright 1993-1996, 1999-2002, 2011 Free Software Foundation, Inc.

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

#ifndef DIV_QR_2_PI2_THRESHOLD
/* Disabled unless explicitly tuned. */
#define DIV_QR_2_PI2_THRESHOLD MP_LIMB_T_MAX
#endif

#ifndef SANITY_CHECK
#define SANITY_CHECK 0
#endif

/* Define some longlong.h-style macros, but for wider operations.
   * add_sssaaaa is like longlong.h's add_ssaaaa but the propagating
     carry-out into an additional sum operand.
   * add_csaac accepts two addends and a carry in, and generates a sum
     and a carry out.  A little like a "full adder".
*/
#if defined (__GNUC__)  && ! defined (__INTEL_COMPILER) && ! defined (NO_ASM)

#if HAVE_HOST_CPU_FAMILY_x86 && W_TYPE_SIZE == 32
#define add_sssaaaa(s2, s1, s0, a1, a0, b1, b0)				\
  __asm__ ("add\t%7, %k2\n\tadc\t%5, %k1\n\tadc\t$0, %k0"		\
	   : "=r" (s2), "=&r" (s1), "=&r" (s0)				\
	   : "0"  ((USItype)(s2)),					\
	     "1"  ((USItype)(a1)), "g" ((USItype)(b1)),			\
	     "%2" ((USItype)(a0)), "g" ((USItype)(b0)))
#define add_csaac(co, s, a, b, ci)					\
  __asm__ ("bt\t$0, %2\n\tadc\t%5, %k1\n\tadc\t%k0, %k0"		\
	   : "=r" (co), "=r" (s)					\
	   : "rm"  ((USItype)(ci)), "0" (CNST_LIMB(0)),			\
	     "%1" ((USItype)(a)), "g" ((USItype)(b)))
#endif

#if defined (__amd64__) && W_TYPE_SIZE == 64
#define add_sssaaaa(s2, s1, s0, a1, a0, b1, b0)				\
  __asm__ ("add\t%7, %q2\n\tadc\t%5, %q1\n\tadc\t$0, %q0"		\
	   : "=r" (s2), "=&r" (s1), "=&r" (s0)				\
	   : "0"  ((UDItype)(s2)),					\
	     "1"  ((UDItype)(a1)), "rme" ((UDItype)(b1)),		\
	     "%2" ((UDItype)(a0)), "rme" ((UDItype)(b0)))
#define add_csaac(co, s, a, b, ci)					\
  __asm__ ("bt\t$0, %2\n\tadc\t%5, %q1\n\tadc\t%q0, %q0"		\
	   : "=r" (co), "=r" (s)					\
	   : "rm"  ((UDItype)(ci)), "0" (CNST_LIMB(0)),			\
	     "%1" ((UDItype)(a)), "g" ((UDItype)(b)))
#endif

#if HAVE_HOST_CPU_FAMILY_powerpc && !defined (_LONG_LONG_LIMB)
/* This works fine for 32-bit and 64-bit limbs, except for 64-bit limbs with a
   processor running in 32-bit mode, since the carry flag then gets the 32-bit
   carry.  */
#define add_sssaaaa(s2, s1, s0, a1, a0, b1, b0)				\
  __asm__ ("add%I7c\t%2,%6,%7\n\tadde\t%1,%4,%5\n\taddze\t%0,%0"	\
	   : "=r" (s2), "=&r" (s1), "=&r" (s0)				\
	   : "r"  (s2), "r"  (a1), "r" (b1), "%r" (a0), "rI" (b0))
#endif

#endif /* __GNUC__ */

#ifndef add_sssaaaa
#define add_sssaaaa(s2, s1, s0, a1, a0, b1, b0)				\
  do {									\
    UWtype __s0, __s1, __c0, __c1;					\
    __s0 = (a0) + (b0);							\
    __s1 = (a1) + (b1);							\
    __c0 = __s0 < (a0);							\
    __c1 = __s1 < (a1);							\
    (s0) = __s0;							\
    __s1 = __s1 + __c0;							\
    (s1) = __s1;							\
    (s2) += __c1 + (__s1 < __c0);					\
  } while (0)
#endif

#ifndef add_csaac
#define add_csaac(co, s, a, b, ci)					\
  do {									\
    UWtype __s, __c;							\
    __s = (a) + (b);							\
    __c = __s < (a);							\
    __s = __s + (ci);							\
    (s) = __s;								\
    (co) = __c + (__s < (ci));						\
  } while (0)
#endif

/* Typically used with r1, r0 same as n3, n2. Other types of overlap
   between inputs and outputs are not supported. */
#define udiv_qr_4by2(q1,q0, r1,r0, n3,n2,n1,n0, d1,d0, di1,di0)		\
  do {									\
    mp_limb_t _q3, _q2a, _q2, _q1, _q2c, _q1c, _q1d, _q0;		\
    mp_limb_t _t1, _t0;							\
    mp_limb_t _c, _mask;						\
									\
    umul_ppmm (_q3,_q2a, n3, di1);					\
    umul_ppmm (_q2,_q1, n2, di1);					\
    umul_ppmm (_q2c,_q1c, n3, di0);					\
    add_sssaaaa (_q3,_q2,_q1, _q2,_q1, _q2c,_q1c);			\
    umul_ppmm (_q1d,_q0, n2, di0);					\
    add_sssaaaa (_q3,_q2,_q1, _q2,_q1, _q2a,_q1d);			\
									\
    add_ssaaaa (r1, r0, n3, n2, CNST_LIMB(0), CNST_LIMB(1));		\
									\
    /* [q3,q2,q1,q0] += [n3,n3,n1,n0] */				\
    add_csaac (_c, _q0, _q0, n0, CNST_LIMB(0));				\
    add_csaac (_c, _q1, _q1, n1, _c);					\
    add_csaac (_c, _q2, _q2, r0, _c);					\
    _q3 = _q3 + r1 + _c;						\
									\
    umul_ppmm (_t1,_t0, _q2, d0);					\
    _t1 += _q2 * d1 + _q3 * d0;						\
									\
    sub_ddmmss (r1, r0, n1, n0, _t1, _t0);				\
									\
    _mask = -(mp_limb_t) ((r1 >= _q1) & ((r1 > _q1) | (r0 >= _q0)));  /* (r1,r0) >= (q1,q0) */  \
    add_ssaaaa (r1, r0, r1, r0, d1 & _mask, d0 & _mask);		\
    sub_ddmmss (_q3, _q2, _q3, _q2, CNST_LIMB(0), -_mask);		\
									\
    if (UNLIKELY (r1 >= d1))						\
      {									\
	if (r1 > d1 || r0 >= d0)					\
	  {								\
	    sub_ddmmss (r1, r0, r1, r0, d1, d0);			\
	    add_ssaaaa (_q3, _q2, _q3, _q2, CNST_LIMB(0), CNST_LIMB(1));\
	  }								\
      }									\
    (q1) = _q3;								\
    (q0) = _q2;								\
  } while (0)

static void
invert_4by2 (mp_ptr di, mp_limb_t d1, mp_limb_t d0)
{
  mp_limb_t v1, v0, p1, t1, t0, p0, mask;
  invert_limb (v1, d1);
  p1 = d1 * v1;
  /* <1, v1> * d1 = <B-1, p1> */
  p1 += d0;
  if (p1 < d0)
    {
      v1--;
      mask = -(mp_limb_t) (p1 >= d1);
      p1 -= d1;
      v1 += mask;
      p1 -= mask & d1;
    }
  /* <1, v1> * d1 + d0 = <B-1, p1> */
  umul_ppmm (t1, p0, d0, v1);
  p1 += t1;
  if (p1 < t1)
    {
      if (UNLIKELY (p1 >= d1))
	{
	  if (p1 > d1 || p0 >= d0)
	    {
	      sub_ddmmss (p1, p0, p1, p0, d1, d0);
	      v1--;
	    }
	}
      sub_ddmmss (p1, p0, p1, p0, d1, d0);
      v1--;
    }
  /* Now v1 is the 3/2 inverse, <1, v1> * <d1, d0> = <B-1, p1, p0>,
   * with <p1, p0> + <d1, d0> >= B^2.
   *
   * The 4/2 inverse is (B^4 - 1) / <d1, d0> = <1, v1, v0>. The
   * partial remainder after <1, v1> is
   *
   * B^4 - 1 - B <1, v1> <d1, d0> = <B-1, B-1, B-1, B-1> - <B-1, p1, p0, 0>
   *                              = <~p1, ~p0, B-1>
   */
  udiv_qr_3by2 (v0, t1, t0, ~p1, ~p0, MP_LIMB_T_MAX, d1, d0, v1);
  di[0] = v0;
  di[1] = v1;

#if SANITY_CHECK
  {
    mp_limb_t tp[4];
    mp_limb_t dp[2];
    dp[0] = d0;
    dp[1] = d1;
    mpn_mul_n (tp, dp, di, 2);
    ASSERT_ALWAYS (mpn_add_n (tp+2, tp+2, dp, 2) == 0);
    ASSERT_ALWAYS (tp[2] == MP_LIMB_T_MAX);
    ASSERT_ALWAYS (tp[3] == MP_LIMB_T_MAX);
    ASSERT_ALWAYS (mpn_add_n (tp, tp, dp, 2) == 1);
  }
#endif
}

static mp_limb_t
mpn_div_qr_2n_pi2 (mp_ptr qp, mp_ptr rp, mp_srcptr np, mp_size_t nn,
		   mp_limb_t d1, mp_limb_t d0, mp_limb_t di1, mp_limb_t di0)
{
  mp_limb_t qh;
  mp_size_t i;
  mp_limb_t r1, r0;

  ASSERT (nn >= 2);
  ASSERT (d1 & GMP_NUMB_HIGHBIT);

  r1 = np[nn-1];
  r0 = np[nn-2];

  qh = 0;
  if (r1 >= d1 && (r1 > d1 || r0 >= d0))
    {
#if GMP_NAIL_BITS == 0
      sub_ddmmss (r1, r0, r1, r0, d1, d0);
#else
      r0 = r0 - d0;
      r1 = r1 - d1 - (r0 >> GMP_LIMB_BITS - 1);
      r0 &= GMP_NUMB_MASK;
#endif
      qh = 1;
    }

  for (i = nn - 2; i >= 2; i -= 2)
    {
      mp_limb_t n1, n0, q1, q0;
      n1 = np[i-1];
      n0 = np[i-2];
      udiv_qr_4by2 (q1, q0, r1, r0, r1, r0, n1, n0, d1, d0, di1, di0);
      qp[i-1] = q1;
      qp[i-2] = q0;
    }

  if (i > 0)
    {
      mp_limb_t q;
      udiv_qr_3by2 (q, r1, r0, r1, r0, np[0], d1, d0, di1);
      qp[0] = q;
    }
  rp[1] = r1;
  rp[0] = r0;

  return qh;
}


/* Divide num {np,nn} by den {dp,2} and write the nn-2 least
   significant quotient limbs at qp and the 2 long remainder at np.
   Return the most significant limb of the quotient.

   Preconditions:
   1. qp must either not overlap with the input operands at all, or
      qp >= np + 2 must hold true.  (This means that it's possible to put
      the quotient in the high part of {np,nn}, right above the remainder.
   2. nn >= 2.  */

mp_limb_t
mpn_div_qr_2 (mp_ptr qp, mp_ptr rp, mp_srcptr np, mp_size_t nn,
	      mp_srcptr dp)
{
  mp_limb_t d1;
  mp_limb_t d0;
  gmp_pi1_t dinv;

  ASSERT (nn >= 2);
  ASSERT (! MPN_OVERLAP_P (qp, nn-2, np, nn) || qp >= np + 2);
  ASSERT_MPN (np, nn);
  ASSERT_MPN (dp, 2);

  d1 = dp[1]; d0 = dp[0];

  ASSERT (d1 > 0);

  if (UNLIKELY (d1 & GMP_NUMB_HIGHBIT))
    {
      if (BELOW_THRESHOLD (nn, DIV_QR_2_PI2_THRESHOLD))
	{
	  gmp_pi1_t dinv;
	  invert_pi1 (dinv, d1, d0);
	  return mpn_div_qr_2n_pi1 (qp, rp, np, nn, d1, d0, dinv.inv32);
	}
      else
	{
	  mp_limb_t di[2];
	  invert_4by2 (di, d1, d0);
	  return mpn_div_qr_2n_pi2 (qp, rp, np, nn, d1, d0, di[1], di[0]);
	}
    }
  else
    {
      int shift;
      count_leading_zeros (shift, d1);
      d1 = (d1 << shift) | (d0 >> (GMP_LIMB_BITS - shift));
      d0 <<= shift;
      invert_pi1 (dinv, d1, d0);
      return mpn_div_qr_2u_pi1 (qp, rp, np, nn, d1, d0, shift, dinv.inv32);
    }
}
