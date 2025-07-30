/* mpn_div_qr_1u_pi2.

   THIS FILE CONTAINS AN INTERNAL FUNCTION WITH A MUTABLE INTERFACE.  IT IS
   ONLY SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2013 Free Software Foundation, Inc.

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

/* ISSUES:

   * Can we really use the high pi2 inverse limb for udiv_qrnnd_preinv?

   * Are there any problems with generating n quotient limbs in the q area?  It
     surely simplifies things.

   * Not yet adequately tested.
*/

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"

/* Define some longlong.h-style macros, but for wider operations.
   * add_sssaaaa is like longlong.h's add_ssaaaa but propagating
     carry-out into an additional sum operand.
*/
#if defined (__GNUC__)  && ! defined (__INTEL_COMPILER) && ! defined (NO_ASM)

#if HAVE_HOST_CPU_FAMILY_x86 && W_TYPE_SIZE == 32
#define add_sssaaaa(s2, s1, s0, a1, a0, b1, b0)				\
  __asm__ ("add\t%7, %k2\n\tadc\t%5, %k1\n\tadc\t$0, %k0"		\
	   : "=r" (s2), "=&r" (s1), "=&r" (s0)				\
	   : "0"  ((USItype)(s2)),					\
	     "1"  ((USItype)(a1)), "g" ((USItype)(b1)),			\
	     "%2" ((USItype)(a0)), "g" ((USItype)(b0)))
#endif

#if defined (__amd64__) && W_TYPE_SIZE == 64
#define add_sssaaaa(s2, s1, s0, a1, a0, b1, b0)				\
  __asm__ ("add\t%7, %q2\n\tadc\t%5, %q1\n\tadc\t$0, %q0"		\
	   : "=r" (s2), "=&r" (s1), "=&r" (s0)				\
	   : "0"  ((UDItype)(s2)),					\
	     "1"  ((UDItype)(a1)), "rme" ((UDItype)(b1)),		\
	     "%2" ((UDItype)(a0)), "rme" ((UDItype)(b0)))
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

struct precomp_div_1_pi2
{
  mp_limb_t dip[2];
  mp_limb_t d;
  int norm_cnt;
};

mp_limb_t
mpn_div_qr_1n_pi2 (mp_ptr qp,
		   mp_srcptr up, mp_size_t un,
		   struct precomp_div_1_pi2 *pd)
{
  mp_limb_t most_significant_q_limb;
  mp_size_t i;
  mp_limb_t r, u2, u1, u0;
  mp_limb_t d0, di1, di0;
  mp_limb_t q3a, q2a, q2b, q1b, q2c, q1c, q1d, q0d;
  mp_limb_t cnd;

  ASSERT (un >= 2);
  ASSERT ((pd->d & GMP_NUMB_HIGHBIT) != 0);
  ASSERT (! MPN_OVERLAP_P (qp, un-2, up, un) || qp+2 >= up);
  ASSERT_MPN (up, un);

#define q3 q3a
#define q2 q2b
#define q1 q1b

  up += un - 3;
  r = up[2];
  d0 = pd->d;

  most_significant_q_limb = (r >= d0);
  r -= d0 & -most_significant_q_limb;

  qp += un - 3;
  qp[2] = most_significant_q_limb;

  di1 = pd->dip[1];
  di0 = pd->dip[0];

  for (i = un - 3; i >= 0; i -= 2)
    {
      u2 = r;
      u1 = up[1];
      u0 = up[0];

      /* Dividend in {r,u1,u0} */

      umul_ppmm (q1d,q0d, u1, di0);
      umul_ppmm (q2b,q1b, u1, di1);
      q2b++;				/* cannot spill */
      add_sssaaaa (r,q2b,q1b, q2b,q1b, u1,u0);

      umul_ppmm (q2c,q1c, u2,  di0);
      add_sssaaaa (r,q2b,q1b, q2b,q1b, q2c,q1c);
      umul_ppmm (q3a,q2a, u2, di1);

      add_sssaaaa (r,q2b,q1b, q2b,q1b, q2a,q1d);

      q3 += r;

      r = u0 - q2 * d0;

      cnd = (r >= q1);
      r += d0 & -cnd;
      sub_ddmmss (q3,q2,  q3,q2,  0,cnd);

      if (UNLIKELY (r >= d0))
	{
	  r -= d0;
	  add_ssaaaa (q3,q2,  q3,q2,  0,1);
	}

      qp[0] = q2;
      qp[1] = q3;

      up -= 2;
      qp -= 2;
    }

  if ((un & 1) == 0)
    {
      u2 = r;
      u1 = up[1];

      udiv_qrnnd_preinv (q3, r, u2, u1, d0, di1);
      qp[1] = q3;
    }

  return r;

#undef q3
#undef q2
#undef q1
}
