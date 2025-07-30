/* mpn_div_qr_1n_pi1

   Contributed to the GNU project by Niels Möller

   THIS FILE CONTAINS INTERNAL FUNCTIONS WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.


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

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"

#if GMP_NAIL_BITS > 0
#error Nail bits not supported
#endif

#ifndef DIV_QR_1N_METHOD
#define DIV_QR_1N_METHOD 2
#endif

/* FIXME: Duplicated in mod_1_1.c. Move to gmp-impl.h */

#if defined (__GNUC__) && ! defined (NO_ASM)

#if HAVE_HOST_CPU_FAMILY_x86 && W_TYPE_SIZE == 32
#define add_mssaaaa(m, s1, s0, a1, a0, b1, b0)				\
  __asm__ (  "add	%6, %k2\n\t"					\
	     "adc	%4, %k1\n\t"					\
	     "sbb	%k0, %k0"					\
	   : "=r" (m), "=r" (s1), "=&r" (s0)				\
	   : "1"  ((USItype)(a1)), "g" ((USItype)(b1)),			\
	     "%2" ((USItype)(a0)), "g" ((USItype)(b0)))
#endif

#if HAVE_HOST_CPU_FAMILY_x86_64 && W_TYPE_SIZE == 64
#define add_mssaaaa(m, s1, s0, a1, a0, b1, b0)				\
  __asm__ (  "add	%6, %q2\n\t"					\
	     "adc	%4, %q1\n\t"					\
	     "sbb	%q0, %q0"					\
	   : "=r" (m), "=r" (s1), "=&r" (s0)				\
	   : "1"  ((UDItype)(a1)), "rme" ((UDItype)(b1)),		\
	     "%2" ((UDItype)(a0)), "rme" ((UDItype)(b0)))
#endif

#if defined (__sparc__) && W_TYPE_SIZE == 32
#define add_mssaaaa(m, sh, sl, ah, al, bh, bl)				\
  __asm__ (  "addcc	%r5, %6, %2\n\t"				\
	     "addxcc	%r3, %4, %1\n\t"				\
	     "subx	%%g0, %%g0, %0"					\
	   : "=r" (m), "=r" (sh), "=&r" (sl)				\
	   : "rJ" (ah), "rI" (bh), "%rJ" (al), "rI" (bl)		\
	 __CLOBBER_CC)
#endif

#if defined (__sparc__) && W_TYPE_SIZE == 64
#define add_mssaaaa(m, sh, sl, ah, al, bh, bl)				\
  __asm__ (  "addcc	%r5, %6, %2\n\t"				\
	     "addccc	%r7, %8, %%g0\n\t"				\
	     "addccc	%r3, %4, %1\n\t"				\
	     "clr	%0\n\t"						\
	     "movcs	%%xcc, -1, %0"					\
	   : "=r" (m), "=r" (sh), "=&r" (sl)				\
	   : "rJ" (ah), "rI" (bh), "%rJ" (al), "rI" (bl),		\
	     "rJ" ((al) >> 32), "rI" ((bl) >> 32)			\
	 __CLOBBER_CC)
#if __VIS__ >= 0x300
#undef add_mssaaaa
#define add_mssaaaa(m, sh, sl, ah, al, bh, bl)				\
  __asm__ (  "addcc	%r5, %6, %2\n\t"				\
	     "addxccc	%r3, %4, %1\n\t"				\
	     "clr	%0\n\t"						\
	     "movcs	%%xcc, -1, %0"					\
	   : "=r" (m), "=r" (sh), "=&r" (sl)				\
	   : "rJ" (ah), "rI" (bh), "%rJ" (al), "rI" (bl)		\
	 __CLOBBER_CC)
#endif
#endif

#if HAVE_HOST_CPU_FAMILY_powerpc && !defined (_LONG_LONG_LIMB)
/* This works fine for 32-bit and 64-bit limbs, except for 64-bit limbs with a
   processor running in 32-bit mode, since the carry flag then gets the 32-bit
   carry.  */
#define add_mssaaaa(m, s1, s0, a1, a0, b1, b0)				\
  __asm__ (  "add%I6c	%2, %5, %6\n\t"					\
	     "adde	%1, %3, %4\n\t"					\
	     "subfe	%0, %0, %0\n\t"					\
	     "nor	%0, %0, %0"					\
	   : "=r" (m), "=r" (s1), "=&r" (s0)				\
	   : "r"  (a1), "r" (b1), "%r" (a0), "rI" (b0))
#endif

#if defined (__s390x__) && W_TYPE_SIZE == 64
#define add_mssaaaa(m, s1, s0, a1, a0, b1, b0)				\
  __asm__ (  "algr	%2, %6\n\t"					\
	     "alcgr	%1, %4\n\t"					\
	     "lghi	%0, 0\n\t"					\
	     "alcgr	%0, %0\n\t"					\
	     "lcgr	%0, %0"						\
	   : "=r" (m), "=r" (s1), "=&r" (s0)				\
	   : "1"  ((UDItype)(a1)), "r" ((UDItype)(b1)),			\
	     "%2" ((UDItype)(a0)), "r" ((UDItype)(b0)) __CLOBBER_CC)
#endif

#if defined (__arm__) && !defined (__thumb__) && W_TYPE_SIZE == 32
#define add_mssaaaa(m, sh, sl, ah, al, bh, bl)				\
  __asm__ (  "adds	%2, %5, %6\n\t"					\
	     "adcs	%1, %3, %4\n\t"					\
	     "movcc	%0, #0\n\t"					\
	     "movcs	%0, #-1"					\
	   : "=r" (m), "=r" (sh), "=&r" (sl)				\
	   : "r" (ah), "rI" (bh), "%r" (al), "rI" (bl) __CLOBBER_CC)
#endif
#endif /* defined (__GNUC__) */

#ifndef add_mssaaaa
#define add_mssaaaa(m, s1, s0, a1, a0, b1, b0)				\
  do {									\
    UWtype __s0, __s1, __c0, __c1;					\
    __s0 = (a0) + (b0);							\
    __s1 = (a1) + (b1);							\
    __c0 = __s0 < (a0);							\
    __c1 = __s1 < (a1);							\
    (s0) = __s0;							\
    __s1 = __s1 + __c0;							\
    (s1) = __s1;							\
    (m) = - (__c1 + (__s1 < __c0));					\
  } while (0)
#endif

#if DIV_QR_1N_METHOD == 1

/* Divides (uh B^n + {up, n}) by d, storing the quotient at {qp, n}.
   Requires that uh < d. */
mp_limb_t
mpn_div_qr_1n_pi1 (mp_ptr qp, mp_srcptr up, mp_size_t n, mp_limb_t uh,
		   mp_limb_t d, mp_limb_t dinv)
{
  ASSERT (n > 0);
  ASSERT (uh < d);
  ASSERT (d & GMP_NUMB_HIGHBIT);
  ASSERT (MPN_SAME_OR_SEPARATE_P (qp, up, n));

  do
    {
      mp_limb_t q, ul;

      ul = up[--n];
      udiv_qrnnd_preinv (q, uh, uh, ul, d, dinv);
      qp[n] = q;
    }
  while (n > 0);

  return uh;
}

#elif DIV_QR_1N_METHOD == 2

mp_limb_t
mpn_div_qr_1n_pi1 (mp_ptr qp, mp_srcptr up, mp_size_t n, mp_limb_t u1,
		   mp_limb_t d, mp_limb_t dinv)
{
  mp_limb_t B2;
  mp_limb_t u0, u2;
  mp_limb_t q0, q1;
  mp_limb_t p0, p1;
  mp_limb_t t;
  mp_size_t j;

  ASSERT (d & GMP_LIMB_HIGHBIT);
  ASSERT (n > 0);
  ASSERT (u1 < d);

  if (n == 1)
    {
      udiv_qrnnd_preinv (qp[0], u1, u1, up[0], d, dinv);
      return u1;
    }

  /* FIXME: Could be precomputed */
  B2 = -d*dinv;

  umul_ppmm (q1, q0, dinv, u1);
  umul_ppmm (p1, p0, B2, u1);
  q1 += u1;
  ASSERT (q1 >= u1);
  u0 = up[n-1];	/* Early read, to allow qp == up. */
  qp[n-1] = q1;

  add_mssaaaa (u2, u1, u0, u0, up[n-2], p1, p0);

  /* FIXME: Keep q1 in a variable between iterations, to reduce number
     of memory accesses. */
  for (j = n-2; j-- > 0; )
    {
      mp_limb_t q2, cy;

      /* Additions for the q update:
       *	+-------+
       *        |u1 * v |
       *        +---+---+
       *        | u1|
       *    +---+---+
       *    | 1 | v |  (conditional on u2)
       *    +---+---+
       *        | 1 |  (conditional on u0 + u2 B2 carry)
       *        +---+
       * +      | q0|
       *   -+---+---+---+
       *    | q2| q1| q0|
       *    +---+---+---+
      */
      umul_ppmm (p1, t, u1, dinv);
      add_ssaaaa (q2, q1, -u2, u2 & dinv, CNST_LIMB(0), u1);
      add_ssaaaa (q2, q1, q2, q1, CNST_LIMB(0), p1);
      add_ssaaaa (q2, q1, q2, q1, CNST_LIMB(0), q0);
      q0 = t;

      umul_ppmm (p1, p0, u1, B2);
      ADDC_LIMB (cy, u0, u0, u2 & B2);
      u0 -= (-cy) & d;

      /* Final q update */
      add_ssaaaa (q2, q1, q2, q1, CNST_LIMB(0), cy);
      qp[j+1] = q1;
      MPN_INCR_U (qp+j+2, n-j-2, q2);

      add_mssaaaa (u2, u1, u0, u0, up[j], p1, p0);
    }

  q1 = (u2 > 0);
  u1 -= (-q1) & d;

  t = (u1 >= d);
  q1 += t;
  u1 -= (-t) & d;

  udiv_qrnnd_preinv (t, u0, u1, u0, d, dinv);
  add_ssaaaa (q1, q0, q1, q0, CNST_LIMB(0), t);

  MPN_INCR_U (qp+1, n-1, q1);

  qp[0] = q0;
  return u0;
}

#else
#error Unknown DIV_QR_1N_METHOD
#endif
