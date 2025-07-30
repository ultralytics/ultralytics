/* mpn_mod_1_1p (ap, n, b, cps)
   Divide (ap,,n) by b.  Return the single-limb remainder.

   Contributed to the GNU project by Torbjorn Granlund and Niels Möller.
   Based on a suggestion by Peter L. Montgomery.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2008-2011, 2013 Free Software Foundation, Inc.

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

#ifndef MOD_1_1P_METHOD
# define MOD_1_1P_METHOD 1    /* need to make sure this is 2 for asm testing */
#endif

/* Define some longlong.h-style macros, but for wider operations.
 * add_mssaaaa is like longlong.h's add_ssaaaa, but also generates
 * carry out, in the form of a mask. */

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

#if MOD_1_1P_METHOD == 1
void
mpn_mod_1_1p_cps (mp_limb_t cps[4], mp_limb_t b)
{
  mp_limb_t bi;
  mp_limb_t B1modb, B2modb;
  int cnt;

  count_leading_zeros (cnt, b);

  b <<= cnt;
  invert_limb (bi, b);

  cps[0] = bi;
  cps[1] = cnt;

  B1modb = -b;
  if (LIKELY (cnt != 0))
    B1modb *= ((bi >> (GMP_LIMB_BITS-cnt)) | (CNST_LIMB(1) << cnt));
  ASSERT (B1modb <= b);		/* NB: not fully reduced mod b */
  cps[2] = B1modb >> cnt;

  /* In the normalized case, this can be simplified to
   *
   *   B2modb = - b * bi;
   *   ASSERT (B2modb <= b);    // NB: equality iff b = B/2
   */
  udiv_rnnd_preinv (B2modb, B1modb, CNST_LIMB(0), b, bi);
  cps[3] = B2modb >> cnt;
}

mp_limb_t
mpn_mod_1_1p (mp_srcptr ap, mp_size_t n, mp_limb_t b, const mp_limb_t bmodb[4])
{
  mp_limb_t rh, rl, bi, ph, pl, r;
  mp_limb_t B1modb, B2modb;
  mp_size_t i;
  int cnt;
  mp_limb_t mask;

  ASSERT (n >= 2);		/* fix tuneup.c if this is changed */

  B1modb = bmodb[2];
  B2modb = bmodb[3];

  rl = ap[n - 1];
  umul_ppmm (ph, pl, rl, B1modb);
  add_ssaaaa (rh, rl, ph, pl, CNST_LIMB(0), ap[n - 2]);

  for (i = n - 3; i >= 0; i -= 1)
    {
      /* rr = ap[i]				< B
	    + LO(rr)  * (B mod b)		<= (B-1)(b-1)
	    + HI(rr)  * (B^2 mod b)		<= (B-1)(b-1)
      */
      umul_ppmm (ph, pl, rl, B1modb);
      add_ssaaaa (ph, pl, ph, pl, CNST_LIMB(0), ap[i]);

      umul_ppmm (rh, rl, rh, B2modb);
      add_ssaaaa (rh, rl, rh, rl, ph, pl);
    }

  cnt = bmodb[1];
  bi = bmodb[0];

  if (LIKELY (cnt != 0))
    rh = (rh << cnt) | (rl >> (GMP_LIMB_BITS - cnt));

  mask = -(mp_limb_t) (rh >= b);
  rh -= mask & b;

  udiv_rnnd_preinv (r, rh, rl << cnt, b, bi);

  return r >> cnt;
}
#endif /* MOD_1_1P_METHOD == 1 */

#if MOD_1_1P_METHOD == 2
void
mpn_mod_1_1p_cps (mp_limb_t cps[4], mp_limb_t b)
{
  mp_limb_t bi;
  mp_limb_t B2modb;
  int cnt;

  count_leading_zeros (cnt, b);

  b <<= cnt;
  invert_limb (bi, b);

  cps[0] = bi;
  cps[1] = cnt;

  if (LIKELY (cnt != 0))
    {
      mp_limb_t B1modb = -b;
      B1modb *= ((bi >> (GMP_LIMB_BITS-cnt)) | (CNST_LIMB(1) << cnt));
      ASSERT (B1modb <= b);		/* NB: not fully reduced mod b */
      cps[2] = B1modb >> cnt;
    }
  B2modb = - b * bi;
  ASSERT (B2modb <= b);    // NB: equality iff b = B/2
  cps[3] = B2modb;
}

mp_limb_t
mpn_mod_1_1p (mp_srcptr ap, mp_size_t n, mp_limb_t b, const mp_limb_t bmodb[4])
{
  int cnt;
  mp_limb_t bi, B1modb;
  mp_limb_t r0, r1;
  mp_limb_t r;

  ASSERT (n >= 2);		/* fix tuneup.c if this is changed */

  r0 = ap[n-2];
  r1 = ap[n-1];

  if (n > 2)
    {
      mp_limb_t B2modb, B2mb;
      mp_limb_t p0, p1;
      mp_limb_t r2;
      mp_size_t j;

      B2modb = bmodb[3];
      B2mb = B2modb - b;

      umul_ppmm (p1, p0, r1, B2modb);
      add_mssaaaa (r2, r1, r0, r0, ap[n-3], p1, p0);

      for (j = n-4; j >= 0; j--)
	{
	  mp_limb_t cy;
	  /* mp_limb_t t = r0 + B2mb; */
	  umul_ppmm (p1, p0, r1, B2modb);

	  ADDC_LIMB (cy, r0, r0, r2 & B2modb);
	  /* Alternative, for cmov: if (cy) r0 = t; */
	  r0 -= (-cy) & b;
	  add_mssaaaa (r2, r1, r0, r0, ap[j], p1, p0);
	}

      r1 -= (r2 & b);
    }

  cnt = bmodb[1];

  if (LIKELY (cnt != 0))
    {
      mp_limb_t t;
      mp_limb_t B1modb = bmodb[2];

      umul_ppmm (r1, t, r1, B1modb);
      r0 += t;
      r1 += (r0 < t);

      /* Normalize */
      r1 = (r1 << cnt) | (r0 >> (GMP_LIMB_BITS - cnt));
      r0 <<= cnt;

      /* NOTE: Might get r1 == b here, but udiv_rnnd_preinv allows that. */
    }
  else
    {
      mp_limb_t mask = -(mp_limb_t) (r1 >= b);
      r1 -= mask & b;
    }

  bi = bmodb[0];

  udiv_rnnd_preinv (r, r1, r0, b, bi);
  return r >> cnt;
}
#endif /* MOD_1_1P_METHOD == 2 */
