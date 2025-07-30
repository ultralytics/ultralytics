/* mpn_sqrtrem -- square root and remainder

   Contributed to the GNU project by Paul Zimmermann (most code),
   Torbjorn Granlund (mpn_sqrtrem1) and Marco Bodrato (mpn_dc_sqrt).

   THE FUNCTIONS IN THIS FILE EXCEPT mpn_sqrtrem ARE INTERNAL WITH A
   MUTABLE INTERFACE.  IT IS ONLY SAFE TO REACH THEM THROUGH DOCUMENTED
   INTERFACES.  IN FACT, IT IS ALMOST GUARANTEED THAT THEY WILL CHANGE OR
   DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 1999-2002, 2004, 2005, 2008, 2010, 2012, 2015 Free Software
Foundation, Inc.

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


/* See "Karatsuba Square Root", reference in gmp.texi.  */


#include <stdio.h>
#include <stdlib.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"
#define USE_DIVAPPR_Q 1
#define TRACE(x)

static const unsigned char invsqrttab[384] = /* The common 0x100 was removed */
{
  0xff,0xfd,0xfb,0xf9,0xf7,0xf5,0xf3,0xf2, /* sqrt(1/80)..sqrt(1/87) */
  0xf0,0xee,0xec,0xea,0xe9,0xe7,0xe5,0xe4, /* sqrt(1/88)..sqrt(1/8f) */
  0xe2,0xe0,0xdf,0xdd,0xdb,0xda,0xd8,0xd7, /* sqrt(1/90)..sqrt(1/97) */
  0xd5,0xd4,0xd2,0xd1,0xcf,0xce,0xcc,0xcb, /* sqrt(1/98)..sqrt(1/9f) */
  0xc9,0xc8,0xc6,0xc5,0xc4,0xc2,0xc1,0xc0, /* sqrt(1/a0)..sqrt(1/a7) */
  0xbe,0xbd,0xbc,0xba,0xb9,0xb8,0xb7,0xb5, /* sqrt(1/a8)..sqrt(1/af) */
  0xb4,0xb3,0xb2,0xb0,0xaf,0xae,0xad,0xac, /* sqrt(1/b0)..sqrt(1/b7) */
  0xaa,0xa9,0xa8,0xa7,0xa6,0xa5,0xa4,0xa3, /* sqrt(1/b8)..sqrt(1/bf) */
  0xa2,0xa0,0x9f,0x9e,0x9d,0x9c,0x9b,0x9a, /* sqrt(1/c0)..sqrt(1/c7) */
  0x99,0x98,0x97,0x96,0x95,0x94,0x93,0x92, /* sqrt(1/c8)..sqrt(1/cf) */
  0x91,0x90,0x8f,0x8e,0x8d,0x8c,0x8c,0x8b, /* sqrt(1/d0)..sqrt(1/d7) */
  0x8a,0x89,0x88,0x87,0x86,0x85,0x84,0x83, /* sqrt(1/d8)..sqrt(1/df) */
  0x83,0x82,0x81,0x80,0x7f,0x7e,0x7e,0x7d, /* sqrt(1/e0)..sqrt(1/e7) */
  0x7c,0x7b,0x7a,0x79,0x79,0x78,0x77,0x76, /* sqrt(1/e8)..sqrt(1/ef) */
  0x76,0x75,0x74,0x73,0x72,0x72,0x71,0x70, /* sqrt(1/f0)..sqrt(1/f7) */
  0x6f,0x6f,0x6e,0x6d,0x6d,0x6c,0x6b,0x6a, /* sqrt(1/f8)..sqrt(1/ff) */
  0x6a,0x69,0x68,0x68,0x67,0x66,0x66,0x65, /* sqrt(1/100)..sqrt(1/107) */
  0x64,0x64,0x63,0x62,0x62,0x61,0x60,0x60, /* sqrt(1/108)..sqrt(1/10f) */
  0x5f,0x5e,0x5e,0x5d,0x5c,0x5c,0x5b,0x5a, /* sqrt(1/110)..sqrt(1/117) */
  0x5a,0x59,0x59,0x58,0x57,0x57,0x56,0x56, /* sqrt(1/118)..sqrt(1/11f) */
  0x55,0x54,0x54,0x53,0x53,0x52,0x52,0x51, /* sqrt(1/120)..sqrt(1/127) */
  0x50,0x50,0x4f,0x4f,0x4e,0x4e,0x4d,0x4d, /* sqrt(1/128)..sqrt(1/12f) */
  0x4c,0x4b,0x4b,0x4a,0x4a,0x49,0x49,0x48, /* sqrt(1/130)..sqrt(1/137) */
  0x48,0x47,0x47,0x46,0x46,0x45,0x45,0x44, /* sqrt(1/138)..sqrt(1/13f) */
  0x44,0x43,0x43,0x42,0x42,0x41,0x41,0x40, /* sqrt(1/140)..sqrt(1/147) */
  0x40,0x3f,0x3f,0x3e,0x3e,0x3d,0x3d,0x3c, /* sqrt(1/148)..sqrt(1/14f) */
  0x3c,0x3b,0x3b,0x3a,0x3a,0x39,0x39,0x39, /* sqrt(1/150)..sqrt(1/157) */
  0x38,0x38,0x37,0x37,0x36,0x36,0x35,0x35, /* sqrt(1/158)..sqrt(1/15f) */
  0x35,0x34,0x34,0x33,0x33,0x32,0x32,0x32, /* sqrt(1/160)..sqrt(1/167) */
  0x31,0x31,0x30,0x30,0x2f,0x2f,0x2f,0x2e, /* sqrt(1/168)..sqrt(1/16f) */
  0x2e,0x2d,0x2d,0x2d,0x2c,0x2c,0x2b,0x2b, /* sqrt(1/170)..sqrt(1/177) */
  0x2b,0x2a,0x2a,0x29,0x29,0x29,0x28,0x28, /* sqrt(1/178)..sqrt(1/17f) */
  0x27,0x27,0x27,0x26,0x26,0x26,0x25,0x25, /* sqrt(1/180)..sqrt(1/187) */
  0x24,0x24,0x24,0x23,0x23,0x23,0x22,0x22, /* sqrt(1/188)..sqrt(1/18f) */
  0x21,0x21,0x21,0x20,0x20,0x20,0x1f,0x1f, /* sqrt(1/190)..sqrt(1/197) */
  0x1f,0x1e,0x1e,0x1e,0x1d,0x1d,0x1d,0x1c, /* sqrt(1/198)..sqrt(1/19f) */
  0x1c,0x1b,0x1b,0x1b,0x1a,0x1a,0x1a,0x19, /* sqrt(1/1a0)..sqrt(1/1a7) */
  0x19,0x19,0x18,0x18,0x18,0x18,0x17,0x17, /* sqrt(1/1a8)..sqrt(1/1af) */
  0x17,0x16,0x16,0x16,0x15,0x15,0x15,0x14, /* sqrt(1/1b0)..sqrt(1/1b7) */
  0x14,0x14,0x13,0x13,0x13,0x12,0x12,0x12, /* sqrt(1/1b8)..sqrt(1/1bf) */
  0x12,0x11,0x11,0x11,0x10,0x10,0x10,0x0f, /* sqrt(1/1c0)..sqrt(1/1c7) */
  0x0f,0x0f,0x0f,0x0e,0x0e,0x0e,0x0d,0x0d, /* sqrt(1/1c8)..sqrt(1/1cf) */
  0x0d,0x0c,0x0c,0x0c,0x0c,0x0b,0x0b,0x0b, /* sqrt(1/1d0)..sqrt(1/1d7) */
  0x0a,0x0a,0x0a,0x0a,0x09,0x09,0x09,0x09, /* sqrt(1/1d8)..sqrt(1/1df) */
  0x08,0x08,0x08,0x07,0x07,0x07,0x07,0x06, /* sqrt(1/1e0)..sqrt(1/1e7) */
  0x06,0x06,0x06,0x05,0x05,0x05,0x04,0x04, /* sqrt(1/1e8)..sqrt(1/1ef) */
  0x04,0x04,0x03,0x03,0x03,0x03,0x02,0x02, /* sqrt(1/1f0)..sqrt(1/1f7) */
  0x02,0x02,0x01,0x01,0x01,0x01,0x00,0x00  /* sqrt(1/1f8)..sqrt(1/1ff) */
};

/* Compute s = floor(sqrt(a0)), and *rp = a0 - s^2.  */

#if GMP_NUMB_BITS > 32
#define MAGIC CNST_LIMB(0x10000000000)	/* 0xffe7debbfc < MAGIC < 0x232b1850f410 */
#else
#define MAGIC CNST_LIMB(0x100000)		/* 0xfee6f < MAGIC < 0x29cbc8 */
#endif

static mp_limb_t
mpn_sqrtrem1 (mp_ptr rp, mp_limb_t a0)
{
#if GMP_NUMB_BITS > 32
  mp_limb_t a1;
#endif
  mp_limb_t x0, t2, t, x2;
  unsigned abits;

  ASSERT_ALWAYS (GMP_NAIL_BITS == 0);
  ASSERT_ALWAYS (GMP_LIMB_BITS == 32 || GMP_LIMB_BITS == 64);
  ASSERT (a0 >= GMP_NUMB_HIGHBIT / 2);

  /* Use Newton iterations for approximating 1/sqrt(a) instead of sqrt(a),
     since we can do the former without division.  As part of the last
     iteration convert from 1/sqrt(a) to sqrt(a).  */

  abits = a0 >> (GMP_LIMB_BITS - 1 - 8);	/* extract bits for table lookup */
  x0 = 0x100 | invsqrttab[abits - 0x80];	/* initial 1/sqrt(a) */

  /* x0 is now an 8 bits approximation of 1/sqrt(a0) */

#if GMP_NUMB_BITS > 32
  a1 = a0 >> (GMP_LIMB_BITS - 1 - 32);
  t = (mp_limb_signed_t) (CNST_LIMB(0x2000000000000) - 0x30000 - a1 * x0 * x0) >> 16;
  x0 = (x0 << 16) + ((mp_limb_signed_t) (x0 * t) >> (16+2));

  /* x0 is now a 16 bits approximation of 1/sqrt(a0) */

  t2 = x0 * (a0 >> (32-8));
  t = t2 >> 25;
  t = ((mp_limb_signed_t) ((a0 << 14) - t * t - MAGIC) >> (32-8));
  x0 = t2 + ((mp_limb_signed_t) (x0 * t) >> 15);
  x0 >>= 32;
#else
  t2 = x0 * (a0 >> (16-8));
  t = t2 >> 13;
  t = ((mp_limb_signed_t) ((a0 << 6) - t * t - MAGIC) >> (16-8));
  x0 = t2 + ((mp_limb_signed_t) (x0 * t) >> 7);
  x0 >>= 16;
#endif

  /* x0 is now a full limb approximation of sqrt(a0) */

  x2 = x0 * x0;
  if (x2 + 2*x0 <= a0 - 1)
    {
      x2 += 2*x0 + 1;
      x0++;
    }

  *rp = a0 - x2;
  return x0;
}


#define Prec (GMP_NUMB_BITS >> 1)

/* same as mpn_sqrtrem, but for size=2 and {np, 2} normalized
   return cc such that {np, 2} = sp[0]^2 + cc*2^GMP_NUMB_BITS + rp[0] */
static mp_limb_t
mpn_sqrtrem2 (mp_ptr sp, mp_ptr rp, mp_srcptr np)
{
  mp_limb_t q, u, np0, sp0, rp0, q2;
  int cc;

  ASSERT (np[1] >= GMP_NUMB_HIGHBIT / 2);

  np0 = np[0];
  sp0 = mpn_sqrtrem1 (rp, np[1]);
  rp0 = rp[0];
  /* rp0 <= 2*sp0 < 2^(Prec + 1) */
  rp0 = (rp0 << (Prec - 1)) + (np0 >> (Prec + 1));
  q = rp0 / sp0;
  /* q <= 2^Prec, if q = 2^Prec, reduce the overestimate. */
  q -= q >> Prec;
  /* now we have q < 2^Prec */
  u = rp0 - q * sp0;
  /* now we have (rp[0]<<Prec + np0>>Prec)/2 = q * sp0 + u */
  sp0 = (sp0 << Prec) | q;
  cc = u >> (Prec - 1);
  rp0 = ((u << (Prec + 1)) & GMP_NUMB_MASK) + (np0 & ((CNST_LIMB (1) << (Prec + 1)) - 1));
  /* subtract q * q from rp */
  q2 = q * q;
  cc -= rp0 < q2;
  rp0 -= q2;
  if (cc < 0)
    {
      rp0 += sp0;
      cc += rp0 < sp0;
      --sp0;
      rp0 += sp0;
      cc += rp0 < sp0;
    }

  rp[0] = rp0;
  sp[0] = sp0;
  return cc;
}

/* writes in {sp, n} the square root (rounded towards zero) of {np, 2n},
   and in {np, n} the low n limbs of the remainder, returns the high
   limb of the remainder (which is 0 or 1).
   Assumes {np, 2n} is normalized, i.e. np[2n-1] >= B/4
   where B=2^GMP_NUMB_BITS.
   Needs a scratch of n/2+1 limbs. */
static mp_limb_t
mpn_dc_sqrtrem (mp_ptr sp, mp_ptr np, mp_size_t n, mp_limb_t approx, mp_ptr scratch)
{
  mp_limb_t q;			/* carry out of {sp, n} */
  int c, b;			/* carry out of remainder */
  mp_size_t l, h;

  ASSERT (np[2 * n - 1] >= GMP_NUMB_HIGHBIT / 2);

  if (n == 1)
    c = mpn_sqrtrem2 (sp, np, np);
  else
    {
      l = n / 2;
      h = n - l;
      q = mpn_dc_sqrtrem (sp + l, np + 2 * l, h, 0, scratch);
      if (q != 0)
	ASSERT_CARRY (mpn_sub_n (np + 2 * l, np + 2 * l, sp + l, h));
      TRACE(printf("tdiv_qr(,,,,%u,,%u) -> %u\n", (unsigned) n, (unsigned) h, (unsigned) (n - h + 1)));
      mpn_tdiv_qr (scratch, np + l, 0, np + l, n, sp + l, h);
      q += scratch[l];
      c = scratch[0] & 1;
      mpn_rshift (sp, scratch, l, 1);
      sp[l - 1] |= (q << (GMP_NUMB_BITS - 1)) & GMP_NUMB_MASK;
      if (UNLIKELY ((sp[0] & approx) != 0)) /* (sp[0] & mask) > 1 */
	return 1; /* Remainder is non-zero */
      q >>= 1;
      if (c != 0)
	c = mpn_add_n (np + l, np + l, sp + l, h);
      TRACE(printf("sqr(,,%u)\n", (unsigned) l));
      mpn_sqr (np + n, sp, l);
      b = q + mpn_sub_n (np, np, np + n, 2 * l);
      c -= (l == h) ? b : mpn_sub_1 (np + 2 * l, np + 2 * l, 1, (mp_limb_t) b);

      if (c < 0)
	{
	  q = mpn_add_1 (sp + l, sp + l, h, q);
#if HAVE_NATIVE_mpn_addlsh1_n_ip1 || HAVE_NATIVE_mpn_addlsh1_n
	  c += mpn_addlsh1_n_ip1 (np, sp, n) + 2 * q;
#else
	  c += mpn_addmul_1 (np, sp, n, CNST_LIMB(2)) + 2 * q;
#endif
	  c -= mpn_sub_1 (np, np, n, CNST_LIMB(1));
	  q -= mpn_sub_1 (sp, sp, n, CNST_LIMB(1));
	}
    }

  return c;
}

#if USE_DIVAPPR_Q
static void
mpn_divappr_q (mp_ptr qp, mp_srcptr np, mp_size_t nn, mp_srcptr dp, mp_size_t dn, mp_ptr scratch)
{
  gmp_pi1_t inv;
  mp_limb_t qh;
  ASSERT (dn > 2);
  ASSERT (nn >= dn);
  ASSERT ((dp[dn-1] & GMP_NUMB_HIGHBIT) != 0);

  MPN_COPY (scratch, np, nn);
  invert_pi1 (inv, dp[dn-1], dp[dn-2]);
  if (BELOW_THRESHOLD (dn, DC_DIVAPPR_Q_THRESHOLD))
    qh = mpn_sbpi1_divappr_q (qp, scratch, nn, dp, dn, inv.inv32);
  else if (BELOW_THRESHOLD (dn, MU_DIVAPPR_Q_THRESHOLD))
    qh = mpn_dcpi1_divappr_q (qp, scratch, nn, dp, dn, &inv);
  else
    {
      mp_size_t itch = mpn_mu_divappr_q_itch (nn, dn, 0);
      TMP_DECL;
      TMP_MARK;
      /* Sadly, scratch is too small. */
      qh = mpn_mu_divappr_q (qp, np, nn, dp, dn, TMP_ALLOC_LIMBS (itch));
      TMP_FREE;
    }
  qp [nn - dn] = qh;
}
#endif

/* writes in {sp, n} the square root (rounded towards zero) of {np, 2n-odd},
   returns zero if the operand was a perfect square, one otherwise.
   Assumes {np, 2n-odd}*4^nsh is normalized, i.e. B > np[2n-1-odd]*4^nsh >= B/4
   where B=2^GMP_NUMB_BITS.
   THINK: In the odd case, three more (dummy) limbs are taken into account,
   when nsh is maximal, two limbs are discarded from the result of the
   division. Too much? Is a single dummy limb enough? */
static int
mpn_dc_sqrt (mp_ptr sp, mp_srcptr np, mp_size_t n, unsigned nsh, unsigned odd)
{
  mp_limb_t q;			/* carry out of {sp, n} */
  int c;			/* carry out of remainder */
  mp_size_t l, h;
  mp_ptr qp, tp, scratch;
  TMP_DECL;
  TMP_MARK;

  ASSERT (np[2 * n - 1 - odd] != 0);
  ASSERT (n > 4);
  ASSERT (nsh < GMP_NUMB_BITS / 2);

  l = (n - 1) / 2;
  h = n - l;
  ASSERT (n >= l + 2 && l + 2 >= h && h > l && l >= 1 + odd);
  scratch = TMP_ALLOC_LIMBS (l + 2 * n + 5 - USE_DIVAPPR_Q); /* n + 2-USE_DIVAPPR_Q */
  tp = scratch + n + 2 - USE_DIVAPPR_Q; /* n + h + 1, but tp [-1] is writable */
  if (nsh != 0)
    {
      /* o is used to exactly set the lowest bits of the dividend, is it needed? */
      int o = l > (1 + odd);
      ASSERT_NOCARRY (mpn_lshift (tp - o, np + l - 1 - o - odd, n + h + 1 + o, 2 * nsh));
    }
  else
    MPN_COPY (tp, np + l - 1 - odd, n + h + 1);
  q = mpn_dc_sqrtrem (sp + l, tp + l + 1, h, 0, scratch);
  if (q != 0)
    ASSERT_CARRY (mpn_sub_n (tp + l + 1, tp + l + 1, sp + l, h));
  qp = tp + n + 1; /* l + 2 */
  TRACE(printf("div(appr)_q(,,%u,,%u) -> %u \n", (unsigned) n+1, (unsigned) h, (unsigned) (n + 1 - h + 1)));
#if USE_DIVAPPR_Q
  mpn_divappr_q (qp, tp, n + 1, sp + l, h, scratch);
#else
  mpn_div_q (qp, tp, n + 1, sp + l, h, scratch);
#endif
  q += qp [l + 1];
  c = 1;
  if (q > 1)
    {
      /* FIXME: if s!=0 we will shift later, a noop on this area. */
      MPN_FILL (sp, l, GMP_NUMB_MAX);
    }
  else
    {
      /* FIXME: if s!=0 we will shift again later, shift just once. */
      mpn_rshift (sp, qp + 1, l, 1);
      sp[l - 1] |= q << (GMP_NUMB_BITS - 1);
      if (((qp[0] >> (2 + USE_DIVAPPR_Q)) | /* < 3 + 4*USE_DIVAPPR_Q */
	   (qp[1] & (GMP_NUMB_MASK >> ((GMP_NUMB_BITS >> odd)- nsh - 1)))) == 0)
	{
	  mp_limb_t cy;
	  /* Approximation is not good enough, the extra limb(+ nsh bits)
	     is smaller than needed to absorb the possible error. */
	  /* {qp + 1, l + 1} equals 2*{sp, l} */
	  /* FIXME: use mullo or wrap-around, or directly evaluate
	     remainder with a single sqrmod_bnm1. */
	  TRACE(printf("mul(,,%u,,%u)\n", (unsigned) h, (unsigned) (l+1)));
	  ASSERT_NOCARRY (mpn_mul (scratch, sp + l, h, qp + 1, l + 1));
	  /* Compute the remainder of the previous mpn_div(appr)_q. */
	  cy = mpn_sub_n (tp + 1, tp + 1, scratch, h);
#if USE_DIVAPPR_Q || WANT_ASSERT
	  MPN_DECR_U (tp + 1 + h, l, cy);
#if USE_DIVAPPR_Q
	  ASSERT (mpn_cmp (tp + 1 + h, scratch + h, l) <= 0);
	  if (mpn_cmp (tp + 1 + h, scratch + h, l) < 0)
	    {
	      /* May happen only if div result was not exact. */
#if HAVE_NATIVE_mpn_addlsh1_n_ip1 || HAVE_NATIVE_mpn_addlsh1_n
	      cy = mpn_addlsh1_n_ip1 (tp + 1, sp + l, h);
#else
	      cy = mpn_addmul_1 (tp + 1, sp + l, h, CNST_LIMB(2));
#endif
	      ASSERT_NOCARRY (mpn_add_1 (tp + 1 + h, tp + 1 + h, l, cy));
	      MPN_DECR_U (sp, l, 1);
	    }
	  /* Can the root be exact when a correction was needed? We
	     did not find an example, but it depends on divappr
	     internals, and we can not assume it true in general...*/
	  /* else */
#else /* WANT_ASSERT */
	  ASSERT (mpn_cmp (tp + 1 + h, scratch + h, l) == 0);
#endif
#endif
	  if (mpn_zero_p (tp + l + 1, h - l))
	    {
	      TRACE(printf("sqr(,,%u)\n", (unsigned) l));
	      mpn_sqr (scratch, sp, l);
	      c = mpn_cmp (tp + 1, scratch + l, l);
	      if (c == 0)
		{
		  if (nsh != 0)
		    {
		      mpn_lshift (tp, np, l, 2 * nsh);
		      np = tp;
		    }
		  c = mpn_cmp (np, scratch + odd, l - odd);
		}
	      if (c < 0)
		{
		  MPN_DECR_U (sp, l, 1);
		  c = 1;
		}
	    }
	}
    }
  TMP_FREE;

  if ((odd | nsh) != 0)
    mpn_rshift (sp, sp, n, nsh + (odd ? GMP_NUMB_BITS / 2 : 0));
  return c;
}


mp_size_t
mpn_sqrtrem (mp_ptr sp, mp_ptr rp, mp_srcptr np, mp_size_t nn)
{
  mp_limb_t *tp, s0[1], cc, high, rl;
  int c;
  mp_size_t rn, tn;
  TMP_DECL;

  ASSERT (nn > 0);
  ASSERT_MPN (np, nn);

  ASSERT (np[nn - 1] != 0);
  ASSERT (rp == NULL || MPN_SAME_OR_SEPARATE_P (np, rp, nn));
  ASSERT (rp == NULL || ! MPN_OVERLAP_P (sp, (nn + 1) / 2, rp, nn));
  ASSERT (! MPN_OVERLAP_P (sp, (nn + 1) / 2, np, nn));

  high = np[nn - 1];
  if (high & (GMP_NUMB_HIGHBIT | (GMP_NUMB_HIGHBIT / 2)))
    c = 0;
  else
    {
      count_leading_zeros (c, high);
      c -= GMP_NAIL_BITS;

      c = c / 2; /* we have to shift left by 2c bits to normalize {np, nn} */
    }
  if (nn == 1) {
    if (c == 0)
      {
	sp[0] = mpn_sqrtrem1 (&rl, high);
	if (rp != NULL)
	  rp[0] = rl;
      }
    else
      {
	cc = mpn_sqrtrem1 (&rl, high << (2*c)) >> c;
	sp[0] = cc;
	if (rp != NULL)
	  rp[0] = rl = high - cc*cc;
      }
    return rl != 0;
  }
  tn = (nn + 1) / 2; /* 2*tn is the smallest even integer >= nn */

  if ((rp == NULL) && (nn > 8))
    return mpn_dc_sqrt (sp, np, tn, c, nn & 1);
  TMP_MARK;
  if (((nn & 1) | c) != 0)
    {
      mp_limb_t mask;
      mp_ptr scratch;
      TMP_ALLOC_LIMBS_2 (tp, 2 * tn, scratch, tn / 2 + 1);
      tp[0] = 0;	     /* needed only when 2*tn > nn, but saves a test */
      if (c != 0)
	mpn_lshift (tp + (nn & 1), np, nn, 2 * c);
      else
	MPN_COPY (tp + (nn & 1), np, nn);
      c += (nn & 1) ? GMP_NUMB_BITS / 2 : 0;		/* c now represents k */
      mask = (CNST_LIMB (1) << c) - 1;
      rl = mpn_dc_sqrtrem (sp, tp, tn, (rp == NULL) ? mask - 1 : 0, scratch);
      /* We have 2^(2k)*N = S^2 + R where k = c + (2tn-nn)*GMP_NUMB_BITS/2,
	 thus 2^(2k)*N = (S-s0)^2 + 2*S*s0 - s0^2 + R where s0=S mod 2^k */
      s0[0] = sp[0] & mask;	/* S mod 2^k */
      rl += mpn_addmul_1 (tp, sp, tn, 2 * s0[0]);	/* R = R + 2*s0*S */
      cc = mpn_submul_1 (tp, s0, 1, s0[0]);
      rl -= (tn > 1) ? mpn_sub_1 (tp + 1, tp + 1, tn - 1, cc) : cc;
      mpn_rshift (sp, sp, tn, c);
      tp[tn] = rl;
      if (rp == NULL)
	rp = tp;
      c = c << 1;
      if (c < GMP_NUMB_BITS)
	tn++;
      else
	{
	  tp++;
	  c -= GMP_NUMB_BITS;
	}
      if (c != 0)
	mpn_rshift (rp, tp, tn, c);
      else
	MPN_COPY_INCR (rp, tp, tn);
      rn = tn;
    }
  else
    {
      if (rp != np)
	{
	  if (rp == NULL) /* nn <= 8 */
	    rp = TMP_SALLOC_LIMBS (nn);
	  MPN_COPY (rp, np, nn);
	}
      rn = tn + (rp[tn] = mpn_dc_sqrtrem (sp, rp, tn, 0, TMP_ALLOC_LIMBS(tn / 2 + 1)));
    }

  MPN_NORMALIZE (rp, rn);

  TMP_FREE;
  return rn;
}
