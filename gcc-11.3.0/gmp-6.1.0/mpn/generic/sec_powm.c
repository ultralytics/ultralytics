/* mpn_sec_powm -- Compute R = U^E mod M.  Secure variant, side-channel silent
   under the assumption that the multiply instruction is side channel silent.

   Contributed to the GNU project by Torbjörn Granlund.

Copyright 2007-2009, 2011-2014 Free Software Foundation, Inc.

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


/*
  BASIC ALGORITHM, Compute U^E mod M, where M < B^n is odd.

  1. T <- (B^n * U) mod M                Convert to REDC form

  2. Compute table U^0, U^1, U^2... of E-dependent size

  3. While there are more bits in E
       W <- power left-to-right base-k


  TODO:

   * Make getbits a macro, thereby allowing it to update the index operand.
     That will simplify the code using getbits.  (Perhaps make getbits' sibling
     getbit then have similar form, for symmetry.)

   * Choose window size without looping.  (Superoptimize or think(tm).)

   * REDC_1_TO_REDC_2_THRESHOLD might actually represent the cutoff between
     redc_1 and redc_n.  On such systems, we will switch to redc_2 causing
     slowdown.
*/

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"

#undef MPN_REDC_1_SEC
#define MPN_REDC_1_SEC(rp, up, mp, n, invm)				\
  do {									\
    mp_limb_t cy;							\
    cy = mpn_redc_1 (rp, up, mp, n, invm);				\
    mpn_cnd_sub_n (cy, rp, rp, mp, n);					\
  } while (0)

#undef MPN_REDC_2_SEC
#define MPN_REDC_2_SEC(rp, up, mp, n, mip)				\
  do {									\
    mp_limb_t cy;							\
    cy = mpn_redc_2 (rp, up, mp, n, mip);				\
    mpn_cnd_sub_n (cy, rp, rp, mp, n);					\
  } while (0)

#if HAVE_NATIVE_mpn_addmul_2 || HAVE_NATIVE_mpn_redc_2
#define WANT_REDC_2 1
#endif

/* Define our own mpn squaring function.  We do this since we cannot use a
   native mpn_sqr_basecase over TUNE_SQR_TOOM2_MAX, or a non-native one over
   SQR_TOOM2_THRESHOLD.  This is so because of fixed size stack allocations
   made inside mpn_sqr_basecase.  */

#if HAVE_NATIVE_mpn_sqr_diagonal
#define MPN_SQR_DIAGONAL(rp, up, n)					\
  mpn_sqr_diagonal (rp, up, n)
#else
#define MPN_SQR_DIAGONAL(rp, up, n)					\
  do {									\
    mp_size_t _i;							\
    for (_i = 0; _i < (n); _i++)					\
      {									\
	mp_limb_t ul, lpl;						\
	ul = (up)[_i];							\
	umul_ppmm ((rp)[2 * _i + 1], lpl, ul, ul << GMP_NAIL_BITS);	\
	(rp)[2 * _i] = lpl >> GMP_NAIL_BITS;				\
      }									\
  } while (0)
#endif


#if ! HAVE_NATIVE_mpn_sqr_basecase
/* The limit of the generic code is SQR_TOOM2_THRESHOLD.  */
#define SQR_BASECASE_LIM  SQR_TOOM2_THRESHOLD
#endif

#if HAVE_NATIVE_mpn_sqr_basecase
#ifdef TUNE_SQR_TOOM2_MAX
/* We slightly abuse TUNE_SQR_TOOM2_MAX here.  If it is set for an assembly
   mpn_sqr_basecase, it comes from SQR_TOOM2_THRESHOLD_MAX in the assembly
   file.  An assembly mpn_sqr_basecase that does not define it should allow
   any size.  */
#define SQR_BASECASE_LIM  SQR_TOOM2_THRESHOLD
#endif
#endif

#ifdef WANT_FAT_BINARY
/* For fat builds, we use SQR_TOOM2_THRESHOLD which will expand to a read from
   __gmpn_cpuvec.  Perhaps any possible sqr_basecase.asm allow any size, and we
   limit the use unnecessarily.  We cannot tell, so play it safe.  FIXME.  */
#define SQR_BASECASE_LIM  SQR_TOOM2_THRESHOLD
#endif

#ifndef SQR_BASECASE_LIM
/* If SQR_BASECASE_LIM is now not defined, use mpn_sqr_basecase for any operand
   size.  */
#define mpn_local_sqr(rp,up,n,tp) mpn_sqr_basecase(rp,up,n)
#else
/* Else use mpn_sqr_basecase for its allowed sizes, else mpn_mul_basecase.  */
#define mpn_local_sqr(rp,up,n,tp) \
  do {									\
    if (BELOW_THRESHOLD (n, SQR_BASECASE_LIM))				\
      mpn_sqr_basecase (rp, up, n);					\
    else								\
      mpn_mul_basecase(rp, up, n, up, n);				\
  } while (0)
#endif

#define getbit(p,bi) \
  ((p[(bi - 1) / GMP_NUMB_BITS] >> (bi - 1) % GMP_NUMB_BITS) & 1)

/* FIXME: Maybe some things would get simpler if all callers ensure
   that bi >= nbits. As far as I understand, with the current code bi
   < nbits can happen only for the final iteration. */
static inline mp_limb_t
getbits (const mp_limb_t *p, mp_bitcnt_t bi, int nbits)
{
  int nbits_in_r;
  mp_limb_t r;
  mp_size_t i;

  if (bi < nbits)
    {
      return p[0] & (((mp_limb_t) 1 << bi) - 1);
    }
  else
    {
      bi -= nbits;			/* bit index of low bit to extract */
      i = bi / GMP_NUMB_BITS;		/* word index of low bit to extract */
      bi %= GMP_NUMB_BITS;		/* bit index in low word */
      r = p[i] >> bi;			/* extract (low) bits */
      nbits_in_r = GMP_NUMB_BITS - bi;	/* number of bits now in r */
      if (nbits_in_r < nbits)		/* did we get enough bits? */
	r += p[i + 1] << nbits_in_r;	/* prepend bits from higher word */
      return r & (((mp_limb_t ) 1 << nbits) - 1);
    }
}

#ifndef POWM_SEC_TABLE
#if GMP_NUMB_BITS < 50
#define POWM_SEC_TABLE  2,33,96,780,2741
#else
#define POWM_SEC_TABLE  2,130,524,2578
#endif
#endif

#if TUNE_PROGRAM_BUILD
extern int win_size (mp_bitcnt_t);
#else
static inline int
win_size (mp_bitcnt_t enb)
{
  int k;
  /* Find k, such that x[k-1] < enb <= x[k].

     We require that x[k] >= k, then it follows that enb > x[k-1] >=
     k-1, which implies k <= enb.
  */
  static const mp_bitcnt_t x[] = {0,POWM_SEC_TABLE,~(mp_bitcnt_t)0};
  for (k = 1; enb > x[k]; k++)
    ;
  ASSERT (k <= enb);
  return k;
}
#endif

/* Convert U to REDC form, U_r = B^n * U mod M.
   Uses scratch space at tp of size 2un + n + 1.  */
static void
redcify (mp_ptr rp, mp_srcptr up, mp_size_t un, mp_srcptr mp, mp_size_t n, mp_ptr tp)
{
  MPN_ZERO (tp, n);
  MPN_COPY (tp + n, up, un);

  mpn_sec_div_r (tp, un + n, mp, n, tp + un + n);
  MPN_COPY (rp, tp, n);
}

/* {rp, n} <-- {bp, bn} ^ {ep, en} mod {mp, n},
   where en = ceil (enb / GMP_NUMB_BITS)
   Requires that {mp, n} is odd (and hence also mp[0] odd).
   Uses scratch space at tp as defined by mpn_sec_powm_itch.  */
void
mpn_sec_powm (mp_ptr rp, mp_srcptr bp, mp_size_t bn,
	      mp_srcptr ep, mp_bitcnt_t enb,
	      mp_srcptr mp, mp_size_t n, mp_ptr tp)
{
  mp_limb_t ip[2], *mip;
  int windowsize, this_windowsize;
  mp_limb_t expbits;
  mp_ptr pp, this_pp;
  long i;
  int cnd;

  ASSERT (enb > 0);
  ASSERT (n > 0);
  /* The code works for bn = 0, but the defined scratch space is 2 limbs
     greater than we supply, when converting 1 to redc form .  */
  ASSERT (bn > 0);
  ASSERT ((mp[0] & 1) != 0);

  windowsize = win_size (enb);

#if WANT_REDC_2
  if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_2_THRESHOLD))
    {
      mip = ip;
      binvert_limb (mip[0], mp[0]);
      mip[0] = -mip[0];
    }
  else
    {
      mip = ip;
      mpn_binvert (mip, mp, 2, tp);
      mip[0] = -mip[0]; mip[1] = ~mip[1];
    }
#else
  mip = ip;
  binvert_limb (mip[0], mp[0]);
  mip[0] = -mip[0];
#endif

  pp = tp;
  tp += (n << windowsize);	/* put tp after power table */

  /* Compute pp[0] table entry */
  /* scratch: |   n   | 1 |   n+2    |  */
  /*          | pp[0] | 1 | redcify  |  */
  this_pp = pp;
  this_pp[n] = 1;
  redcify (this_pp, this_pp + n, 1, mp, n, this_pp + n + 1);
  this_pp += n;

  /* Compute pp[1] table entry.  To avoid excessive scratch usage in the
     degenerate situation where B >> M, we let redcify use scratch space which
     will later be used by the pp table (element 2 and up).  */
  /* scratch: |   n   |   n   |  bn + n + 1  |  */
  /*          | pp[0] | pp[1] |   redcify    |  */
  redcify (this_pp, bp, bn, mp, n, this_pp + n);

  /* Precompute powers of b and put them in the temporary area at pp.  */
  /* scratch: |   n   |   n   | ...  |                    |   2n      |  */
  /*          | pp[0] | pp[1] | ...  | pp[2^windowsize-1] |  product  |  */
  for (i = (1 << windowsize) - 2; i > 0; i--)
    {
      mpn_mul_basecase (tp, this_pp, n, pp + n, n);
      this_pp += n;
#if WANT_REDC_2
      if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_2_THRESHOLD))
	MPN_REDC_1_SEC (this_pp, tp, mp, n, mip[0]);
      else
	MPN_REDC_2_SEC (this_pp, tp, mp, n, mip);
#else
      MPN_REDC_1_SEC (this_pp, tp, mp, n, mip[0]);
#endif
    }

  expbits = getbits (ep, enb, windowsize);
  ASSERT_ALWAYS (enb >= windowsize);
  enb -= windowsize;

  mpn_sec_tabselect (rp, pp, n, 1 << windowsize, expbits);

  /* Main exponentiation loop.  */
  /* scratch: |   n   |   n   | ...  |                    |     3n-4n     |  */
  /*          | pp[0] | pp[1] | ...  | pp[2^windowsize-1] |  loop scratch |  */

#define INNERLOOP							\
  while (enb != 0)							\
    {									\
      expbits = getbits (ep, enb, windowsize);				\
      this_windowsize = windowsize;					\
      if (enb < windowsize)						\
	{								\
	  this_windowsize -= windowsize - enb;				\
	  enb = 0;							\
	}								\
      else								\
	enb -= windowsize;						\
									\
      do								\
	{								\
	  mpn_local_sqr (tp, rp, n, tp + 2 * n);			\
	  MPN_REDUCE (rp, tp, mp, n, mip);				\
	  this_windowsize--;						\
	}								\
      while (this_windowsize != 0);					\
									\
      mpn_sec_tabselect (tp + 2*n, pp, n, 1 << windowsize, expbits);	\
      mpn_mul_basecase (tp, rp, n, tp + 2*n, n);			\
									\
      MPN_REDUCE (rp, tp, mp, n, mip);					\
    }

#if WANT_REDC_2
  if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_2_THRESHOLD))
    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr_basecase (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_1_SEC (rp, tp, mp, n, mip[0])
      INNERLOOP;
    }
  else
    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr_basecase (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_2_SEC (rp, tp, mp, n, mip)
      INNERLOOP;
    }
#else
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr_basecase (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_1_SEC (rp, tp, mp, n, mip[0])
  INNERLOOP;
#endif

  MPN_COPY (tp, rp, n);
  MPN_ZERO (tp + n, n);

#if WANT_REDC_2
  if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_2_THRESHOLD))
    MPN_REDC_1_SEC (rp, tp, mp, n, mip[0]);
  else
    MPN_REDC_2_SEC (rp, tp, mp, n, mip);
#else
  MPN_REDC_1_SEC (rp, tp, mp, n, mip[0]);
#endif
  cnd = mpn_sub_n (tp, rp, mp, n);	/* we need just retval */
  mpn_cnd_sub_n (!cnd, rp, rp, mp, n);
}

mp_size_t
mpn_sec_powm_itch (mp_size_t bn, mp_bitcnt_t enb, mp_size_t n)
{
  int windowsize;
  mp_size_t redcify_itch, itch;

  /* The top scratch usage will either be when reducing B in the 2nd redcify
     call, or more typically n*2^windowsize + 3n or 4n, in the main loop.  (It
     is 3n or 4n depending on if we use mpn_local_sqr or a native
     mpn_sqr_basecase.  We assume 4n always for now.) */

  windowsize = win_size (enb);

  /* The 2n term is due to pp[0] and pp[1] at the time of the 2nd redcify call,
     the (bn + n) term is due to redcify's own usage, and the rest is due to
     mpn_sec_div_r's usage when called from redcify.  */
  redcify_itch = (2 * n) + (bn + n) + ((bn + n) + 2 * n + 2);

  /* The n * 2^windowsize term is due to the power table, the 4n term is due to
     scratch needs of squaring/multiplication in the exponentiation loop.  */
  itch = (n << windowsize) + (4 * n);

  return MAX (itch, redcify_itch);
}
