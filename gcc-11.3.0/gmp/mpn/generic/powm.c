/* mpn_powm -- Compute R = U^E mod M.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2007-2012 Free Software Foundation, Inc.

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

  1. W <- U

  2. T <- (B^n * U) mod M                Convert to REDC form

  3. Compute table U^1, U^3, U^5... of E-dependent size

  4. While there are more bits in E
       W <- power left-to-right base-k


  TODO:

   * Make getbits a macro, thereby allowing it to update the index operand.
     That will simplify the code using getbits.  (Perhaps make getbits' sibling
     getbit then have similar form, for symmetry.)

   * Write an itch function.  Or perhaps get rid of tp parameter since the huge
     pp area is allocated locally anyway?

   * Choose window size without looping.  (Superoptimize or think(tm).)

   * Handle small bases with initial, reduction-free exponentiation.

   * Call new division functions, not mpn_tdiv_qr.

   * Consider special code for one-limb M.

   * How should we handle the redc1/redc2/redc_n choice?
     - redc1:  T(binvert_1limb)  + e * (n)   * (T(mullo-1x1) + n*T(addmul_1))
     - redc2:  T(binvert_2limbs) + e * (n/2) * (T(mullo-2x2) + n*T(addmul_2))
     - redc_n: T(binvert_nlimbs) + e * (T(mullo-nxn) + T(M(n)))
     This disregards the addmul_N constant term, but we could think of
     that as part of the respective mullo.

   * When U (the base) is small, we should start the exponentiation with plain
     operations, then convert that partial result to REDC form.

   * When U is just one limb, should it be handled without the k-ary tricks?
     We could keep a factor of B^n in W, but use U' = BU as base.  After
     multiplying by this (pseudo two-limb) number, we need to multiply by 1/B
     mod M.
*/

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"

#undef MPN_REDC_1
#define MPN_REDC_1(rp, up, mp, n, invm)					\
  do {									\
    mp_limb_t cy;							\
    cy = mpn_redc_1 (rp, up, mp, n, invm);				\
    if (cy != 0)							\
      mpn_sub_n (rp, rp, mp, n);					\
  } while (0)

#undef MPN_REDC_2
#define MPN_REDC_2(rp, up, mp, n, mip)					\
  do {									\
    mp_limb_t cy;							\
    cy = mpn_redc_2 (rp, up, mp, n, mip);				\
    if (cy != 0)							\
      mpn_sub_n (rp, rp, mp, n);					\
  } while (0)

#if HAVE_NATIVE_mpn_addmul_2 || HAVE_NATIVE_mpn_redc_2
#define WANT_REDC_2 1
#endif

#define getbit(p,bi) \
  ((p[(bi - 1) / GMP_LIMB_BITS] >> (bi - 1) % GMP_LIMB_BITS) & 1)

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

static inline int
win_size (mp_bitcnt_t eb)
{
  int k;
  static mp_bitcnt_t x[] = {0,7,25,81,241,673,1793,4609,11521,28161,~(mp_bitcnt_t)0};
  for (k = 1; eb > x[k]; k++)
    ;
  return k;
}

/* Convert U to REDC form, U_r = B^n * U mod M */
static void
redcify (mp_ptr rp, mp_srcptr up, mp_size_t un, mp_srcptr mp, mp_size_t n)
{
  mp_ptr tp, qp;
  TMP_DECL;
  TMP_MARK;

  TMP_ALLOC_LIMBS_2 (tp, un + n, qp, un + 1);

  MPN_ZERO (tp, n);
  MPN_COPY (tp + n, up, un);
  mpn_tdiv_qr (qp, rp, 0L, tp, un + n, mp, n);
  TMP_FREE;
}

/* rp[n-1..0] = bp[bn-1..0] ^ ep[en-1..0] mod mp[n-1..0]
   Requires that mp[n-1..0] is odd.
   Requires that ep[en-1..0] is > 1.
   Uses scratch space at tp of MAX(mpn_binvert_itch(n),2n) limbs.  */
void
mpn_powm (mp_ptr rp, mp_srcptr bp, mp_size_t bn,
	  mp_srcptr ep, mp_size_t en,
	  mp_srcptr mp, mp_size_t n, mp_ptr tp)
{
  mp_limb_t ip[2], *mip;
  int cnt;
  mp_bitcnt_t ebi;
  int windowsize, this_windowsize;
  mp_limb_t expbits;
  mp_ptr pp, this_pp;
  long i;
  TMP_DECL;

  ASSERT (en > 1 || (en == 1 && ep[0] > 1));
  ASSERT (n >= 1 && ((mp[0] & 1) != 0));

  TMP_MARK;

  MPN_SIZEINBASE_2EXP(ebi, ep, en, 1);

#if 0
  if (bn < n)
    {
      /* Do the first few exponent bits without mod reductions,
	 until the result is greater than the mod argument.  */
      for (;;)
	{
	  mpn_sqr (tp, this_pp, tn);
	  tn = tn * 2 - 1,  tn += tp[tn] != 0;
	  if (getbit (ep, ebi) != 0)
	    mpn_mul (..., tp, tn, bp, bn);
	  ebi--;
	}
    }
#endif

  windowsize = win_size (ebi);

#if WANT_REDC_2
  if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_2_THRESHOLD))
    {
      mip = ip;
      binvert_limb (mip[0], mp[0]);
      mip[0] = -mip[0];
    }
  else if (BELOW_THRESHOLD (n, REDC_2_TO_REDC_N_THRESHOLD))
    {
      mip = ip;
      mpn_binvert (mip, mp, 2, tp);
      mip[0] = -mip[0]; mip[1] = ~mip[1];
    }
#else
  if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_N_THRESHOLD))
    {
      mip = ip;
      binvert_limb (mip[0], mp[0]);
      mip[0] = -mip[0];
    }
#endif
  else
    {
      mip = TMP_ALLOC_LIMBS (n);
      mpn_binvert (mip, mp, n, tp);
    }

  pp = TMP_ALLOC_LIMBS (n << (windowsize - 1));

  this_pp = pp;
  redcify (this_pp, bp, bn, mp, n);

  /* Store b^2 at rp.  */
  mpn_sqr (tp, this_pp, n);
#if WANT_REDC_2
  if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_2_THRESHOLD))
    MPN_REDC_1 (rp, tp, mp, n, mip[0]);
  else if (BELOW_THRESHOLD (n, REDC_2_TO_REDC_N_THRESHOLD))
    MPN_REDC_2 (rp, tp, mp, n, mip);
#else
  if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_N_THRESHOLD))
    MPN_REDC_1 (rp, tp, mp, n, mip[0]);
#endif
  else
    mpn_redc_n (rp, tp, mp, n, mip);

  /* Precompute odd powers of b and put them in the temporary area at pp.  */
  for (i = (1 << (windowsize - 1)) - 1; i > 0; i--)
    {
      mpn_mul_n (tp, this_pp, rp, n);
      this_pp += n;
#if WANT_REDC_2
      if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_2_THRESHOLD))
	MPN_REDC_1 (this_pp, tp, mp, n, mip[0]);
      else if (BELOW_THRESHOLD (n, REDC_2_TO_REDC_N_THRESHOLD))
	MPN_REDC_2 (this_pp, tp, mp, n, mip);
#else
      if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_N_THRESHOLD))
	MPN_REDC_1 (this_pp, tp, mp, n, mip[0]);
#endif
      else
	mpn_redc_n (this_pp, tp, mp, n, mip);
    }

  expbits = getbits (ep, ebi, windowsize);
  if (ebi < windowsize)
    ebi = 0;
  else
    ebi -= windowsize;

  count_trailing_zeros (cnt, expbits);
  ebi += cnt;
  expbits >>= cnt;

  MPN_COPY (rp, pp + n * (expbits >> 1), n);

#define INNERLOOP							\
  while (ebi != 0)							\
    {									\
      while (getbit (ep, ebi) == 0)					\
	{								\
	  MPN_SQR (tp, rp, n);						\
	  MPN_REDUCE (rp, tp, mp, n, mip);				\
	  ebi--;							\
	  if (ebi == 0)							\
	    goto done;							\
	}								\
									\
      /* The next bit of the exponent is 1.  Now extract the largest	\
	 block of bits <= windowsize, and such that the least		\
	 significant bit is 1.  */					\
									\
      expbits = getbits (ep, ebi, windowsize);				\
      this_windowsize = windowsize;					\
      if (ebi < windowsize)						\
	{								\
	  this_windowsize -= windowsize - ebi;				\
	  ebi = 0;							\
	}								\
      else								\
        ebi -= windowsize;						\
									\
      count_trailing_zeros (cnt, expbits);				\
      this_windowsize -= cnt;						\
      ebi += cnt;							\
      expbits >>= cnt;							\
									\
      do								\
	{								\
	  MPN_SQR (tp, rp, n);						\
	  MPN_REDUCE (rp, tp, mp, n, mip);				\
	  this_windowsize--;						\
	}								\
      while (this_windowsize != 0);					\
									\
      MPN_MUL_N (tp, rp, pp + n * (expbits >> 1), n);			\
      MPN_REDUCE (rp, tp, mp, n, mip);					\
    }


#if WANT_REDC_2
  if (REDC_1_TO_REDC_2_THRESHOLD < MUL_TOOM22_THRESHOLD)
    {
      if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_2_THRESHOLD))
	{
	  if (REDC_1_TO_REDC_2_THRESHOLD < SQR_BASECASE_THRESHOLD
	      || BELOW_THRESHOLD (n, SQR_BASECASE_THRESHOLD))
	    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_mul_basecase (r,a,n,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_1 (rp, tp, mp, n, mip[0])
	      INNERLOOP;
	    }
	  else
	    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr_basecase (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_1 (rp, tp, mp, n, mip[0])
	      INNERLOOP;
	    }
	}
      else if (BELOW_THRESHOLD (n, MUL_TOOM22_THRESHOLD))
	{
	  if (MUL_TOOM22_THRESHOLD < SQR_BASECASE_THRESHOLD
	      || BELOW_THRESHOLD (n, SQR_BASECASE_THRESHOLD))
	    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_mul_basecase (r,a,n,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_2 (rp, tp, mp, n, mip)
	      INNERLOOP;
	    }
	  else
	    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr_basecase (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_2 (rp, tp, mp, n, mip)
	      INNERLOOP;
	    }
	}
      else if (BELOW_THRESHOLD (n, REDC_2_TO_REDC_N_THRESHOLD))
	{
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_n (r,a,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_2 (rp, tp, mp, n, mip)
	  INNERLOOP;
	}
      else
	{
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_n (r,a,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	mpn_redc_n (rp, tp, mp, n, mip)
	  INNERLOOP;
	}
    }
  else
    {
      if (BELOW_THRESHOLD (n, MUL_TOOM22_THRESHOLD))
	{
	  if (MUL_TOOM22_THRESHOLD < SQR_BASECASE_THRESHOLD
	      || BELOW_THRESHOLD (n, SQR_BASECASE_THRESHOLD))
	    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_mul_basecase (r,a,n,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_1 (rp, tp, mp, n, mip[0])
	      INNERLOOP;
	    }
	  else
	    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr_basecase (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_1 (rp, tp, mp, n, mip[0])
	      INNERLOOP;
	    }
	}
      else if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_2_THRESHOLD))
	{
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_n (r,a,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_1 (rp, tp, mp, n, mip[0])
	  INNERLOOP;
	}
      else if (BELOW_THRESHOLD (n, REDC_2_TO_REDC_N_THRESHOLD))
	{
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_n (r,a,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_2 (rp, tp, mp, n, mip)
	  INNERLOOP;
	}
      else
	{
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_n (r,a,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	mpn_redc_n (rp, tp, mp, n, mip)
	  INNERLOOP;
	}
    }

#else  /* WANT_REDC_2 */

  if (REDC_1_TO_REDC_N_THRESHOLD < MUL_TOOM22_THRESHOLD)
    {
      if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_N_THRESHOLD))
	{
	  if (REDC_1_TO_REDC_N_THRESHOLD < SQR_BASECASE_THRESHOLD
	      || BELOW_THRESHOLD (n, SQR_BASECASE_THRESHOLD))
	    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_mul_basecase (r,a,n,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_1 (rp, tp, mp, n, mip[0])
	      INNERLOOP;
	    }
	  else
	    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr_basecase (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_1 (rp, tp, mp, n, mip[0])
	      INNERLOOP;
	    }
	}
      else if (BELOW_THRESHOLD (n, MUL_TOOM22_THRESHOLD))
	{
	  if (MUL_TOOM22_THRESHOLD < SQR_BASECASE_THRESHOLD
	      || BELOW_THRESHOLD (n, SQR_BASECASE_THRESHOLD))
	    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_mul_basecase (r,a,n,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	mpn_redc_n (rp, tp, mp, n, mip)
	      INNERLOOP;
	    }
	  else
	    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr_basecase (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	mpn_redc_n (rp, tp, mp, n, mip)
	      INNERLOOP;
	    }
	}
      else
	{
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_n (r,a,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	mpn_redc_n (rp, tp, mp, n, mip)
	  INNERLOOP;
	}
    }
  else
    {
      if (BELOW_THRESHOLD (n, MUL_TOOM22_THRESHOLD))
	{
	  if (MUL_TOOM22_THRESHOLD < SQR_BASECASE_THRESHOLD
	      || BELOW_THRESHOLD (n, SQR_BASECASE_THRESHOLD))
	    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_mul_basecase (r,a,n,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_1 (rp, tp, mp, n, mip[0])
	      INNERLOOP;
	    }
	  else
	    {
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_basecase (r,a,n,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr_basecase (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_1 (rp, tp, mp, n, mip[0])
	      INNERLOOP;
	    }
	}
      else if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_N_THRESHOLD))
	{
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_n (r,a,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	MPN_REDC_1 (rp, tp, mp, n, mip[0])
	  INNERLOOP;
	}
      else
	{
#undef MPN_MUL_N
#undef MPN_SQR
#undef MPN_REDUCE
#define MPN_MUL_N(r,a,b,n)		mpn_mul_n (r,a,b,n)
#define MPN_SQR(r,a,n)			mpn_sqr (r,a,n)
#define MPN_REDUCE(rp,tp,mp,n,mip)	mpn_redc_n (rp, tp, mp, n, mip)
	  INNERLOOP;
	}
    }
#endif  /* WANT_REDC_2 */

 done:

  MPN_COPY (tp, rp, n);
  MPN_ZERO (tp + n, n);

#if WANT_REDC_2
  if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_2_THRESHOLD))
    MPN_REDC_1 (rp, tp, mp, n, mip[0]);
  else if (BELOW_THRESHOLD (n, REDC_2_TO_REDC_N_THRESHOLD))
    MPN_REDC_2 (rp, tp, mp, n, mip);
#else
  if (BELOW_THRESHOLD (n, REDC_1_TO_REDC_N_THRESHOLD))
    MPN_REDC_1 (rp, tp, mp, n, mip[0]);
#endif
  else
    mpn_redc_n (rp, tp, mp, n, mip);

  if (mpn_cmp (rp, mp, n) >= 0)
    mpn_sub_n (rp, rp, mp, n);

  TMP_FREE;
}
