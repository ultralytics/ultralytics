/* mpn_rootrem(rootp,remp,ap,an,nth) -- Compute the nth root of {ap,an}, and
   store the truncated integer part at rootp and the remainder at remp.

   Contributed by Paul Zimmermann (algorithm) and
   Paul Zimmermann and Torbjorn Granlund (implementation).
   Marco Bodrato wrote logbased_root to seed the loop. 

   THE FUNCTIONS IN THIS FILE ARE INTERNAL, AND HAVE MUTABLE INTERFACES.  IT'S
   ONLY SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT'S ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2002, 2005, 2009-2012, 2015 Free Software Foundation, Inc.

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

/* FIXME:
     This implementation is not optimal when remp == NULL, since the complexity
     is M(n), whereas it should be M(n/k) on average.
*/

#include <stdio.h>		/* for NULL */

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"

static mp_size_t mpn_rootrem_internal (mp_ptr, mp_ptr, mp_srcptr, mp_size_t,
				       mp_limb_t, int);

#define MPN_RSHIFT(rp,up,un,cnt) \
  do {									\
    if ((cnt) != 0)							\
      mpn_rshift (rp, up, un, cnt);					\
    else								\
      {									\
	MPN_COPY_INCR (rp, up, un);					\
      }									\
  } while (0)

#define MPN_LSHIFT(cy,rp,up,un,cnt) \
  do {									\
    if ((cnt) != 0)							\
      cy = mpn_lshift (rp, up, un, cnt);				\
    else								\
      {									\
	MPN_COPY_DECR (rp, up, un);					\
	cy = 0;								\
      }									\
  } while (0)


/* Put in {rootp, ceil(un/k)} the kth root of {up, un}, rounded toward zero.
   If remp <> NULL, put in {remp, un} the remainder.
   Return the size (in limbs) of the remainder if remp <> NULL,
	  or a non-zero value iff the remainder is non-zero when remp = NULL.
   Assumes:
   (a) up[un-1] is not zero
   (b) rootp has at least space for ceil(un/k) limbs
   (c) remp has at least space for un limbs (in case remp <> NULL)
   (d) the operands do not overlap.

   The auxiliary memory usage is 3*un+2 if remp = NULL,
   and 2*un+2 if remp <> NULL.  FIXME: This is an incorrect comment.
*/
mp_size_t
mpn_rootrem (mp_ptr rootp, mp_ptr remp,
	     mp_srcptr up, mp_size_t un, mp_limb_t k)
{
  ASSERT (un > 0);
  ASSERT (up[un - 1] != 0);
  ASSERT (k > 1);

  if (UNLIKELY (k == 2))
    return mpn_sqrtrem (rootp, remp, up, un);
  /* (un-1)/k > 2 <=> un > 3k <=> (un + 2)/3 > k */
  if (remp == NULL && (un + 2) / 3 > k)
    /* Pad {up,un} with k zero limbs.  This will produce an approximate root
       with one more limb, allowing us to compute the exact integral result. */
    {
      mp_ptr sp, wp;
      mp_size_t rn, sn, wn;
      TMP_DECL;
      TMP_MARK;
      wn = un + k;
      sn = (un - 1) / k + 2; /* ceil(un/k) + 1 */
      TMP_ALLOC_LIMBS_2 (wp, wn, /* will contain the padded input */
			 sp, sn); /* approximate root of padded input */
      MPN_COPY (wp + k, up, un);
      MPN_FILL (wp, k, 0);
      rn = mpn_rootrem_internal (sp, NULL, wp, wn, k, 1);
      /* The approximate root S = {sp,sn} is either the correct root of
	 {sp,sn}, or 1 too large.  Thus unless the least significant limb of
	 S is 0 or 1, we can deduce the root of {up,un} is S truncated by one
	 limb.  (In case sp[0]=1, we can deduce the root, but not decide
	 whether it is exact or not.) */
      MPN_COPY (rootp, sp + 1, sn - 1);
      TMP_FREE;
      return rn;
    }
  else
    {
      return mpn_rootrem_internal (rootp, remp, up, un, k, 0);
    }
}

#define LOGROOT_USED_BITS 8
#define LOGROOT_NEEDS_TWO_CORRECTIONS 1
#define LOGROOT_RETURNED_BITS (LOGROOT_USED_BITS + LOGROOT_NEEDS_TWO_CORRECTIONS)
/* Puts in *rootp some bits of the k^nt root of the number
   2^bitn * 1.op ; where op represents the "fractional" bits.

   The returned value is the number of bits of the root minus one;
   i.e. an approximation of the root will be
   (*rootp) * 2^(retval-LOGROOT_RETURNED_BITS+1).

   Currently, only LOGROOT_USED_BITS bits of op are used (the implicit
   one is not counted).
 */
static unsigned
logbased_root (mp_ptr rootp, mp_limb_t op, mp_bitcnt_t bitn, mp_limb_t k)
{
  /* vlog=vector(256,i,floor((log(256+i)/log(2)-8)*256)-(i>255)) */
  static const
  unsigned char vlog[] = {1,   2,   4,   5,   7,   8,   9,  11,  12,  14,  15,  16,  18,  19,  21,  22,
			 23,  25,  26,  27,  29,  30,  31,  33,  34,  35,  37,  38,  39,  40,  42,  43,
			 44,  46,  47,  48,  49,  51,  52,  53,  54,  56,  57,  58,  59,  61,  62,  63,
			 64,  65,  67,  68,  69,  70,  71,  73,  74,  75,  76,  77,  78,  80,  81,  82,
			 83,  84,  85,  87,  88,  89,  90,  91,  92,  93,  94,  96,  97,  98,  99, 100,
			101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
			118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 131, 132, 133, 134,
			135, 136, 137, 138, 139, 140, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
			150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 162, 163, 164,
			165, 166, 167, 168, 169, 170, 171, 172, 173, 173, 174, 175, 176, 177, 178, 179,
			180, 181, 181, 182, 183, 184, 185, 186, 187, 188, 188, 189, 190, 191, 192, 193,
			194, 194, 195, 196, 197, 198, 199, 200, 200, 201, 202, 203, 204, 205, 205, 206,
			207, 208, 209, 209, 210, 211, 212, 213, 214, 214, 215, 216, 217, 218, 218, 219,
			220, 221, 222, 222, 223, 224, 225, 225, 226, 227, 228, 229, 229, 230, 231, 232,
			232, 233, 234, 235, 235, 236, 237, 238, 239, 239, 240, 241, 242, 242, 243, 244,
			245, 245, 246, 247, 247, 248, 249, 250, 250, 251, 252, 253, 253, 254, 255, 255};

  /* vexp=vector(256,i,floor(2^(8+i/256)-256)-(i>255)) */
  static const
  unsigned char vexp[] = {0,   1,   2,   2,   3,   4,   4,   5,   6,   7,   7,   8,   9,   9,  10,  11,
			 12,  12,  13,  14,  14,  15,  16,  17,  17,  18,  19,  20,  20,  21,  22,  23,
			 23,  24,  25,  26,  26,  27,  28,  29,  30,  30,  31,  32,  33,  33,  34,  35,
			 36,  37,  37,  38,  39,  40,  41,  41,  42,  43,  44,  45,  45,  46,  47,  48,
			 49,  50,  50,  51,  52,  53,  54,  55,  55,  56,  57,  58,  59,  60,  61,  61,
			 62,  63,  64,  65,  66,  67,  67,  68,  69,  70,  71,  72,  73,  74,  75,  75,
			 76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  86,  87,  88,  89,  90,
			 91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106,
			107, 108, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 119, 120, 121, 122,
			123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
			139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156,
			157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 168, 169, 171, 172, 173, 174,
			175, 176, 178, 179, 180, 181, 182, 183, 185, 186, 187, 188, 189, 191, 192, 193,
			194, 196, 197, 198, 199, 200, 202, 203, 204, 205, 207, 208, 209, 210, 212, 213,
			214, 216, 217, 218, 219, 221, 222, 223, 225, 226, 227, 229, 230, 231, 232, 234,
			235, 236, 238, 239, 240, 242, 243, 245, 246, 247, 249, 250, 251, 253, 254, 255};
  mp_bitcnt_t retval;

  if (UNLIKELY (bitn > (~ (mp_bitcnt_t) 0) >> LOGROOT_USED_BITS))
    {
      /* In the unlikely case, we use two divisions and a modulo. */
      retval = bitn / k;
      bitn %= k;
      bitn = (bitn << LOGROOT_USED_BITS |
	      vlog[op >> (GMP_NUMB_BITS - LOGROOT_USED_BITS)]) / k;
    }
  else
    {
      bitn = (bitn << LOGROOT_USED_BITS |
	      vlog[op >> (GMP_NUMB_BITS - LOGROOT_USED_BITS)]) / k;
      retval = bitn >> LOGROOT_USED_BITS;
      bitn &= (CNST_LIMB (1) << LOGROOT_USED_BITS) - 1;
    }
  ASSERT(bitn < CNST_LIMB (1) << LOGROOT_USED_BITS);
  *rootp = CNST_LIMB(1) << (LOGROOT_USED_BITS - ! LOGROOT_NEEDS_TWO_CORRECTIONS)
    | vexp[bitn] >> ! LOGROOT_NEEDS_TWO_CORRECTIONS;
  return retval;
}

/* if approx is non-zero, does not compute the final remainder */
static mp_size_t
mpn_rootrem_internal (mp_ptr rootp, mp_ptr remp, mp_srcptr up, mp_size_t un,
		      mp_limb_t k, int approx)
{
  mp_ptr qp, rp, sp, wp, scratch;
  mp_size_t qn, rn, sn, wn, nl, bn;
  mp_limb_t save, save2, cy, uh;
  mp_bitcnt_t unb; /* number of significant bits of {up,un} */
  mp_bitcnt_t xnb; /* number of significant bits of the result */
  mp_bitcnt_t b, kk;
  mp_bitcnt_t sizes[GMP_NUMB_BITS + 1];
  int ni;
  int perf_pow;
  unsigned ulz, snb, c, logk;
  TMP_DECL;

  /* MPN_SIZEINBASE_2EXP(unb, up, un, 1); --unb; */
  uh = up[un - 1];
  count_leading_zeros (ulz, uh);
  ulz = ulz - GMP_NAIL_BITS + 1; /* Ignore the first 1. */
  unb = (mp_bitcnt_t) un * GMP_NUMB_BITS - ulz;
  /* unb is the (truncated) logarithm of the input U in base 2*/

  if (unb < k) /* root is 1 */
    {
      rootp[0] = 1;
      if (remp == NULL)
	un -= (*up == CNST_LIMB (1)); /* Non-zero iif {up,un} > 1 */
      else
	{
	  mpn_sub_1 (remp, up, un, CNST_LIMB (1));
	  un -= (remp [un - 1] == 0);	/* There should be at most one zero limb,
				   if we demand u to be normalized  */
	}
      return un;
    }
  /* if (unb - k < k/2 + k/16) // root is 2 */

  if (ulz == GMP_NUMB_BITS)
    uh = up[un - 2];
  else
    uh = (uh << ulz & GMP_NUMB_MASK) | up[un - 1 - (un != 1)] >> (GMP_NUMB_BITS - ulz);
  ASSERT (un != 1 || up[un - 1 - (un != 1)] >> (GMP_NUMB_BITS - ulz) == 1);

  xnb = logbased_root (rootp, uh, unb, k);
  snb = LOGROOT_RETURNED_BITS - 1;
  /* xnb+1 is the number of bits of the root R */
  /* snb+1 is the number of bits of the current approximation S */

  kk = k * xnb;		/* number of truncated bits in the input */

  /* FIXME: Should we skip the next two loops when xnb <= snb ? */
  for (uh = (k - 1) / 2, logk = 3; (uh >>= 1) != 0; ++logk )
    ;
  /* logk = ceil(log(k)/log(2)) + 1 */

  /* xnb is the number of remaining bits to determine in the kth root */
  for (ni = 0; (sizes[ni] = xnb) > snb; ++ni)
    {
      /* invariant: here we want xnb+1 total bits for the kth root */

      /* if c is the new value of xnb, this means that we'll go from a
	 root of c+1 bits (say s') to a root of xnb+1 bits.
	 It is proved in the book "Modern Computer Arithmetic" by Brent
	 and Zimmermann, Chapter 1, that
	 if s' >= k*beta, then at most one correction is necessary.
	 Here beta = 2^(xnb-c), and s' >= 2^c, thus it suffices that
	 c >= ceil((xnb + log2(k))/2). */
      if (xnb > logk)
	xnb = (xnb + logk) / 2;
      else
	--xnb;	/* add just one bit at a time */
    }

  *rootp >>= snb - xnb;
  kk -= xnb;

  ASSERT_ALWAYS (ni < GMP_NUMB_BITS + 1);
  /* We have sizes[0] = b > sizes[1] > ... > sizes[ni] = 0 with
     sizes[i] <= 2 * sizes[i+1].
     Newton iteration will first compute sizes[ni-1] extra bits,
     then sizes[ni-2], ..., then sizes[0] = b. */

  TMP_MARK;
  /* qp and wp need enough space to store S'^k where S' is an approximate
     root. Since S' can be as large as S+2, the worst case is when S=2 and
     S'=4. But then since we know the number of bits of S in advance, S'
     can only be 3 at most. Similarly for S=4, then S' can be 6 at most.
     So the worst case is S'/S=3/2, thus S'^k <= (3/2)^k * S^k. Since S^k
     fits in un limbs, the number of extra limbs needed is bounded by
     ceil(k*log2(3/2)/GMP_NUMB_BITS). */
  /* THINK: with the use of logbased_root, maybe the constant is
     258/256 instead of 3/2 ? log2(258/256) < 1/89 < 1/64 */
#define EXTRA 2 + (mp_size_t) (0.585 * (double) k / (double) GMP_NUMB_BITS)
  TMP_ALLOC_LIMBS_3 (scratch, un + 1, /* used by mpn_div_q */
		     qp, un + EXTRA,  /* will contain quotient and remainder
					 of R/(k*S^(k-1)), and S^k */
		     wp, un + EXTRA); /* will contain S^(k-1), k*S^(k-1),
					 and temporary for mpn_pow_1 */

  if (remp == NULL)
    rp = scratch;	/* will contain the remainder */
  else
    rp = remp;
  sp = rootp;

  sn = 1;		/* Initial approximation has one limb */

  for (b = xnb; ni != 0; --ni)
    {
      /* 1: loop invariant:
	 {sp, sn} is the current approximation of the root, which has
		  exactly 1 + sizes[ni] bits.
	 {rp, rn} is the current remainder
	 {wp, wn} = {sp, sn}^(k-1)
	 kk = number of truncated bits of the input
      */

      /* Since each iteration treats b bits from the root and thus k*b bits
	 from the input, and we already considered b bits from the input,
	 we now have to take another (k-1)*b bits from the input. */
      kk -= (k - 1) * b; /* remaining input bits */
      /* {rp, rn} = floor({up, un} / 2^kk) */
      rn = un - kk / GMP_NUMB_BITS;
      MPN_RSHIFT (rp, up + kk / GMP_NUMB_BITS, rn, kk % GMP_NUMB_BITS);
      rn -= rp[rn - 1] == 0;

      /* 9: current buffers: {sp,sn}, {rp,rn} */

      for (c = 0;; c++)
	{
	  /* Compute S^k in {qp,qn}. */
	  /* W <- S^(k-1) for the next iteration,
	     and S^k = W * S. */
	  wn = mpn_pow_1 (wp, sp, sn, k - 1, qp);
	  mpn_mul (qp, wp, wn, sp, sn);
	  qn = wn + sn;
	  qn -= qp[qn - 1] == 0;

	  perf_pow = 1;
	  /* if S^k > floor(U/2^kk), the root approximation was too large */
	  if (qn > rn || (qn == rn && (perf_pow=mpn_cmp (qp, rp, rn)) > 0))
	    MPN_DECR_U (sp, sn, 1);
	  else
	    break;
	}

      /* 10: current buffers: {sp,sn}, {rp,rn}, {qp,qn}, {wp,wn} */

      /* sometimes two corrections are needed with logbased_root*/
      ASSERT (c <= 1 + LOGROOT_NEEDS_TWO_CORRECTIONS);
      ASSERT_ALWAYS (rn >= qn);

      b = sizes[ni - 1] - sizes[ni]; /* number of bits to compute in the
				      next iteration */
      bn = b / GMP_NUMB_BITS; /* lowest limb from high part of rp[], after shift */

      kk = kk - b;
      /* nl is the number of limbs in U which contain bits [kk,kk+b-1] */
      nl = 1 + (kk + b - 1) / GMP_NUMB_BITS - (kk / GMP_NUMB_BITS);
      /* nl  = 1 + floor((kk + b - 1) / GMP_NUMB_BITS)
		 - floor(kk / GMP_NUMB_BITS)
	     <= 1 + (kk + b - 1) / GMP_NUMB_BITS
		  - (kk - GMP_NUMB_BITS + 1) / GMP_NUMB_BITS
	     = 2 + (b - 2) / GMP_NUMB_BITS
	 thus since nl is an integer:
	 nl <= 2 + floor(b/GMP_NUMB_BITS) <= 2 + bn. */

      /* 11: current buffers: {sp,sn}, {rp,rn}, {wp,wn} */

      /* R = R - Q = floor(U/2^kk) - S^k */
      if (perf_pow != 0)
	{
	  mpn_sub (rp, rp, rn, qp, qn);
	  MPN_NORMALIZE_NOT_ZERO (rp, rn);

	  /* first multiply the remainder by 2^b */
	  MPN_LSHIFT (cy, rp + bn, rp, rn, b % GMP_NUMB_BITS);
	  rn = rn + bn;
	  if (cy != 0)
	    {
	      rp[rn] = cy;
	      rn++;
	    }

	  save = rp[bn];
	  /* we have to save rp[bn] up to rp[nl-1], i.e. 1 or 2 limbs */
	  if (nl - 1 > bn)
	    save2 = rp[bn + 1];
	}
      else
	{
	  rn = bn;
	  save2 = save = 0;
	}
      /* 2: current buffers: {sp,sn}, {rp,rn}, {wp,wn} */

      /* Now insert bits [kk,kk+b-1] from the input U */
      MPN_RSHIFT (rp, up + kk / GMP_NUMB_BITS, nl, kk % GMP_NUMB_BITS);
      /* set to zero high bits of rp[bn] */
      rp[bn] &= (CNST_LIMB (1) << (b % GMP_NUMB_BITS)) - 1;
      /* restore corresponding bits */
      rp[bn] |= save;
      if (nl - 1 > bn)
	rp[bn + 1] = save2; /* the low b bits go in rp[0..bn] only, since
			       they start by bit 0 in rp[0], so they use
			       at most ceil(b/GMP_NUMB_BITS) limbs */
      /* FIXME: Should we normalise {rp,rn} here ?*/

      /* 3: current buffers: {sp,sn}, {rp,rn}, {wp,wn} */

      /* compute {wp, wn} = k * {sp, sn}^(k-1) */
      cy = mpn_mul_1 (wp, wp, wn, k);
      wp[wn] = cy;
      wn += cy != 0;

      /* 6: current buffers: {sp,sn}, {qp,qn} */

      /* multiply the root approximation by 2^b */
      MPN_LSHIFT (cy, sp + b / GMP_NUMB_BITS, sp, sn, b % GMP_NUMB_BITS);
      sn = sn + b / GMP_NUMB_BITS;
      if (cy != 0)
	{
	  sp[sn] = cy;
	  sn++;
	}

      save = sp[b / GMP_NUMB_BITS];

      /* Number of limbs used by b bits, when least significant bit is
	 aligned to least limb */
      bn = (b - 1) / GMP_NUMB_BITS + 1;

      /* 4: current buffers: {sp,sn}, {rp,rn}, {wp,wn} */

      /* now divide {rp, rn} by {wp, wn} to get the low part of the root */
      if (UNLIKELY (rn < wn))
	{
	  MPN_FILL (sp, bn, 0);
	}
      else
	{
	  qn = rn - wn; /* expected quotient size */
	  if (qn <= bn) { /* Divide only if result is not too big. */
	    mpn_div_q (qp, rp, rn, wp, wn, scratch);
	    qn += qp[qn] != 0;
	  }

      /* 5: current buffers: {sp,sn}, {qp,qn}.
	 Note: {rp,rn} is not needed any more since we'll compute it from
	 scratch at the end of the loop.
       */

      /* the quotient should be smaller than 2^b, since the previous
	 approximation was correctly rounded toward zero */
	  if (qn > bn || (qn == bn && (b % GMP_NUMB_BITS != 0) &&
			  qp[qn - 1] >= (CNST_LIMB (1) << (b % GMP_NUMB_BITS))))
	    {
	      for (qn = 1; qn < bn; ++qn)
		sp[qn - 1] = GMP_NUMB_MAX;
	      sp[qn - 1] = GMP_NUMB_MAX >> (GMP_NUMB_BITS - 1 - ((b - 1) % GMP_NUMB_BITS));
	    }
	  else
	    {
      /* 7: current buffers: {sp,sn}, {qp,qn} */

      /* Combine sB and q to form sB + q.  */
	      MPN_COPY (sp, qp, qn);
	      MPN_ZERO (sp + qn, bn - qn);
	    }
	}
      sp[b / GMP_NUMB_BITS] |= save;

      /* 8: current buffer: {sp,sn} */

    };

  /* otherwise we have rn > 0, thus the return value is ok */
  if (!approx || sp[0] <= CNST_LIMB (1))
    {
      for (c = 0;; c++)
	{
	  /* Compute S^k in {qp,qn}. */
	  /* Last iteration: we don't need W anymore. */
	  /* mpn_pow_1 requires that both qp and wp have enough
	     space to store the result {sp,sn}^k + 1 limb */
	  qn = mpn_pow_1 (qp, sp, sn, k, wp);

	  perf_pow = 1;
	  if (qn > un || (qn == un && (perf_pow=mpn_cmp (qp, up, un)) > 0))
	    MPN_DECR_U (sp, sn, 1);
	  else
	    break;
	};

      /* sometimes two corrections are needed with logbased_root*/
      ASSERT (c <= 1 + LOGROOT_NEEDS_TWO_CORRECTIONS);

      rn = perf_pow != 0;
      if (rn != 0 && remp != NULL)
	{
	  mpn_sub (remp, up, un, qp, qn);
	  rn = un;
	  MPN_NORMALIZE_NOT_ZERO (remp, rn);
	}
    }

  TMP_FREE;
  return rn;
}
