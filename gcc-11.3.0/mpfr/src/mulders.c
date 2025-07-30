/* Mulders' MulHigh function (short product)

Copyright 2005-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

/* References:
   [1] Short Division of Long Integers, David Harvey and Paul Zimmermann,
       Proceedings of the 20th Symposium on Computer Arithmetic (ARITH-20),
       July 25-27, 2011, pages 7-14.
*/

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

#ifndef MUL_FFT_THRESHOLD
#define MUL_FFT_THRESHOLD 8448
#endif

/* Don't use MPFR_MULHIGH_SIZE since it is handled by tuneup */
#ifdef MPFR_MULHIGH_TAB_SIZE
static short mulhigh_ktab[MPFR_MULHIGH_TAB_SIZE];
#else
static short mulhigh_ktab[] = {MPFR_MULHIGH_TAB};
#define MPFR_MULHIGH_TAB_SIZE \
  ((mp_size_t) (sizeof(mulhigh_ktab) / sizeof(mulhigh_ktab[0])))
#endif

/* Put in  rp[n..2n-1] an approximation of the n high limbs
   of {up, n} * {vp, n}. The error is less than n ulps of rp[n] (and the
   approximation is always less or equal to the truncated full product).
   Assume 2n limbs are allocated at rp.

   Implements Algorithm ShortMulNaive from [1].
*/
static void
mpfr_mulhigh_n_basecase (mpfr_limb_ptr rp, mpfr_limb_srcptr up,
                         mpfr_limb_srcptr vp, mp_size_t n)
{
  mp_size_t i;

  rp += n - 1;
  umul_ppmm (rp[1], rp[0], up[n-1], vp[0]); /* we neglect up[0..n-2]*vp[0],
                                               which is less than B^n */
  for (i = 1 ; i < n ; i++)
    /* here, we neglect up[0..n-i-2] * vp[i], which is less than B^n too */
    rp[i + 1] = mpn_addmul_1 (rp, up + (n - i - 1), i + 1, vp[i]);
  /* in total, we neglect less than n*B^n, i.e., n ulps of rp[n]. */
}

/* Put in  rp[0..n] the n+1 low limbs of {up, n} * {vp, n}.
   Assume 2n limbs are allocated at rp. */
static void
mpfr_mullow_n_basecase (mpfr_limb_ptr rp, mpfr_limb_srcptr up,
                        mpfr_limb_srcptr vp, mp_size_t n)
{
  mp_size_t i;

  rp[n] = mpn_mul_1 (rp, up, n, vp[0]);
  for (i = 1 ; i < n ; i++)
    mpn_addmul_1 (rp + i, up, n - i + 1, vp[i]);
}

/* Put in  rp[n..2n-1] an approximation of the n high limbs
   of {np, n} * {mp, n}. The error is less than n ulps of rp[n] (and the
   approximation is always less or equal to the truncated full product).

   Implements Algorithm ShortMul from [1].
*/
void
mpfr_mulhigh_n (mpfr_limb_ptr rp, mpfr_limb_srcptr np, mpfr_limb_srcptr mp,
                mp_size_t n)
{
  mp_size_t k;

  MPFR_ASSERTN (MPFR_MULHIGH_TAB_SIZE >= 8); /* so that 3*(n/4) > n/2 */
  k = MPFR_LIKELY (n < MPFR_MULHIGH_TAB_SIZE) ? mulhigh_ktab[n] : 3*(n/4);
  /* Algorithm ShortMul from [1] requires k >= (n+3)/2, which translates
     into k >= (n+4)/2 in the C language. */
  MPFR_ASSERTD (k == -1 || k == 0 || (k >= (n+4)/2 && k < n));
  if (k < 0)
    mpn_mul_basecase (rp, np, n, mp, n); /* result is exact, no error */
  else if (k == 0)
    mpfr_mulhigh_n_basecase (rp, np, mp, n); /* basecase error < n ulps */
  else if (n > MUL_FFT_THRESHOLD)
    mpn_mul_n (rp, np, mp, n); /* result is exact, no error */
  else
    {
      mp_size_t l = n - k;
      mp_limb_t cy;

      mpn_mul_n (rp + 2 * l, np + l, mp + l, k); /* fills rp[2l..2n-1] */
      mpfr_mulhigh_n (rp, np + k, mp, l);        /* fills rp[l-1..2l-1] */
      cy = mpn_add_n (rp + n - 1, rp + n - 1, rp + l - 1, l + 1);
      mpfr_mulhigh_n (rp, np, mp + k, l);        /* fills rp[l-1..2l-1] */
      cy += mpn_add_n (rp + n - 1, rp + n - 1, rp + l - 1, l + 1);
      mpn_add_1 (rp + n + l, rp + n + l, k, cy); /* propagate carry */
    }
}

/* Put in  rp[0..n] the n+1 low limbs of {np, n} * {mp, n}.
   Assume 2n limbs are allocated at rp. */
void
mpfr_mullow_n (mpfr_limb_ptr rp, mpfr_limb_srcptr np, mpfr_limb_srcptr mp,
               mp_size_t n)
{
  mp_size_t k;

  MPFR_ASSERTN (MPFR_MULHIGH_TAB_SIZE >= 8); /* so that 3*(n/4) > n/2 */
  k = MPFR_LIKELY (n < MPFR_MULHIGH_TAB_SIZE) ? mulhigh_ktab[n] : 3*(n/4);
  MPFR_ASSERTD (k == -1 || k == 0 || (2 * k >= n && k < n));
  if (k < 0)
    mpn_mul_basecase (rp, np, n, mp, n);
  else if (k == 0)
    mpfr_mullow_n_basecase (rp, np, mp, n);
  else if (n > MUL_FFT_THRESHOLD)
    mpn_mul_n (rp, np, mp, n);
  else
    {
      mp_size_t l = n - k;

      mpn_mul_n (rp, np, mp, k);                      /* fills rp[0..2k] */
      mpfr_mullow_n (rp + n, np + k, mp, l);          /* fills rp[n..n+2l] */
      mpn_add_n (rp + k, rp + k, rp + n, l + 1);
      mpfr_mullow_n (rp + n, np, mp + k, l);          /* fills rp[n..n+2l] */
      mpn_add_n (rp + k, rp + k, rp + n, l + 1);
    }
}

#ifdef MPFR_SQRHIGH_TAB_SIZE
static short sqrhigh_ktab[MPFR_SQRHIGH_TAB_SIZE];
#else
static short sqrhigh_ktab[] = {MPFR_SQRHIGH_TAB};
#define MPFR_SQRHIGH_TAB_SIZE (sizeof(sqrhigh_ktab) / sizeof(sqrhigh_ktab[0]))
#endif

/* Put in  rp[n..2n-1] an approximation of the n high limbs
   of {np, n}^2. The error is less than n ulps of rp[n]. */
void
mpfr_sqrhigh_n (mpfr_limb_ptr rp, mpfr_limb_srcptr np, mp_size_t n)
{
  mp_size_t k;

  MPFR_ASSERTN (MPFR_SQRHIGH_TAB_SIZE > 2); /* ensures k < n */
  k = MPFR_LIKELY (n < MPFR_SQRHIGH_TAB_SIZE) ? sqrhigh_ktab[n]
    : (n+4)/2; /* ensures that k >= (n+3)/2 */
  MPFR_ASSERTD (k == -1 || k == 0 || (k >= (n+4)/2 && k < n));
  if (k < 0)
    /* we can't use mpn_sqr_basecase here, since it requires
       n <= SQR_KARATSUBA_THRESHOLD, where SQR_KARATSUBA_THRESHOLD
       is not exported by GMP */
    mpn_sqr_n (rp, np, n);
  else if (k == 0)
    mpfr_mulhigh_n_basecase (rp, np, np, n);
  else
    {
      mp_size_t l = n - k;
      mp_limb_t cy;

      mpn_sqr_n (rp + 2 * l, np + l, k);          /* fills rp[2l..2n-1] */
      mpfr_mulhigh_n (rp, np, np + k, l);         /* fills rp[l-1..2l-1] */
      /* {rp+n-1,l+1} += 2 * {rp+l-1,l+1} */
      cy = mpn_lshift (rp + l - 1, rp + l - 1, l + 1, 1);
      cy += mpn_add_n (rp + n - 1, rp + n - 1, rp + l - 1, l + 1);
      mpn_add_1 (rp + n + l, rp + n + l, k, cy); /* propagate carry */
    }
}

#ifdef MPFR_DIVHIGH_TAB_SIZE
static short divhigh_ktab[MPFR_DIVHIGH_TAB_SIZE];
#else
static short divhigh_ktab[] = {MPFR_DIVHIGH_TAB};
#define MPFR_DIVHIGH_TAB_SIZE (sizeof(divhigh_ktab) / sizeof(divhigh_ktab[0]))
#endif

#ifndef __GMPFR_GMP_H__
#define mpfr_pi1_t gmp_pi1_t /* with a GMP build */
#endif

#if !(defined(WANT_GMP_INTERNALS) && defined(HAVE___GMPN_SBPI1_DIVAPPR_Q))
/* Put in Q={qp, n} an approximation of N={np, 2*n} divided by D={dp, n},
   with the most significant limb of the quotient as return value (0 or 1).
   Assumes the most significant bit of D is set. Clobbers N.

   The approximate quotient Q satisfies - 2(n-1) < N/D - Q <= 4.
*/
static mp_limb_t
mpfr_divhigh_n_basecase (mpfr_limb_ptr qp, mpfr_limb_ptr np,
                         mpfr_limb_srcptr dp, mp_size_t n)
{
  mp_limb_t qh, d1, d0, dinv, q2, q1, q0;
  mpfr_pi1_t dinv2;

  np += n;

  if ((qh = (mpn_cmp (np, dp, n) >= 0)))
    mpn_sub_n (np, np, dp, n);

  /* now {np, n} is less than D={dp, n}, which implies np[n-1] <= dp[n-1] */

  d1 = dp[n - 1];

  if (n == 1)
    {
      invert_limb (dinv, d1);
      umul_ppmm (q1, q0, np[0], dinv);
      qp[0] = np[0] + q1;
      return qh;
    }

  /* now n >= 2 */
  d0 = dp[n - 2];
  invert_pi1 (dinv2, d1, d0);
  /* dinv2.inv32 = floor ((B^3 - 1) / (d0 + d1 B)) - B */
  while (n > 1)
    {
      /* Invariant: it remains to reduce n limbs from N (in addition to the
         initial low n limbs).
         Since n >= 2 here, necessarily we had n >= 2 initially, which means
         that in addition to the limb np[n-1] to reduce, we have at least 2
         extra limbs, thus accessing np[n-3] is valid. */

      /* Warning: we can have np[n-1]>d1 or (np[n-1]=d1 and np[n-2]>=d0) here,
         since we truncate the divisor at each step, but since {np,n} < D
         originally, the largest possible partial quotient is B-1. */
      if (MPFR_UNLIKELY(np[n-1] > d1 || (np[n-1] == d1 && np[n-2] >= d0)))
        q2 = ~ (mp_limb_t) 0;
      else
        udiv_qr_3by2 (q2, q1, q0, np[n - 1], np[n - 2], np[n - 3],
                      d1, d0, dinv2.inv32);
      /* since q2 = floor((np[n-1]*B^2+np[n-2]*B+np[n-3])/(d1*B+d0)),
         we have q2 <= (np[n-1]*B^2+np[n-2]*B+np[n-3])/(d1*B+d0),
         thus np[n-1]*B^2+np[n-2]*B+np[n-3] >= q2*(d1*B+d0)
         and {np-1, n} >= q2*D - q2*B^(n-2) >= q2*D - B^(n-1)
         thus {np-1, n} - (q2-1)*D >= D - B^(n-1) >= 0
         which proves that at most one correction is needed */
      q0 = mpn_submul_1 (np - 1, dp, n, q2);
      if (MPFR_UNLIKELY(q0 > np[n - 1]))
        {
          mpn_add_n (np - 1, np - 1, dp, n);
          q2 --;
        }
      qp[--n] = q2;
      dp ++;
    }

  /* we have B+dinv2 = floor((B^3-1)/(d1*B+d0)) < B^2/d1
     q1 = floor(np[0]*(B+dinv2)/B) <= floor(np[0]*B/d1)
        <= floor((np[0]*B+np[1])/d1)
     thus q1 is not larger than the true quotient.
     q1 > np[0]*(B+dinv2)/B - 1 > np[0]*(B^3-1)/(d1*B+d0)/B - 2
     For d1*B+d0 <> B^2/2, we have B+dinv2 = floor(B^3/(d1*B+d0))
     thus q1 > np[0]*B^2/(d1*B+d0) - 2, i.e.,
     (d1*B+d0)*q1 > np[0]*B^2 - 2*(d1*B+d0)
     d1*B*q1 > np[0]*B^2 - 2*d1*B - 2*d0 - d0*q1 >= np[0]*B^2 - 2*d1*B - B^2
     thus q1 > np[0]*B/d1 - 2 - B/d1 > np[0]*B/d1 - 4.

     For d1*B+d0 = B^2/2, dinv2 = B-1 thus q1 > np[0]*(2B-1)/B - 1 >
     np[0]*B/d1 - 2.

     In all cases, if q = floor((np[0]*B+np[1])/d1), we have:
     q - 4 <= q1 <= q
  */
  umul_ppmm (q1, q0, np[0], dinv2.inv32);
  qp[0] = np[0] + q1;

  return qh;
}
#endif

/* Put in {qp, n} an approximation of N={np, 2*n} divided by D={dp, n},
   with the most significant limb of the quotient as return value (0 or 1).
   Assumes the most significant bit of D is set. Clobbers N.

   This implements the ShortDiv algorithm from reference [1].
*/
#if 1
mp_limb_t
mpfr_divhigh_n (mpfr_limb_ptr qp, mpfr_limb_ptr np, mpfr_limb_ptr dp,
                mp_size_t n)
{
  mp_size_t k, l;
  mp_limb_t qh, cy;
  mpfr_limb_ptr tp;
  MPFR_TMP_DECL(marker);

  MPFR_ASSERTN (MPFR_MULHIGH_TAB_SIZE >= 15); /* so that 2*(n/3) >= (n+4)/2 */
  k = MPFR_LIKELY (n < MPFR_DIVHIGH_TAB_SIZE) ? divhigh_ktab[n] : 2*(n/3);

  if (k == 0)
#if defined(WANT_GMP_INTERNALS) && defined(HAVE___GMPN_SBPI1_DIVAPPR_Q)
  {
    mpfr_pi1_t dinv2;
    invert_pi1 (dinv2, dp[n - 1], dp[n - 2]);
    return __gmpn_sbpi1_divappr_q (qp, np, n + n, dp, n, dinv2.inv32);
  }
#else /* use our own code for base-case short division */
    return mpfr_divhigh_n_basecase (qp, np, dp, n);
#endif
  else if (k == n)
    /* for k=n, we use a division with remainder (mpn_divrem),
     which computes the exact quotient */
    return mpn_divrem (qp, 0, np, 2 * n, dp, n);

  MPFR_ASSERTD ((n+4)/2 <= k && k < n); /* bounds from [1] */
  MPFR_TMP_MARK (marker);
  l = n - k;
  /* first divide the most significant 2k limbs from N by the most significant
     k limbs of D */
  qh = mpn_divrem (qp + l, 0, np + 2 * l, 2 * k, dp + l, k); /* exact */

  /* it remains {np,2l+k} = {np,n+l} as remainder */

  /* now we have to subtract high(Q1)*D0 where Q1=qh*B^k+{qp+l,k} and
     D0={dp,l} */
  tp = MPFR_TMP_LIMBS_ALLOC (2 * l);
  mpfr_mulhigh_n (tp, qp + k, dp, l);
  /* we are only interested in the upper l limbs from {tp,2l} */
  cy = mpn_sub_n (np + n, np + n, tp + l, l);
  if (qh)
    cy += mpn_sub_n (np + n, np + n, dp, l);
  while (cy > 0) /* Q1 was too large: subtract 1 to Q1 and add D to np+l */
    {
      qh -= mpn_sub_1 (qp + l, qp + l, k, MPFR_LIMB_ONE);
      cy -= mpn_add_n (np + l, np + l, dp, n);
    }

  /* now it remains {np,n+l} to divide by D */
  cy = mpfr_divhigh_n (qp, np + k, dp + k, l);
  qh += mpn_add_1 (qp + l, qp + l, k, cy);
  MPFR_TMP_FREE(marker);

  return qh;
}
#else /* below is the FoldDiv(K) algorithm from [1] */
mp_limb_t
mpfr_divhigh_n (mpfr_limb_ptr qp, mpfr_limb_ptr np, mpfr_limb_ptr dp,
                mp_size_t n)
{
  mp_size_t k, r;
  mpfr_limb_ptr ip, tp, up;
  mp_limb_t qh = 0, cy, cc;
  int count;
  MPFR_TMP_DECL(marker);

#define K 3
  if (n < K)
    return mpn_divrem (qp, 0, np, 2 * n, dp, n);

  k = (n - 1) / K + 1; /* ceil(n/K) */

  MPFR_TMP_MARK (marker);
  ip = MPFR_TMP_LIMBS_ALLOC (k + 1);
  tp = MPFR_TMP_LIMBS_ALLOC (n + k);
  up = MPFR_TMP_LIMBS_ALLOC (2 * (k + 1));
  mpn_invert (ip, dp + n - (k + 1), k + 1, NULL); /* takes about 13% for n=1000 */
  /* {ip, k+1} = floor((B^(2k+2)-1)/D - B^(k+1) where D = {dp+n-(k+1),k+1} */
  for (r = n, cc = 0UL; r > 0;)
    {
      /* cc is the carry at np[n+r] */
      MPFR_ASSERTD(cc <= 1);
      /* FIXME: why can we have cc as large as say 8? */
      count = 0;
      while (cc > 0)
        {
          count ++;
          MPFR_ASSERTD(count <= 1);
          /* subtract {dp+n-r,r} from {np+n,r} */
          cc -= mpn_sub_n (np + n, np + n, dp + n - r, r);
          /* add 1 at qp[r] */
          qh += mpn_add_1 (qp + r, qp + r, n - r, 1UL);
        }
      /* it remains r limbs to reduce, i.e., the remainder is {np, n+r} */
      if (r < k)
        {
          ip += k - r;
          k = r;
        }
      /* now r >= k */
      /* qp + r - 2 * k -> up */
      mpfr_mulhigh_n (up, np + n + r - (k + 1), ip, k + 1);
      /* take into account the term B^k in the inverse: B^k * {np+n+r-k, k} */
      cy = mpn_add_n (qp + r - k, up + k + 2, np + n + r - k, k);
      /* since we need only r limbs of tp (below), it suffices to consider
         r high limbs of dp */
      if (r > k)
        {
#if 0
          mpn_mul (tp, dp + n - r, r, qp + r - k, k);
#else  /* use a short product for the low k x k limbs */
          /* we know the upper k limbs of the r-limb product cancel with the
             remainder, thus we only need to compute the low r-k limbs */
          if (r - k >= k)
            mpn_mul (tp + k, dp + n - r + k, r - k, qp + r - k, k);
          else /* r-k < k */
            {
/* #define LOW */
#ifndef LOW
              mpn_mul (tp + k, qp + r - k, k, dp + n - r + k, r - k);
#else
              mpfr_mullow_n_basecase (tp + k, qp + r - k, dp + n - r + k, r - k);
              /* take into account qp[2r-2k] * dp[n - r + k] */
              tp[r] += qp[2*r-2*k] * dp[n - r + k];
#endif
              /* tp[k..r] is filled */
            }
#if 0
          mpfr_mulhigh_n (up, dp + n - r, qp + r - k, k);
#else /* compute one more limb. FIXME: we could add one limb of dp in the
         above, to save one mpn_addmul_1 call */
          mpfr_mulhigh_n (up, dp + n - r, qp + r - k, k - 1); /* {up,2k-2} */
          /* add {qp + r - k, k - 1} * dp[n-r+k-1] */
          up[2*k-2] = mpn_addmul_1 (up + k - 1, qp + r - k, k-1, dp[n-r+k-1]);
          /* add {dp+n-r, k} * qp[r-1] */
          up[2*k-1] = mpn_addmul_1 (up + k - 1, dp + n - r, k, qp[r-1]);
#endif
#ifndef LOW
          cc = mpn_add_n (tp + k, tp + k, up + k, k);
          mpn_add_1 (tp + 2 * k, tp + 2 * k, r - k, cc);
#else
          /* update tp[k..r] */
          if (r - k + 1 <= k)
            mpn_add_n (tp + k, tp + k, up + k, r - k + 1);
          else /* r - k >= k */
            {
              cc = mpn_add_n (tp + k, tp + k, up + k, k);
              mpn_add_1 (tp + 2 * k, tp + 2 * k, r - 2 * k + 1, cc);
            }
#endif
#endif
        }
      else /* last step: since we only want the quotient, no need to update,
              just propagate the carry cy */
        {
          MPFR_ASSERTD(r < n);
          if (cy > 0)
            qh += mpn_add_1 (qp + r, qp + r, n - r, cy);
          break;
        }
      /* subtract {tp, n+k} from {np+r-k, n+k}; however we only want to
         update {np+n, n} */
      /* we should have tp[r] = np[n+r-k] up to 1 */
      MPFR_ASSERTD(tp[r] == np[n + r - k] || tp[r] + 1 == np[n + r - k]);
#ifndef LOW
      cc = mpn_sub_n (np + n - 1, np + n - 1, tp + k - 1, r + 1); /* borrow at np[n+r] */
#else
      cc = mpn_sub_n (np + n - 1, np + n - 1, tp + k - 1, r - k + 2);
#endif
      /* if cy = 1, subtract {dp, n} from {np+r, n}, thus
         {dp+n-r,r} from {np+n,r} */
      if (cy)
        {
          if (r < n)
            cc += mpn_sub_n (np + n - 1, np + n - 1, dp + n - r - 1, r + 1);
          else
            cc += mpn_sub_n (np + n, np + n, dp + n - r, r);
          /* propagate cy */
          if (r == n)
            qh = cy;
          else
            qh += mpn_add_1 (qp + r, qp + r, n - r, cy);
        }
      /* cc is the borrow at np[n+r] */
      count = 0;
      while (cc > 0) /* quotient was too large */
        {
          count++;
          MPFR_ASSERTD (count <= 1);
          cy = mpn_add_n (np + n, np + n, dp + n - (r - k), r - k);
          cc -= mpn_add_1 (np + n + r - k, np + n + r - k, k, cy);
          qh -= mpn_sub_1 (qp + r - k, qp + r - k, n - (r - k), 1UL);
        }
      r -= k;
      cc = np[n + r];
    }
  MPFR_TMP_FREE(marker);

  return qh;
}
#endif
