/* mpn_perfect_power_p -- mpn perfect power detection.

   Contributed to the GNU project by Martin Boij.

Copyright 2009, 2010, 2012, 2014 Free Software Foundation, Inc.

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

#define SMALL 20
#define MEDIUM 100

/* Return non-zero if {np,nn} == {xp,xn} ^ k.
   Algorithm:
       For s = 1, 2, 4, ..., s_max, compute the s least significant limbs of
       {xp,xn}^k. Stop if they don't match the s least significant limbs of
       {np,nn}.

   FIXME: Low xn limbs can be expected to always match, if computed as a mod
   B^{xn} root. So instead of using mpn_powlo, compute an approximation of the
   most significant (normalized) limb of {xp,xn} ^ k (and an error bound), and
   compare to {np, nn}. Or use an even cruder approximation based on fix-point
   base 2 logarithm.  */
static int
pow_equals (mp_srcptr np, mp_size_t n,
	    mp_srcptr xp,mp_size_t xn,
	    mp_limb_t k, mp_bitcnt_t f,
	    mp_ptr tp)
{
  mp_bitcnt_t y, z;
  mp_size_t bn;
  mp_limb_t h, l;

  ASSERT (n > 1 || (n == 1 && np[0] > 1));
  ASSERT (np[n - 1] > 0);
  ASSERT (xn > 0);

  if (xn == 1 && xp[0] == 1)
    return 0;

  z = 1 + (n >> 1);
  for (bn = 1; bn < z; bn <<= 1)
    {
      mpn_powlo (tp, xp, &k, 1, bn, tp + bn);
      if (mpn_cmp (tp, np, bn) != 0)
	return 0;
    }

  /* Final check. Estimate the size of {xp,xn}^k before computing the power
     with full precision.  Optimization: It might pay off to make a more
     accurate estimation of the logarithm of {xp,xn}, rather than using the
     index of the MSB.  */

  MPN_SIZEINBASE_2EXP(y, xp, xn, 1);
  y -= 1;  /* msb_index (xp, xn) */

  umul_ppmm (h, l, k, y);
  h -= l == 0;  --l;	/* two-limb decrement */

  z = f - 1; /* msb_index (np, n) */
  if (h == 0 && l <= z)
    {
      mp_limb_t *tp2;
      mp_size_t i;
      int ans;
      mp_limb_t size;
      TMP_DECL;

      size = l + k;
      ASSERT_ALWAYS (size >= k);

      TMP_MARK;
      y = 2 + size / GMP_LIMB_BITS;
      tp2 = TMP_ALLOC_LIMBS (y);

      i = mpn_pow_1 (tp, xp, xn, k, tp2);
      if (i == n && mpn_cmp (tp, np, n) == 0)
	ans = 1;
      else
	ans = 0;
      TMP_FREE;
      return ans;
    }

  return 0;
}


/* Return non-zero if N = {np,n} is a kth power.
   I = {ip,n} = N^(-1) mod B^n.  */
static int
is_kth_power (mp_ptr rp, mp_srcptr np,
	      mp_limb_t k, mp_srcptr ip,
	      mp_size_t n, mp_bitcnt_t f,
	      mp_ptr tp)
{
  mp_bitcnt_t b;
  mp_size_t rn, xn;

  ASSERT (n > 0);
  ASSERT ((k & 1) != 0 || k == 2);
  ASSERT ((np[0] & 1) != 0);

  if (k == 2)
    {
      b = (f + 1) >> 1;
      rn = 1 + b / GMP_LIMB_BITS;
      if (mpn_bsqrtinv (rp, ip, b, tp) != 0)
	{
	  rp[rn - 1] &= (CNST_LIMB(1) << (b % GMP_LIMB_BITS)) - 1;
	  xn = rn;
	  MPN_NORMALIZE (rp, xn);
	  if (pow_equals (np, n, rp, xn, k, f, tp) != 0)
	    return 1;

	  /* Check if (2^b - r)^2 == n */
	  mpn_neg (rp, rp, rn);
	  rp[rn - 1] &= (CNST_LIMB(1) << (b % GMP_LIMB_BITS)) - 1;
	  MPN_NORMALIZE (rp, rn);
	  if (pow_equals (np, n, rp, rn, k, f, tp) != 0)
	    return 1;
	}
    }
  else
    {
      b = 1 + (f - 1) / k;
      rn = 1 + (b - 1) / GMP_LIMB_BITS;
      mpn_brootinv (rp, ip, rn, k, tp);
      if ((b % GMP_LIMB_BITS) != 0)
	rp[rn - 1] &= (CNST_LIMB(1) << (b % GMP_LIMB_BITS)) - 1;
      MPN_NORMALIZE (rp, rn);
      if (pow_equals (np, n, rp, rn, k, f, tp) != 0)
	return 1;
    }
  MPN_ZERO (rp, rn); /* Untrash rp */
  return 0;
}

static int
perfpow (mp_srcptr np, mp_size_t n,
	 mp_limb_t ub, mp_limb_t g,
	 mp_bitcnt_t f, int neg)
{
  mp_ptr ip, tp, rp;
  mp_limb_t k;
  int ans;
  mp_bitcnt_t b;
  gmp_primesieve_t ps;
  TMP_DECL;

  ASSERT (n > 0);
  ASSERT ((np[0] & 1) != 0);
  ASSERT (ub > 0);

  TMP_MARK;
  gmp_init_primesieve (&ps);
  b = (f + 3) >> 1;

  TMP_ALLOC_LIMBS_3 (ip, n, rp, n, tp, 5 * n);

  MPN_ZERO (rp, n);

  /* FIXME: It seems the inverse in ninv is needed only to get non-inverted
     roots. I.e., is_kth_power computes n^{1/2} as (n^{-1})^{-1/2} and
     similarly for nth roots. It should be more efficient to compute n^{1/2} as
     n * n^{-1/2}, with a mullo instead of a binvert. And we can do something
     similar for kth roots if we switch to an iteration converging to n^{1/k -
     1}, and we can then eliminate this binvert call. */
  mpn_binvert (ip, np, 1 + (b - 1) / GMP_LIMB_BITS, tp);
  if (b % GMP_LIMB_BITS)
    ip[(b - 1) / GMP_LIMB_BITS] &= (CNST_LIMB(1) << (b % GMP_LIMB_BITS)) - 1;

  if (neg)
    gmp_nextprime (&ps);

  ans = 0;
  if (g > 0)
    {
      ub = MIN (ub, g + 1);
      while ((k = gmp_nextprime (&ps)) < ub)
	{
	  if ((g % k) == 0)
	    {
	      if (is_kth_power (rp, np, k, ip, n, f, tp) != 0)
		{
		  ans = 1;
		  goto ret;
		}
	    }
	}
    }
  else
    {
      while ((k = gmp_nextprime (&ps)) < ub)
	{
	  if (is_kth_power (rp, np, k, ip, n, f, tp) != 0)
	    {
	      ans = 1;
	      goto ret;
	    }
	}
    }
 ret:
  TMP_FREE;
  return ans;
}

static const unsigned short nrtrial[] = { 100, 500, 1000 };

/* Table of (log_{p_i} 2) values, where p_i is the (nrtrial[i] + 1)'th prime
   number.  */
static const double logs[] =
  { 0.1099457228193620, 0.0847016403115322, 0.0772048195144415 };

int
mpn_perfect_power_p (mp_srcptr np, mp_size_t n)
{
  mp_limb_t *nc, factor, g;
  mp_limb_t exp, d;
  mp_bitcnt_t twos, count;
  int ans, where, neg, trial;
  TMP_DECL;

  neg = n < 0;
  if (neg)
    {
      n = -n;
    }

  if (n == 0 || (n == 1 && np[0] == 1)) /* Valgrind doesn't like
					   (n <= (np[0] == 1)) */
    return 1;

  TMP_MARK;

  count = 0;

  twos = mpn_scan1 (np, 0);
  if (twos != 0)
    {
      mp_size_t s;
      if (twos == 1)
	{
	  return 0;
	}
      s = twos / GMP_LIMB_BITS;
      if (s + 1 == n && POW2_P (np[s]))
	{
	  return ! (neg && POW2_P (twos));
	}
      count = twos % GMP_LIMB_BITS;
      n -= s;
      np += s;
      if (count > 0)
	{
	  nc = TMP_ALLOC_LIMBS (n);
	  mpn_rshift (nc, np, n, count);
	  n -= (nc[n - 1] == 0);
	  np = nc;
	}
    }
  g = twos;

  trial = (n > SMALL) + (n > MEDIUM);

  where = 0;
  factor = mpn_trialdiv (np, n, nrtrial[trial], &where);

  if (factor != 0)
    {
      if (count == 0) /* We did not allocate nc yet. */
	{
	  nc = TMP_ALLOC_LIMBS (n);
	}

      /* Remove factors found by trialdiv.  Optimization: If remove
	 define _itch, we can allocate its scratch just once */

      do
	{
	  binvert_limb (d, factor);

	  /* After the first round we always have nc == np */
	  exp = mpn_remove (nc, &n, np, n, &d, 1, ~(mp_bitcnt_t)0);

	  if (g == 0)
	    g = exp;
	  else
	    g = mpn_gcd_1 (&g, 1, exp);

	  if (g == 1)
	    {
	      ans = 0;
	      goto ret;
	    }

	  if ((n == 1) & (nc[0] == 1))
	    {
	      ans = ! (neg && POW2_P (g));
	      goto ret;
	    }

	  np = nc;
	  factor = mpn_trialdiv (np, n, nrtrial[trial], &where);
	}
      while (factor != 0);
    }

  MPN_SIZEINBASE_2EXP(count, np, n, 1);   /* log (np) + 1 */
  d = (mp_limb_t) (count * logs[trial] + 1e-9) + 1;
  ans = perfpow (np, n, d, g, count, neg);

 ret:
  TMP_FREE;
  return ans;
}
