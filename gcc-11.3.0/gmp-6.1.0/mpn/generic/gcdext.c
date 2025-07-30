/* mpn_gcdext -- Extended Greatest Common Divisor.

Copyright 1996, 1998, 2000-2005, 2008, 2009, 2012 Free Software Foundation,
Inc.

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

/* Computes (r;b) = (a; b) M. Result is of size n + M->n +/- 1, and
   the size is returned (if inputs are non-normalized, result may be
   non-normalized too). Temporary space needed is M->n + n.
 */
static size_t
hgcd_mul_matrix_vector (struct hgcd_matrix *M,
			mp_ptr rp, mp_srcptr ap, mp_ptr bp, mp_size_t n, mp_ptr tp)
{
  mp_limb_t ah, bh;

  /* Compute (r,b) <-- (u00 a + u10 b, u01 a + u11 b) as

     t  = u00 * a
     r  = u10 * b
     r += t;

     t  = u11 * b
     b  = u01 * a
     b += t;
  */

  if (M->n >= n)
    {
      mpn_mul (tp, M->p[0][0], M->n, ap, n);
      mpn_mul (rp, M->p[1][0], M->n, bp, n);
    }
  else
    {
      mpn_mul (tp, ap, n, M->p[0][0], M->n);
      mpn_mul (rp, bp, n, M->p[1][0], M->n);
    }

  ah = mpn_add_n (rp, rp, tp, n + M->n);

  if (M->n >= n)
    {
      mpn_mul (tp, M->p[1][1], M->n, bp, n);
      mpn_mul (bp, M->p[0][1], M->n, ap, n);
    }
  else
    {
      mpn_mul (tp, bp, n, M->p[1][1], M->n);
      mpn_mul (bp, ap, n, M->p[0][1], M->n);
    }
  bh = mpn_add_n (bp, bp, tp, n + M->n);

  n += M->n;
  if ( (ah | bh) > 0)
    {
      rp[n] = ah;
      bp[n] = bh;
      n++;
    }
  else
    {
      /* Normalize */
      while ( (rp[n-1] | bp[n-1]) == 0)
	n--;
    }

  return n;
}

#define COMPUTE_V_ITCH(n) (2*(n))

/* Computes |v| = |(g - u a)| / b, where u may be positive or
   negative, and v is of the opposite sign. max(a, b) is of size n, u and
   v at most size n, and v must have space for n+1 limbs. */
static mp_size_t
compute_v (mp_ptr vp,
	   mp_srcptr ap, mp_srcptr bp, mp_size_t n,
	   mp_srcptr gp, mp_size_t gn,
	   mp_srcptr up, mp_size_t usize,
	   mp_ptr tp)
{
  mp_size_t size;
  mp_size_t an;
  mp_size_t bn;
  mp_size_t vn;

  ASSERT (n > 0);
  ASSERT (gn > 0);
  ASSERT (usize != 0);

  size = ABS (usize);
  ASSERT (size <= n);
  ASSERT (up[size-1] > 0);

  an = n;
  MPN_NORMALIZE (ap, an);
  ASSERT (gn <= an);

  if (an >= size)
    mpn_mul (tp, ap, an, up, size);
  else
    mpn_mul (tp, up, size, ap, an);

  size += an;

  if (usize > 0)
    {
      /* |v| = -v = (u a - g) / b */

      ASSERT_NOCARRY (mpn_sub (tp, tp, size, gp, gn));
      MPN_NORMALIZE (tp, size);
      if (size == 0)
	return 0;
    }
  else
    { /* |v| = v = (g - u a) / b = (g + |u| a) / b. Since g <= a,
	 (g + |u| a) always fits in (|usize| + an) limbs. */

      ASSERT_NOCARRY (mpn_add (tp, tp, size, gp, gn));
      size -= (tp[size - 1] == 0);
    }

  /* Now divide t / b. There must be no remainder */
  bn = n;
  MPN_NORMALIZE (bp, bn);
  ASSERT (size >= bn);

  vn = size + 1 - bn;
  ASSERT (vn <= n + 1);

  mpn_divexact (vp, tp, size, bp, bn);
  vn -= (vp[vn-1] == 0);

  return vn;
}

/* Temporary storage:

   Initial division: Quotient of at most an - n + 1 <= an limbs.

   Storage for u0 and u1: 2(n+1).

   Storage for hgcd matrix M, with input ceil(n/2): 5 * ceil(n/4)

   Storage for hgcd, input (n + 1)/2: 9 n/4 plus some.

   When hgcd succeeds: 1 + floor(3n/2) for adjusting a and b, and 2(n+1) for the cofactors.

   When hgcd fails: 2n + 1 for mpn_gcdext_subdiv_step, which is less.

   For the lehmer call after the loop, Let T denote
   GCDEXT_DC_THRESHOLD. For the gcdext_lehmer call, we need T each for
   u, a and b, and 4T+3 scratch space. Next, for compute_v, we need T
   for u, T+1 for v and 2T scratch space. In all, 7T + 3 is
   sufficient for both operations.

*/

/* Optimal choice of p seems difficult. In each iteration the division
 * of work between hgcd and the updates of u0 and u1 depends on the
 * current size of the u. It may be desirable to use a different
 * choice of p in each iteration. Also the input size seems to matter;
 * choosing p = n / 3 in the first iteration seems to improve
 * performance slightly for input size just above the threshold, but
 * degrade performance for larger inputs. */
#define CHOOSE_P_1(n) ((n) / 2)
#define CHOOSE_P_2(n) ((n) / 3)

mp_size_t
mpn_gcdext (mp_ptr gp, mp_ptr up, mp_size_t *usizep,
	    mp_ptr ap, mp_size_t an, mp_ptr bp, mp_size_t n)
{
  mp_size_t talloc;
  mp_size_t scratch;
  mp_size_t matrix_scratch;
  mp_size_t ualloc = n + 1;

  struct gcdext_ctx ctx;
  mp_size_t un;
  mp_ptr u0;
  mp_ptr u1;

  mp_ptr tp;

  TMP_DECL;

  ASSERT (an >= n);
  ASSERT (n > 0);
  ASSERT (bp[n-1] > 0);

  TMP_MARK;

  /* FIXME: Check for small sizes first, before setting up temporary
     storage etc. */
  talloc = MPN_GCDEXT_LEHMER_N_ITCH(n);

  /* For initial division */
  scratch = an - n + 1;
  if (scratch > talloc)
    talloc = scratch;

  if (ABOVE_THRESHOLD (n, GCDEXT_DC_THRESHOLD))
    {
      /* For hgcd loop. */
      mp_size_t hgcd_scratch;
      mp_size_t update_scratch;
      mp_size_t p1 = CHOOSE_P_1 (n);
      mp_size_t p2 = CHOOSE_P_2 (n);
      mp_size_t min_p = MIN(p1, p2);
      mp_size_t max_p = MAX(p1, p2);
      matrix_scratch = MPN_HGCD_MATRIX_INIT_ITCH (n - min_p);
      hgcd_scratch = mpn_hgcd_itch (n - min_p);
      update_scratch = max_p + n - 1;

      scratch = matrix_scratch + MAX(hgcd_scratch, update_scratch);
      if (scratch > talloc)
	talloc = scratch;

      /* Final mpn_gcdext_lehmer_n call. Need space for u and for
	 copies of a and b. */
      scratch = MPN_GCDEXT_LEHMER_N_ITCH (GCDEXT_DC_THRESHOLD)
	+ 3*GCDEXT_DC_THRESHOLD;

      if (scratch > talloc)
	talloc = scratch;

      /* Cofactors u0 and u1 */
      talloc += 2*(n+1);
    }

  tp = TMP_ALLOC_LIMBS(talloc);

  if (an > n)
    {
      mpn_tdiv_qr (tp, ap, 0, ap, an, bp, n);

      if (mpn_zero_p (ap, n))
	{
	  MPN_COPY (gp, bp, n);
	  *usizep = 0;
	  TMP_FREE;
	  return n;
	}
    }

  if (BELOW_THRESHOLD (n, GCDEXT_DC_THRESHOLD))
    {
      mp_size_t gn = mpn_gcdext_lehmer_n(gp, up, usizep, ap, bp, n, tp);

      TMP_FREE;
      return gn;
    }

  MPN_ZERO (tp, 2*ualloc);
  u0 = tp; tp += ualloc;
  u1 = tp; tp += ualloc;

  ctx.gp = gp;
  ctx.up = up;
  ctx.usize = usizep;

  {
    /* For the first hgcd call, there are no u updates, and it makes
       some sense to use a different choice for p. */

    /* FIXME: We could trim use of temporary storage, since u0 and u1
       are not used yet. For the hgcd call, we could swap in the u0
       and u1 pointers for the relevant matrix elements. */

    struct hgcd_matrix M;
    mp_size_t p = CHOOSE_P_1 (n);
    mp_size_t nn;

    mpn_hgcd_matrix_init (&M, n - p, tp);
    nn = mpn_hgcd (ap + p, bp + p, n - p, &M, tp + matrix_scratch);
    if (nn > 0)
      {
	ASSERT (M.n <= (n - p - 1)/2);
	ASSERT (M.n + p <= (p + n - 1) / 2);

	/* Temporary storage 2 (p + M->n) <= p + n - 1 */
	n = mpn_hgcd_matrix_adjust (&M, p + nn, ap, bp, p, tp + matrix_scratch);

	MPN_COPY (u0, M.p[1][0], M.n);
	MPN_COPY (u1, M.p[1][1], M.n);
	un = M.n;
	while ( (u0[un-1] | u1[un-1] ) == 0)
	  un--;
      }
    else
      {
	/* mpn_hgcd has failed. Then either one of a or b is very
	   small, or the difference is very small. Perform one
	   subtraction followed by one division. */
	u1[0] = 1;

	ctx.u0 = u0;
	ctx.u1 = u1;
	ctx.tp = tp + n; /* ualloc */
	ctx.un = 1;

	/* Temporary storage n */
	n = mpn_gcd_subdiv_step (ap, bp, n, 0, mpn_gcdext_hook, &ctx, tp);
	if (n == 0)
	  {
	    TMP_FREE;
	    return ctx.gn;
	  }

	un = ctx.un;
	ASSERT (un < ualloc);
      }
  }

  while (ABOVE_THRESHOLD (n, GCDEXT_DC_THRESHOLD))
    {
      struct hgcd_matrix M;
      mp_size_t p = CHOOSE_P_2 (n);
      mp_size_t nn;

      mpn_hgcd_matrix_init (&M, n - p, tp);
      nn = mpn_hgcd (ap + p, bp + p, n - p, &M, tp + matrix_scratch);
      if (nn > 0)
	{
	  mp_ptr t0;

	  t0 = tp + matrix_scratch;
	  ASSERT (M.n <= (n - p - 1)/2);
	  ASSERT (M.n + p <= (p + n - 1) / 2);

	  /* Temporary storage 2 (p + M->n) <= p + n - 1 */
	  n = mpn_hgcd_matrix_adjust (&M, p + nn, ap, bp, p, t0);

	  /* By the same analysis as for mpn_hgcd_matrix_mul */
	  ASSERT (M.n + un <= ualloc);

	  /* FIXME: This copying could be avoided by some swapping of
	   * pointers. May need more temporary storage, though. */
	  MPN_COPY (t0, u0, un);

	  /* Temporary storage ualloc */
	  un = hgcd_mul_matrix_vector (&M, u0, t0, u1, un, t0 + un);

	  ASSERT (un < ualloc);
	  ASSERT ( (u0[un-1] | u1[un-1]) > 0);
	}
      else
	{
	  /* mpn_hgcd has failed. Then either one of a or b is very
	     small, or the difference is very small. Perform one
	     subtraction followed by one division. */
	  ctx.u0 = u0;
	  ctx.u1 = u1;
	  ctx.tp = tp + n; /* ualloc */
	  ctx.un = un;

	  /* Temporary storage n */
	  n = mpn_gcd_subdiv_step (ap, bp, n, 0, mpn_gcdext_hook, &ctx, tp);
	  if (n == 0)
	    {
	      TMP_FREE;
	      return ctx.gn;
	    }

	  un = ctx.un;
	  ASSERT (un < ualloc);
	}
    }
  /* We have A = ... a + ... b
	     B =  u0 a +  u1 b

	     a = u1  A + ... B
	     b = -u0 A + ... B

     with bounds

       |u0|, |u1| <= B / min(a, b)

     We always have u1 > 0, and u0 == 0 is possible only if u1 == 1,
     in which case the only reduction done so far is a = A - k B for
     some k.

     Compute g = u a + v b = (u u1 - v u0) A + (...) B
     Here, u, v are bounded by

       |u| <= b,
       |v| <= a
  */

  ASSERT ( (ap[n-1] | bp[n-1]) > 0);

  if (UNLIKELY (mpn_cmp (ap, bp, n) == 0))
    {
      /* Must return the smallest cofactor, +u1 or -u0 */
      int c;

      MPN_COPY (gp, ap, n);

      MPN_CMP (c, u0, u1, un);
      /* c == 0 can happen only when A = (2k+1) G, B = 2 G. And in
	 this case we choose the cofactor + 1, corresponding to G = A
	 - k B, rather than -1, corresponding to G = - A + (k+1) B. */
      ASSERT (c != 0 || (un == 1 && u0[0] == 1 && u1[0] == 1));
      if (c < 0)
	{
	  MPN_NORMALIZE (u0, un);
	  MPN_COPY (up, u0, un);
	  *usizep = -un;
	}
      else
	{
	  MPN_NORMALIZE_NOT_ZERO (u1, un);
	  MPN_COPY (up, u1, un);
	  *usizep = un;
	}

      TMP_FREE;
      return n;
    }
  else if (UNLIKELY (u0[0] == 0) && un == 1)
    {
      mp_size_t gn;
      ASSERT (u1[0] == 1);

      /* g = u a + v b = (u u1 - v u0) A + (...) B = u A + (...) B */
      gn = mpn_gcdext_lehmer_n (gp, up, usizep, ap, bp, n, tp);

      TMP_FREE;
      return gn;
    }
  else
    {
      mp_size_t u0n;
      mp_size_t u1n;
      mp_size_t lehmer_un;
      mp_size_t lehmer_vn;
      mp_size_t gn;

      mp_ptr lehmer_up;
      mp_ptr lehmer_vp;
      int negate;

      lehmer_up = tp; tp += n;

      /* Call mpn_gcdext_lehmer_n with copies of a and b. */
      MPN_COPY (tp, ap, n);
      MPN_COPY (tp + n, bp, n);
      gn = mpn_gcdext_lehmer_n (gp, lehmer_up, &lehmer_un, tp, tp + n, n, tp + 2*n);

      u0n = un;
      MPN_NORMALIZE (u0, u0n);
      ASSERT (u0n > 0);

      if (lehmer_un == 0)
	{
	  /* u == 0  ==>  v = g / b == 1  ==> g = - u0 A + (...) B */
	  MPN_COPY (up, u0, u0n);
	  *usizep = -u0n;

	  TMP_FREE;
	  return gn;
	}

      lehmer_vp = tp;
      /* Compute v = (g - u a) / b */
      lehmer_vn = compute_v (lehmer_vp,
			     ap, bp, n, gp, gn, lehmer_up, lehmer_un, tp + n + 1);

      if (lehmer_un > 0)
	negate = 0;
      else
	{
	  lehmer_un = -lehmer_un;
	  negate = 1;
	}

      u1n = un;
      MPN_NORMALIZE (u1, u1n);
      ASSERT (u1n > 0);

      ASSERT (lehmer_un + u1n <= ualloc);
      ASSERT (lehmer_vn + u0n <= ualloc);

      /* We may still have v == 0 */

      /* Compute u u0 */
      if (lehmer_un <= u1n)
	/* Should be the common case */
	mpn_mul (up, u1, u1n, lehmer_up, lehmer_un);
      else
	mpn_mul (up, lehmer_up, lehmer_un, u1, u1n);

      un = u1n + lehmer_un;
      un -= (up[un - 1] == 0);

      if (lehmer_vn > 0)
	{
	  mp_limb_t cy;

	  /* Overwrites old u1 value */
	  if (lehmer_vn <= u0n)
	    /* Should be the common case */
	    mpn_mul (u1, u0, u0n, lehmer_vp, lehmer_vn);
	  else
	    mpn_mul (u1, lehmer_vp, lehmer_vn, u0, u0n);

	  u1n = u0n + lehmer_vn;
	  u1n -= (u1[u1n - 1] == 0);

	  if (u1n <= un)
	    {
	      cy = mpn_add (up, up, un, u1, u1n);
	    }
	  else
	    {
	      cy = mpn_add (up, u1, u1n, up, un);
	      un = u1n;
	    }
	  up[un] = cy;
	  un += (cy != 0);

	  ASSERT (un < ualloc);
	}
      *usizep = negate ? -un : un;

      TMP_FREE;
      return gn;
    }
}
