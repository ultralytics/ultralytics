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

/* Here, d is the index of the cofactor to update. FIXME: Could use qn
   = 0 for the common case q = 1. */
void
mpn_gcdext_hook (void *p, mp_srcptr gp, mp_size_t gn,
		 mp_srcptr qp, mp_size_t qn, int d)
{
  struct gcdext_ctx *ctx = (struct gcdext_ctx *) p;
  mp_size_t un = ctx->un;

  if (gp)
    {
      mp_srcptr up;

      ASSERT (gn > 0);
      ASSERT (gp[gn-1] > 0);

      MPN_COPY (ctx->gp, gp, gn);
      ctx->gn = gn;

      if (d < 0)
	{
	  int c;

	  /* Must return the smallest cofactor, +u1 or -u0 */
	  MPN_CMP (c, ctx->u0, ctx->u1, un);
	  ASSERT (c != 0 || (un == 1 && ctx->u0[0] == 1 && ctx->u1[0] == 1));

	  d = c < 0;
	}

      up = d ? ctx->u0 : ctx->u1;

      MPN_NORMALIZE (up, un);
      MPN_COPY (ctx->up, up, un);

      *ctx->usize = d ? -un : un;
    }
  else
    {
      mp_limb_t cy;
      mp_ptr u0 = ctx->u0;
      mp_ptr u1 = ctx->u1;

      ASSERT (d >= 0);

      if (d)
	MP_PTR_SWAP (u0, u1);

      qn -= (qp[qn-1] == 0);

      /* Update u0 += q  * u1 */
      if (qn == 1)
	{
	  mp_limb_t q = qp[0];

	  if (q == 1)
	    /* A common case. */
	    cy = mpn_add_n (u0, u0, u1, un);
	  else
	    cy = mpn_addmul_1 (u0, u1, un, q);
	}
      else
	{
	  mp_size_t u1n;
	  mp_ptr tp;

	  u1n = un;
	  MPN_NORMALIZE (u1, u1n);

	  if (u1n == 0)
	    return;

	  /* Should always have u1n == un here, and u1 >= u0. The
	     reason is that we alternate adding u0 to u1 and u1 to u0
	     (corresponding to subtractions a - b and b - a), and we
	     can get a large quotient only just after a switch, which
	     means that we'll add (a multiple of) the larger u to the
	     smaller. */

	  tp = ctx->tp;

	  if (qn > u1n)
	    mpn_mul (tp, qp, qn, u1, u1n);
	  else
	    mpn_mul (tp, u1, u1n, qp, qn);

	  u1n += qn;
	  u1n -= tp[u1n-1] == 0;

	  if (u1n >= un)
	    {
	      cy = mpn_add (u0, tp, u1n, u0, un);
	      un = u1n;
	    }
	  else
	    /* Note: Unlikely case, maybe never happens? */
	    cy = mpn_add (u0, u0, un, tp, u1n);

	}
      u0[un] = cy;
      ctx->un = un + (cy > 0);
    }
}

/* Temporary storage: 3*(n+1) for u. If hgcd2 succeeds, we need n for
   the matrix-vector multiplication adjusting a, b. If hgcd fails, we
   need at most n for the quotient and n+1 for the u update (reusing
   the extra u). In all, 4n + 3. */

mp_size_t
mpn_gcdext_lehmer_n (mp_ptr gp, mp_ptr up, mp_size_t *usize,
		     mp_ptr ap, mp_ptr bp, mp_size_t n,
		     mp_ptr tp)
{
  mp_size_t ualloc = n + 1;

  /* Keeps track of the second row of the reduction matrix
   *
   *   M = (v0, v1 ; u0, u1)
   *
   * which correspond to the first column of the inverse
   *
   *   M^{-1} = (u1, -v1; -u0, v0)
   *
   * This implies that
   *
   *   a =  u1 A (mod B)
   *   b = -u0 A (mod B)
   *
   * where A, B denotes the input values.
   */

  struct gcdext_ctx ctx;
  mp_size_t un;
  mp_ptr u0;
  mp_ptr u1;
  mp_ptr u2;

  MPN_ZERO (tp, 3*ualloc);
  u0 = tp; tp += ualloc;
  u1 = tp; tp += ualloc;
  u2 = tp; tp += ualloc;

  u1[0] = 1; un = 1;

  ctx.gp = gp;
  ctx.up = up;
  ctx.usize = usize;

  /* FIXME: Handle n == 2 differently, after the loop? */
  while (n >= 2)
    {
      struct hgcd_matrix1 M;
      mp_limb_t ah, al, bh, bl;
      mp_limb_t mask;

      mask = ap[n-1] | bp[n-1];
      ASSERT (mask > 0);

      if (mask & GMP_NUMB_HIGHBIT)
	{
	  ah = ap[n-1]; al = ap[n-2];
	  bh = bp[n-1]; bl = bp[n-2];
	}
      else if (n == 2)
	{
	  /* We use the full inputs without truncation, so we can
	     safely shift left. */
	  int shift;

	  count_leading_zeros (shift, mask);
	  ah = MPN_EXTRACT_NUMB (shift, ap[1], ap[0]);
	  al = ap[0] << shift;
	  bh = MPN_EXTRACT_NUMB (shift, bp[1], bp[0]);
	  bl = bp[0] << shift;
	}
      else
	{
	  int shift;

	  count_leading_zeros (shift, mask);
	  ah = MPN_EXTRACT_NUMB (shift, ap[n-1], ap[n-2]);
	  al = MPN_EXTRACT_NUMB (shift, ap[n-2], ap[n-3]);
	  bh = MPN_EXTRACT_NUMB (shift, bp[n-1], bp[n-2]);
	  bl = MPN_EXTRACT_NUMB (shift, bp[n-2], bp[n-3]);
	}

      /* Try an mpn_nhgcd2 step */
      if (mpn_hgcd2 (ah, al, bh, bl, &M))
	{
	  n = mpn_matrix22_mul1_inverse_vector (&M, tp, ap, bp, n);
	  MP_PTR_SWAP (ap, tp);
	  un = mpn_hgcd_mul_matrix1_vector(&M, u2, u0, u1, un);
	  MP_PTR_SWAP (u0, u2);
	}
      else
	{
	  /* mpn_hgcd2 has failed. Then either one of a or b is very
	     small, or the difference is very small. Perform one
	     subtraction followed by one division. */
	  ctx.u0 = u0;
	  ctx.u1 = u1;
	  ctx.tp = u2;
	  ctx.un = un;

	  /* Temporary storage n for the quotient and ualloc for the
	     new cofactor. */
	  n = mpn_gcd_subdiv_step (ap, bp, n, 0, mpn_gcdext_hook, &ctx, tp);
	  if (n == 0)
	    return ctx.gn;

	  un = ctx.un;
	}
    }
  ASSERT_ALWAYS (ap[0] > 0);
  ASSERT_ALWAYS (bp[0] > 0);

  if (ap[0] == bp[0])
    {
      int c;

      /* Which cofactor to return now? Candidates are +u1 and -u0,
	 depending on which of a and b was most recently reduced,
	 which we don't keep track of. So compare and get the smallest
	 one. */

      gp[0] = ap[0];

      MPN_CMP (c, u0, u1, un);
      ASSERT (c != 0 || (un == 1 && u0[0] == 1 && u1[0] == 1));
      if (c < 0)
	{
	  MPN_NORMALIZE (u0, un);
	  MPN_COPY (up, u0, un);
	  *usize = -un;
	}
      else
	{
	  MPN_NORMALIZE_NOT_ZERO (u1, un);
	  MPN_COPY (up, u1, un);
	  *usize = un;
	}
      return 1;
    }
  else
    {
      mp_limb_t uh, vh;
      mp_limb_signed_t u;
      mp_limb_signed_t v;
      int negate;

      gp[0] = mpn_gcdext_1 (&u, &v, ap[0], bp[0]);

      /* Set up = u u1 - v u0. Keep track of size, un grows by one or
	 two limbs. */

      if (u == 0)
	{
	  ASSERT (v == 1);
	  MPN_NORMALIZE (u0, un);
	  MPN_COPY (up, u0, un);
	  *usize = -un;
	  return 1;
	}
      else if (v == 0)
	{
	  ASSERT (u == 1);
	  MPN_NORMALIZE (u1, un);
	  MPN_COPY (up, u1, un);
	  *usize = un;
	  return 1;
	}
      else if (u > 0)
	{
	  negate = 0;
	  ASSERT (v < 0);
	  v = -v;
	}
      else
	{
	  negate = 1;
	  ASSERT (v > 0);
	  u = -u;
	}

      uh = mpn_mul_1 (up, u1, un, u);
      vh = mpn_addmul_1 (up, u0, un, v);

      if ( (uh | vh) > 0)
	{
	  uh += vh;
	  up[un++] = uh;
	  if (uh < vh)
	    up[un++] = 1;
	}

      MPN_NORMALIZE_NOT_ZERO (up, un);

      *usize = negate ? -un : un;
      return 1;
    }
}
