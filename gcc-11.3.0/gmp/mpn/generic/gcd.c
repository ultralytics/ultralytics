/* mpn/gcd.c: mpn_gcd for gcd of two odd integers.

Copyright 1991, 1993-1998, 2000-2005, 2008, 2010, 2012 Free Software
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

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"

/* Uses the HGCD operation described in

     N. Möller, On Schönhage's algorithm and subquadratic integer gcd
     computation, Math. Comp. 77 (2008), 589-607.

  to reduce inputs until they are of size below GCD_DC_THRESHOLD, and
  then uses Lehmer's algorithm.
*/

/* Some reasonable choices are n / 2 (same as in hgcd), and p = (n +
 * 2)/3, which gives a balanced multiplication in
 * mpn_hgcd_matrix_adjust. However, p = 2 n/3 gives slightly better
 * performance. The matrix-vector multiplication is then
 * 4:1-unbalanced, with matrix elements of size n/6, and vector
 * elements of size p = 2n/3. */

/* From analysis of the theoretical running time, it appears that when
 * multiplication takes time O(n^alpha), p should be chosen so that
 * the ratio of the time for the mpn_hgcd call, and the time for the
 * multiplication in mpn_hgcd_matrix_adjust, is roughly 1/(alpha -
 * 1). */
#ifdef TUNE_GCD_P
#define P_TABLE_SIZE 10000
mp_size_t p_table[P_TABLE_SIZE];
#define CHOOSE_P(n) ( (n) < P_TABLE_SIZE ? p_table[n] : 2*(n)/3)
#else
#define CHOOSE_P(n) (2*(n) / 3)
#endif

struct gcd_ctx
{
  mp_ptr gp;
  mp_size_t gn;
};

static void
gcd_hook (void *p, mp_srcptr gp, mp_size_t gn,
	  mp_srcptr qp, mp_size_t qn, int d)
{
  struct gcd_ctx *ctx = (struct gcd_ctx *) p;
  MPN_COPY (ctx->gp, gp, gn);
  ctx->gn = gn;
}

#if GMP_NAIL_BITS > 0
/* Nail supports should be easy, replacing the sub_ddmmss with nails
 * logic. */
#error Nails not supported.
#endif

/* Use binary algorithm to compute G <-- GCD (U, V) for usize, vsize == 2.
   Both U and V must be odd. */
static inline mp_size_t
gcd_2 (mp_ptr gp, mp_srcptr up, mp_srcptr vp)
{
  mp_limb_t u0, u1, v0, v1;
  mp_size_t gn;

  u0 = up[0];
  u1 = up[1];
  v0 = vp[0];
  v1 = vp[1];

  ASSERT (u0 & 1);
  ASSERT (v0 & 1);

  /* Check for u0 != v0 needed to ensure that argument to
   * count_trailing_zeros is non-zero. */
  while (u1 != v1 && u0 != v0)
    {
      unsigned long int r;
      if (u1 > v1)
	{
	  sub_ddmmss (u1, u0, u1, u0, v1, v0);
	  count_trailing_zeros (r, u0);
	  u0 = ((u1 << (GMP_NUMB_BITS - r)) & GMP_NUMB_MASK) | (u0 >> r);
	  u1 >>= r;
	}
      else  /* u1 < v1.  */
	{
	  sub_ddmmss (v1, v0, v1, v0, u1, u0);
	  count_trailing_zeros (r, v0);
	  v0 = ((v1 << (GMP_NUMB_BITS - r)) & GMP_NUMB_MASK) | (v0 >> r);
	  v1 >>= r;
	}
    }

  gp[0] = u0, gp[1] = u1, gn = 1 + (u1 != 0);

  /* If U == V == GCD, done.  Otherwise, compute GCD (V, |U - V|).  */
  if (u1 == v1 && u0 == v0)
    return gn;

  v0 = (u0 == v0) ? ((u1 > v1) ? u1-v1 : v1-u1) : ((u0 > v0) ? u0-v0 : v0-u0);
  gp[0] = mpn_gcd_1 (gp, gn, v0);

  return 1;
}

mp_size_t
mpn_gcd (mp_ptr gp, mp_ptr up, mp_size_t usize, mp_ptr vp, mp_size_t n)
{
  mp_size_t talloc;
  mp_size_t scratch;
  mp_size_t matrix_scratch;

  struct gcd_ctx ctx;
  mp_ptr tp;
  TMP_DECL;

  ASSERT (usize >= n);
  ASSERT (n > 0);
  ASSERT (vp[n-1] > 0);

  /* FIXME: Check for small sizes first, before setting up temporary
     storage etc. */
  talloc = MPN_GCD_SUBDIV_STEP_ITCH(n);

  /* For initial division */
  scratch = usize - n + 1;
  if (scratch > talloc)
    talloc = scratch;

#if TUNE_GCD_P
  if (CHOOSE_P (n) > 0)
#else
  if (ABOVE_THRESHOLD (n, GCD_DC_THRESHOLD))
#endif
    {
      mp_size_t hgcd_scratch;
      mp_size_t update_scratch;
      mp_size_t p = CHOOSE_P (n);
      mp_size_t scratch;
#if TUNE_GCD_P
      /* Worst case, since we don't guarantee that n - CHOOSE_P(n)
	 is increasing */
      matrix_scratch = MPN_HGCD_MATRIX_INIT_ITCH (n);
      hgcd_scratch = mpn_hgcd_itch (n);
      update_scratch = 2*(n - 1);
#else
      matrix_scratch = MPN_HGCD_MATRIX_INIT_ITCH (n - p);
      hgcd_scratch = mpn_hgcd_itch (n - p);
      update_scratch = p + n - 1;
#endif
      scratch = matrix_scratch + MAX(hgcd_scratch, update_scratch);
      if (scratch > talloc)
	talloc = scratch;
    }

  TMP_MARK;
  tp = TMP_ALLOC_LIMBS(talloc);

  if (usize > n)
    {
      mpn_tdiv_qr (tp, up, 0, up, usize, vp, n);

      if (mpn_zero_p (up, n))
	{
	  MPN_COPY (gp, vp, n);
	  ctx.gn = n;
	  goto done;
	}
    }

  ctx.gp = gp;

#if TUNE_GCD_P
  while (CHOOSE_P (n) > 0)
#else
  while (ABOVE_THRESHOLD (n, GCD_DC_THRESHOLD))
#endif
    {
      struct hgcd_matrix M;
      mp_size_t p = CHOOSE_P (n);
      mp_size_t matrix_scratch = MPN_HGCD_MATRIX_INIT_ITCH (n - p);
      mp_size_t nn;
      mpn_hgcd_matrix_init (&M, n - p, tp);
      nn = mpn_hgcd (up + p, vp + p, n - p, &M, tp + matrix_scratch);
      if (nn > 0)
	{
	  ASSERT (M.n <= (n - p - 1)/2);
	  ASSERT (M.n + p <= (p + n - 1) / 2);
	  /* Temporary storage 2 (p + M->n) <= p + n - 1. */
	  n = mpn_hgcd_matrix_adjust (&M, p + nn, up, vp, p, tp + matrix_scratch);
	}
      else
	{
	  /* Temporary storage n */
	  n = mpn_gcd_subdiv_step (up, vp, n, 0, gcd_hook, &ctx, tp);
	  if (n == 0)
	    goto done;
	}
    }

  while (n > 2)
    {
      struct hgcd_matrix1 M;
      mp_limb_t uh, ul, vh, vl;
      mp_limb_t mask;

      mask = up[n-1] | vp[n-1];
      ASSERT (mask > 0);

      if (mask & GMP_NUMB_HIGHBIT)
	{
	  uh = up[n-1]; ul = up[n-2];
	  vh = vp[n-1]; vl = vp[n-2];
	}
      else
	{
	  int shift;

	  count_leading_zeros (shift, mask);
	  uh = MPN_EXTRACT_NUMB (shift, up[n-1], up[n-2]);
	  ul = MPN_EXTRACT_NUMB (shift, up[n-2], up[n-3]);
	  vh = MPN_EXTRACT_NUMB (shift, vp[n-1], vp[n-2]);
	  vl = MPN_EXTRACT_NUMB (shift, vp[n-2], vp[n-3]);
	}

      /* Try an mpn_hgcd2 step */
      if (mpn_hgcd2 (uh, ul, vh, vl, &M))
	{
	  n = mpn_matrix22_mul1_inverse_vector (&M, tp, up, vp, n);
	  MP_PTR_SWAP (up, tp);
	}
      else
	{
	  /* mpn_hgcd2 has failed. Then either one of a or b is very
	     small, or the difference is very small. Perform one
	     subtraction followed by one division. */

	  /* Temporary storage n */
	  n = mpn_gcd_subdiv_step (up, vp, n, 0, &gcd_hook, &ctx, tp);
	  if (n == 0)
	    goto done;
	}
    }

  ASSERT(up[n-1] | vp[n-1]);

  if (n == 1)
    {
      *gp = mpn_gcd_1(up, 1, vp[0]);
      ctx.gn = 1;
      goto done;
    }

  /* Due to the calling convention for mpn_gcd, at most one can be
     even. */

  if (! (up[0] & 1))
    MP_PTR_SWAP (up, vp);

  ASSERT (up[0] & 1);

  if (vp[0] == 0)
    {
      *gp = mpn_gcd_1 (up, 2, vp[1]);
      ctx.gn = 1;
      goto done;
    }
  else if (! (vp[0] & 1))
    {
      int r;
      count_trailing_zeros (r, vp[0]);
      vp[0] = ((vp[1] << (GMP_NUMB_BITS - r)) & GMP_NUMB_MASK) | (vp[0] >> r);
      vp[1] >>= r;
    }

  ctx.gn = gcd_2(gp, up, vp);

done:
  TMP_FREE;
  return ctx.gn;
}
