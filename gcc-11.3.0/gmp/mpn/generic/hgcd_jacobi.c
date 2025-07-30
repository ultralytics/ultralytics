/* hgcd_jacobi.c.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2003-2005, 2008, 2011, 2012 Free Software Foundation, Inc.

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

/* This file is almost a copy of hgcd.c, with some added calls to
   mpn_jacobi_update */

struct hgcd_jacobi_ctx
{
  struct hgcd_matrix *M;
  unsigned *bitsp;
};

static void
hgcd_jacobi_hook (void *p, mp_srcptr gp, mp_size_t gn,
		  mp_srcptr qp, mp_size_t qn, int d)
{
  ASSERT (!gp);
  ASSERT (d >= 0);

  MPN_NORMALIZE (qp, qn);
  if (qn > 0)
    {
      struct hgcd_jacobi_ctx *ctx = (struct hgcd_jacobi_ctx *) p;
      /* NOTES: This is a bit ugly. A tp area is passed to
	 gcd_subdiv_step, which stores q at the start of that area. We
	 now use the rest. */
      mp_ptr tp = (mp_ptr) qp + qn;

      mpn_hgcd_matrix_update_q (ctx->M, qp, qn, d, tp);
      *ctx->bitsp = mpn_jacobi_update (*ctx->bitsp, d, qp[0] & 3);
    }
}

/* Perform a few steps, using some of mpn_hgcd2, subtraction and
   division. Reduces the size by almost one limb or more, but never
   below the given size s. Return new size for a and b, or 0 if no
   more steps are possible.

   If hgcd2 succeeds, needs temporary space for hgcd_matrix_mul_1, M->n
   limbs, and hgcd_mul_matrix1_inverse_vector, n limbs. If hgcd2
   fails, needs space for the quotient, qn <= n - s + 1 limbs, for and
   hgcd_matrix_update_q, qn + (size of the appropriate column of M) <=
   resulting size of M.

   If N is the input size to the calling hgcd, then s = floor(N/2) +
   1, M->n < N, qn + matrix size <= n - s + 1 + n - s = 2 (n - s) + 1
   < N, so N is sufficient.
*/

static mp_size_t
hgcd_jacobi_step (mp_size_t n, mp_ptr ap, mp_ptr bp, mp_size_t s,
		  struct hgcd_matrix *M, unsigned *bitsp, mp_ptr tp)
{
  struct hgcd_matrix1 M1;
  mp_limb_t mask;
  mp_limb_t ah, al, bh, bl;

  ASSERT (n > s);

  mask = ap[n-1] | bp[n-1];
  ASSERT (mask > 0);

  if (n == s + 1)
    {
      if (mask < 4)
	goto subtract;

      ah = ap[n-1]; al = ap[n-2];
      bh = bp[n-1]; bl = bp[n-2];
    }
  else if (mask & GMP_NUMB_HIGHBIT)
    {
      ah = ap[n-1]; al = ap[n-2];
      bh = bp[n-1]; bl = bp[n-2];
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

  /* Try an mpn_hgcd2 step */
  if (mpn_hgcd2_jacobi (ah, al, bh, bl, &M1, bitsp))
    {
      /* Multiply M <- M * M1 */
      mpn_hgcd_matrix_mul_1 (M, &M1, tp);

      /* Can't swap inputs, so we need to copy. */
      MPN_COPY (tp, ap, n);
      /* Multiply M1^{-1} (a;b) */
      return mpn_matrix22_mul1_inverse_vector (&M1, ap, tp, bp, n);
    }

 subtract:
  {
    struct hgcd_jacobi_ctx ctx;
    ctx.M = M;
    ctx.bitsp = bitsp;

    return mpn_gcd_subdiv_step (ap, bp, n, s, hgcd_jacobi_hook, &ctx, tp);
  }
}

/* Reduces a,b until |a-b| fits in n/2 + 1 limbs. Constructs matrix M
   with elements of size at most (n+1)/2 - 1. Returns new size of a,
   b, or zero if no reduction is possible. */

/* Same scratch requirements as for mpn_hgcd. */
mp_size_t
mpn_hgcd_jacobi (mp_ptr ap, mp_ptr bp, mp_size_t n,
		 struct hgcd_matrix *M, unsigned *bitsp, mp_ptr tp)
{
  mp_size_t s = n/2 + 1;

  mp_size_t nn;
  int success = 0;

  if (n <= s)
    /* Happens when n <= 2, a fairly uninteresting case but exercised
       by the random inputs of the testsuite. */
    return 0;

  ASSERT ((ap[n-1] | bp[n-1]) > 0);

  ASSERT ((n+1)/2 - 1 < M->alloc);

  if (ABOVE_THRESHOLD (n, HGCD_THRESHOLD))
    {
      mp_size_t n2 = (3*n)/4 + 1;
      mp_size_t p = n/2;

      nn = mpn_hgcd_jacobi (ap + p, bp + p, n - p, M, bitsp, tp);
      if (nn > 0)
	{
	  /* Needs 2*(p + M->n) <= 2*(floor(n/2) + ceil(n/2) - 1)
	     = 2 (n - 1) */
	  n = mpn_hgcd_matrix_adjust (M, p + nn, ap, bp, p, tp);
	  success = 1;
	}
      while (n > n2)
	{
	  /* Needs n + 1 storage */
	  nn = hgcd_jacobi_step (n, ap, bp, s, M, bitsp, tp);
	  if (!nn)
	    return success ? n : 0;
	  n = nn;
	  success = 1;
	}

      if (n > s + 2)
	{
	  struct hgcd_matrix M1;
	  mp_size_t scratch;

	  p = 2*s - n + 1;
	  scratch = MPN_HGCD_MATRIX_INIT_ITCH (n-p);

	  mpn_hgcd_matrix_init(&M1, n - p, tp);
	  nn = mpn_hgcd_jacobi (ap + p, bp + p, n - p, &M1, bitsp, tp + scratch);
	  if (nn > 0)
	    {
	      /* We always have max(M) > 2^{-(GMP_NUMB_BITS + 1)} max(M1) */
	      ASSERT (M->n + 2 >= M1.n);

	      /* Furthermore, assume M ends with a quotient (1, q; 0, 1),
		 then either q or q + 1 is a correct quotient, and M1 will
		 start with either (1, 0; 1, 1) or (2, 1; 1, 1). This
		 rules out the case that the size of M * M1 is much
		 smaller than the expected M->n + M1->n. */

	      ASSERT (M->n + M1.n < M->alloc);

	      /* Needs 2 (p + M->n) <= 2 (2*s - n2 + 1 + n2 - s - 1)
		 = 2*s <= 2*(floor(n/2) + 1) <= n + 2. */
	      n = mpn_hgcd_matrix_adjust (&M1, p + nn, ap, bp, p, tp + scratch);

	      /* We need a bound for of M->n + M1.n. Let n be the original
		 input size. Then

		 ceil(n/2) - 1 >= size of product >= M.n + M1.n - 2

		 and it follows that

		 M.n + M1.n <= ceil(n/2) + 1

		 Then 3*(M.n + M1.n) + 5 <= 3 * ceil(n/2) + 8 is the
		 amount of needed scratch space. */
	      mpn_hgcd_matrix_mul (M, &M1, tp + scratch);
	      success = 1;
	    }
	}
    }

  for (;;)
    {
      /* Needs s+3 < n */
      nn = hgcd_jacobi_step (n, ap, bp, s, M, bitsp, tp);
      if (!nn)
	return success ? n : 0;

      n = nn;
      success = 1;
    }
}
