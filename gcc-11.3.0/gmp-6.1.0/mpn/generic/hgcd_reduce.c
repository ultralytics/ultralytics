/* hgcd_reduce.c.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2011, 2012 Free Software Foundation, Inc.

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

/* Computes R -= A * B. Result must be non-negative. Normalized down
   to size an, and resulting size is returned. */
static mp_size_t
submul (mp_ptr rp, mp_size_t rn,
	mp_srcptr ap, mp_size_t an, mp_srcptr bp, mp_size_t bn)
{
  mp_ptr tp;
  TMP_DECL;

  ASSERT (bn > 0);
  ASSERT (an >= bn);
  ASSERT (rn >= an);
  ASSERT (an + bn <= rn + 1);

  TMP_MARK;
  tp = TMP_ALLOC_LIMBS (an + bn);

  mpn_mul (tp, ap, an, bp, bn);
  ASSERT ((an + bn <= rn) || (tp[rn] == 0));
  ASSERT_NOCARRY (mpn_sub (rp, rp, rn, tp, an + bn - (an + bn > rn)));
  TMP_FREE;

  while (rn > an && (rp[rn-1] == 0))
    rn--;

  return rn;
}

/* Computes (a, b)  <--  M^{-1} (a; b) */
/* FIXME:
    x Take scratch parameter, and figure out scratch need.

    x Use some fallback for small M->n?
*/
static mp_size_t
hgcd_matrix_apply (const struct hgcd_matrix *M,
		   mp_ptr ap, mp_ptr bp,
		   mp_size_t n)
{
  mp_size_t an, bn, un, vn, nn;
  mp_size_t mn[2][2];
  mp_size_t modn;
  mp_ptr tp, sp, scratch;
  mp_limb_t cy;
  unsigned i, j;

  TMP_DECL;

  ASSERT ( (ap[n-1] | bp[n-1]) > 0);

  an = n;
  MPN_NORMALIZE (ap, an);
  bn = n;
  MPN_NORMALIZE (bp, bn);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      {
	mp_size_t k;
	k = M->n;
	MPN_NORMALIZE (M->p[i][j], k);
	mn[i][j] = k;
      }

  ASSERT (mn[0][0] > 0);
  ASSERT (mn[1][1] > 0);
  ASSERT ( (mn[0][1] | mn[1][0]) > 0);

  TMP_MARK;

  if (mn[0][1] == 0)
    {
      /* A unchanged, M = (1, 0; q, 1) */
      ASSERT (mn[0][0] == 1);
      ASSERT (M->p[0][0][0] == 1);
      ASSERT (mn[1][1] == 1);
      ASSERT (M->p[1][1][0] == 1);

      /* Put B <-- B - q A */
      nn = submul (bp, bn, ap, an, M->p[1][0], mn[1][0]);
    }
  else if (mn[1][0] == 0)
    {
      /* B unchanged, M = (1, q; 0, 1) */
      ASSERT (mn[0][0] == 1);
      ASSERT (M->p[0][0][0] == 1);
      ASSERT (mn[1][1] == 1);
      ASSERT (M->p[1][1][0] == 1);

      /* Put A  <-- A - q * B */
      nn = submul (ap, an, bp, bn, M->p[0][1], mn[0][1]);
    }
  else
    {
      /* A = m00 a + m01 b  ==> a <= A / m00, b <= A / m01.
	 B = m10 a + m11 b  ==> a <= B / m10, b <= B / m11. */
      un = MIN (an - mn[0][0], bn - mn[1][0]) + 1;
      vn = MIN (an - mn[0][1], bn - mn[1][1]) + 1;

      nn = MAX (un, vn);
      /* In the range of interest, mulmod_bnm1 should always beat mullo. */
      modn = mpn_mulmod_bnm1_next_size (nn + 1);

      TMP_ALLOC_LIMBS_3 (tp, modn,
			 sp, modn,
			 scratch, mpn_mulmod_bnm1_itch (modn, modn, M->n));

      ASSERT (n <= 2*modn);

      if (n > modn)
	{
	  cy = mpn_add (ap, ap, modn, ap + modn, n - modn);
	  MPN_INCR_U (ap, modn, cy);

	  cy = mpn_add (bp, bp, modn, bp + modn, n - modn);
	  MPN_INCR_U (bp, modn, cy);

	  n = modn;
	}

      mpn_mulmod_bnm1 (tp, modn, ap, n, M->p[1][1], mn[1][1], scratch);
      mpn_mulmod_bnm1 (sp, modn, bp, n, M->p[0][1], mn[0][1], scratch);

      /* FIXME: Handle the small n case in some better way. */
      if (n + mn[1][1] < modn)
	MPN_ZERO (tp + n + mn[1][1], modn - n - mn[1][1]);
      if (n + mn[0][1] < modn)
	MPN_ZERO (sp + n + mn[0][1], modn - n - mn[0][1]);

      cy = mpn_sub_n (tp, tp, sp, modn);
      MPN_DECR_U (tp, modn, cy);

      ASSERT (mpn_zero_p (tp + nn, modn - nn));

      mpn_mulmod_bnm1 (sp, modn, ap, n, M->p[1][0], mn[1][0], scratch);
      MPN_COPY (ap, tp, nn);
      mpn_mulmod_bnm1 (tp, modn, bp, n, M->p[0][0], mn[0][0], scratch);

      if (n + mn[1][0] < modn)
	MPN_ZERO (sp + n + mn[1][0], modn - n - mn[1][0]);
      if (n + mn[0][0] < modn)
	MPN_ZERO (tp + n + mn[0][0], modn - n - mn[0][0]);

      cy = mpn_sub_n (tp, tp, sp, modn);
      MPN_DECR_U (tp, modn, cy);

      ASSERT (mpn_zero_p (tp + nn, modn - nn));
      MPN_COPY (bp, tp, nn);

      while ( (ap[nn-1] | bp[nn-1]) == 0)
	{
	  nn--;
	  ASSERT (nn > 0);
	}
    }
  TMP_FREE;

  return nn;
}

mp_size_t
mpn_hgcd_reduce_itch (mp_size_t n, mp_size_t p)
{
  mp_size_t itch;
  if (BELOW_THRESHOLD (n, HGCD_REDUCE_THRESHOLD))
    {
      itch = mpn_hgcd_itch (n-p);

      /* For arbitrary p, the storage for _adjust is 2*(p + M->n) = 2 *
	 (p + ceil((n-p)/2) - 1 <= n + p - 1 */
      if (itch < n + p - 1)
	itch = n + p - 1;
    }
  else
    {
      itch = 2*(n-p) + mpn_hgcd_itch (n-p);
      /* Currently, hgcd_matrix_apply allocates its own storage. */
    }
  return itch;
}

/* FIXME: Document storage need. */
mp_size_t
mpn_hgcd_reduce (struct hgcd_matrix *M,
		 mp_ptr ap, mp_ptr bp, mp_size_t n, mp_size_t p,
		 mp_ptr tp)
{
  mp_size_t nn;
  if (BELOW_THRESHOLD (n, HGCD_REDUCE_THRESHOLD))
    {
      nn = mpn_hgcd (ap + p, bp + p, n - p, M, tp);
      if (nn > 0)
	/* Needs 2*(p + M->n) <= 2*(floor(n/2) + ceil(n/2) - 1)
	   = 2 (n - 1) */
	return mpn_hgcd_matrix_adjust (M, p + nn, ap, bp, p, tp);
    }
  else
    {
      MPN_COPY (tp, ap + p, n - p);
      MPN_COPY (tp + n - p, bp + p, n - p);
      if (mpn_hgcd_appr (tp, tp + n - p, n - p, M, tp + 2*(n-p)))
	return hgcd_matrix_apply (M, ap, bp, n);
    }
  return 0;
}
