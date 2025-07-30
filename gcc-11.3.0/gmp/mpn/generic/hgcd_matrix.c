/* hgcd_matrix.c.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2003-2005, 2008, 2012 Free Software Foundation, Inc.

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

/* For input of size n, matrix elements are of size at most ceil(n/2)
   - 1, but we need two limbs extra. */
void
mpn_hgcd_matrix_init (struct hgcd_matrix *M, mp_size_t n, mp_ptr p)
{
  mp_size_t s = (n+1)/2 + 1;
  M->alloc = s;
  M->n = 1;
  MPN_ZERO (p, 4 * s);
  M->p[0][0] = p;
  M->p[0][1] = p + s;
  M->p[1][0] = p + 2 * s;
  M->p[1][1] = p + 3 * s;

  M->p[0][0][0] = M->p[1][1][0] = 1;
}

/* Update column COL, adding in Q * column (1-COL). Temporary storage:
 * qn + n <= M->alloc, where n is the size of the largest element in
 * column 1 - COL. */
void
mpn_hgcd_matrix_update_q (struct hgcd_matrix *M, mp_srcptr qp, mp_size_t qn,
			  unsigned col, mp_ptr tp)
{
  ASSERT (col < 2);

  if (qn == 1)
    {
      mp_limb_t q = qp[0];
      mp_limb_t c0, c1;

      c0 = mpn_addmul_1 (M->p[0][col], M->p[0][1-col], M->n, q);
      c1 = mpn_addmul_1 (M->p[1][col], M->p[1][1-col], M->n, q);

      M->p[0][col][M->n] = c0;
      M->p[1][col][M->n] = c1;

      M->n += (c0 | c1) != 0;
    }
  else
    {
      unsigned row;

      /* Carries for the unlikely case that we get both high words
	 from the multiplication and carries from the addition. */
      mp_limb_t c[2];
      mp_size_t n;

      /* The matrix will not necessarily grow in size by qn, so we
	 need normalization in order not to overflow M. */

      for (n = M->n; n + qn > M->n; n--)
	{
	  ASSERT (n > 0);
	  if (M->p[0][1-col][n-1] > 0 || M->p[1][1-col][n-1] > 0)
	    break;
	}

      ASSERT (qn + n <= M->alloc);

      for (row = 0; row < 2; row++)
	{
	  if (qn <= n)
	    mpn_mul (tp, M->p[row][1-col], n, qp, qn);
	  else
	    mpn_mul (tp, qp, qn, M->p[row][1-col], n);

	  ASSERT (n + qn >= M->n);
	  c[row] = mpn_add (M->p[row][col], tp, n + qn, M->p[row][col], M->n);
	}

      n += qn;

      if (c[0] | c[1])
	{
	  M->p[0][col][n] = c[0];
	  M->p[1][col][n] = c[1];
	  n++;
	}
      else
	{
	  n -= (M->p[0][col][n-1] | M->p[1][col][n-1]) == 0;
	  ASSERT (n >= M->n);
	}
      M->n = n;
    }

  ASSERT (M->n < M->alloc);
}

/* Multiply M by M1 from the right. Since the M1 elements fit in
   GMP_NUMB_BITS - 1 bits, M grows by at most one limb. Needs
   temporary space M->n */
void
mpn_hgcd_matrix_mul_1 (struct hgcd_matrix *M, const struct hgcd_matrix1 *M1,
		       mp_ptr tp)
{
  mp_size_t n0, n1;

  /* Could avoid copy by some swapping of pointers. */
  MPN_COPY (tp, M->p[0][0], M->n);
  n0 = mpn_hgcd_mul_matrix1_vector (M1, M->p[0][0], tp, M->p[0][1], M->n);
  MPN_COPY (tp, M->p[1][0], M->n);
  n1 = mpn_hgcd_mul_matrix1_vector (M1, M->p[1][0], tp, M->p[1][1], M->n);

  /* Depends on zero initialization */
  M->n = MAX(n0, n1);
  ASSERT (M->n < M->alloc);
}

/* Multiply M by M1 from the right. Needs 3*(M->n + M1->n) + 5 limbs
   of temporary storage (see mpn_matrix22_mul_itch). */
void
mpn_hgcd_matrix_mul (struct hgcd_matrix *M, const struct hgcd_matrix *M1,
		     mp_ptr tp)
{
  mp_size_t n;

  /* About the new size of M:s elements. Since M1's diagonal elements
     are > 0, no element can decrease. The new elements are of size
     M->n + M1->n, one limb more or less. The computation of the
     matrix product produces elements of size M->n + M1->n + 1. But
     the true size, after normalization, may be three limbs smaller.

     The reason that the product has normalized size >= M->n + M1->n -
     2 is subtle. It depends on the fact that M and M1 can be factored
     as products of (1,1; 0,1) and (1,0; 1,1), and that we can't have
     M ending with a large power and M1 starting with a large power of
     the same matrix. */

  /* FIXME: Strassen multiplication gives only a small speedup. In FFT
     multiplication range, this function could be sped up quite a lot
     using invariance. */
  ASSERT (M->n + M1->n < M->alloc);

  ASSERT ((M->p[0][0][M->n-1] | M->p[0][1][M->n-1]
	   | M->p[1][0][M->n-1] | M->p[1][1][M->n-1]) > 0);

  ASSERT ((M1->p[0][0][M1->n-1] | M1->p[0][1][M1->n-1]
	   | M1->p[1][0][M1->n-1] | M1->p[1][1][M1->n-1]) > 0);

  mpn_matrix22_mul (M->p[0][0], M->p[0][1],
		    M->p[1][0], M->p[1][1], M->n,
		    M1->p[0][0], M1->p[0][1],
		    M1->p[1][0], M1->p[1][1], M1->n, tp);

  /* Index of last potentially non-zero limb, size is one greater. */
  n = M->n + M1->n;

  n -= ((M->p[0][0][n] | M->p[0][1][n] | M->p[1][0][n] | M->p[1][1][n]) == 0);
  n -= ((M->p[0][0][n] | M->p[0][1][n] | M->p[1][0][n] | M->p[1][1][n]) == 0);
  n -= ((M->p[0][0][n] | M->p[0][1][n] | M->p[1][0][n] | M->p[1][1][n]) == 0);

  ASSERT ((M->p[0][0][n] | M->p[0][1][n] | M->p[1][0][n] | M->p[1][1][n]) > 0);

  M->n = n + 1;
}

/* Multiplies the least significant p limbs of (a;b) by M^-1.
   Temporary space needed: 2 * (p + M->n)*/
mp_size_t
mpn_hgcd_matrix_adjust (const struct hgcd_matrix *M,
			mp_size_t n, mp_ptr ap, mp_ptr bp,
			mp_size_t p, mp_ptr tp)
{
  /* M^-1 (a;b) = (r11, -r01; -r10, r00) (a ; b)
     = (r11 a - r01 b; - r10 a + r00 b */

  mp_ptr t0 = tp;
  mp_ptr t1 = tp + p + M->n;
  mp_limb_t ah, bh;
  mp_limb_t cy;

  ASSERT (p + M->n  < n);

  /* First compute the two values depending on a, before overwriting a */

  if (M->n >= p)
    {
      mpn_mul (t0, M->p[1][1], M->n, ap, p);
      mpn_mul (t1, M->p[1][0], M->n, ap, p);
    }
  else
    {
      mpn_mul (t0, ap, p, M->p[1][1], M->n);
      mpn_mul (t1, ap, p, M->p[1][0], M->n);
    }

  /* Update a */
  MPN_COPY (ap, t0, p);
  ah = mpn_add (ap + p, ap + p, n - p, t0 + p, M->n);

  if (M->n >= p)
    mpn_mul (t0, M->p[0][1], M->n, bp, p);
  else
    mpn_mul (t0, bp, p, M->p[0][1], M->n);

  cy = mpn_sub (ap, ap, n, t0, p + M->n);
  ASSERT (cy <= ah);
  ah -= cy;

  /* Update b */
  if (M->n >= p)
    mpn_mul (t0, M->p[0][0], M->n, bp, p);
  else
    mpn_mul (t0, bp, p, M->p[0][0], M->n);

  MPN_COPY (bp, t0, p);
  bh = mpn_add (bp + p, bp + p, n - p, t0 + p, M->n);
  cy = mpn_sub (bp, bp, n, t1, p + M->n);
  ASSERT (cy <= bh);
  bh -= cy;

  if (ah > 0 || bh > 0)
    {
      ap[n] = ah;
      bp[n] = bh;
      n++;
    }
  else
    {
      /* The subtraction can reduce the size by at most one limb. */
      if (ap[n-1] == 0 && bp[n-1] == 0)
	n--;
    }
  ASSERT (ap[n-1] > 0 || bp[n-1] > 0);
  return n;
}
