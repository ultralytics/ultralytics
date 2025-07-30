/* jacobi.c

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 1996, 1998, 2000-2004, 2008, 2010, 2011 Free Software Foundation,
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

#ifndef JACOBI_DC_THRESHOLD
#define JACOBI_DC_THRESHOLD GCD_DC_THRESHOLD
#endif

/* Schönhage's rules:
 *
 * Assume r0 = r1 q1 + r2, with r0 odd, and r1 = q2 r2 + r3
 *
 * If r1 is odd, then
 *
 *   (r1 | r0) = s(r1, r0) (r0 | r1) = s(r1, r0) (r2, r1)
 *
 * where s(x,y) = (-1)^{(x-1)(y-1)/4} = (-1)^[x = y = 3 (mod 4)].
 *
 * If r1 is even, r2 must be odd. We have
 *
 *   (r1 | r0) = (r1 - r0 | r0) = (-1)^(r0-1)/2 (r0 - r1 | r0)
 *             = (-1)^(r0-1)/2 s(r0, r0 - r1) (r0 | r0 - r1)
 *             = (-1)^(r0-1)/2 s(r0, r0 - r1) (r1 | r0 - r1)
 *
 * Now, if r1 = 0 (mod 4), then the sign factor is +1, and repeating
 * q1 times gives
 *
 *   (r1 | r0) = (r1 | r2) = (r3 | r2)
 *
 * On the other hand, if r1 = 2 (mod 4), the sign factor is
 * (-1)^{(r0-1)/2}, and repeating q1 times gives the exponent
 *
 *   (r0-1)/2 + (r0-r1-1)/2 + ... + (r0 - (q1-1) r1)/2
 *   = q1 (r0-1)/2 + q1 (q1-1)/2
 *
 * and we can summarize the even case as
 *
 *   (r1 | r0) = t(r1, r0, q1) (r3 | r2)
 *
 * where t(x,y,q) = (-1)^{[x = 2 (mod 4)] (q(y-1)/2 + y(q-1)/2)}
 *
 * What about termination? The remainder sequence ends with (0|1) = 1
 * (or (0 | r) = 0 if r != 1). What are the possible cases? If r1 is
 * odd, r2 may be zero. If r1 is even, then r2 = r0 - q1 r1 is odd and
 * hence non-zero. We may have r3 = r1 - q2 r2 = 0.
 *
 * Examples: (11|15) = - (15|11) = - (4|11)
 *            (4|11) =    (4| 3) =   (1| 3)
 *            (1| 3) = (3|1) = (0|1) = 1
 *
 *             (2|7) = (2|1) = (0|1) = 1
 *
 * Detail:     (2|7) = (2-7|7) = (-1|7)(5|7) = -(7|5) = -(2|5)
 *             (2|5) = (2-5|5) = (-1|5)(3|5) =  (5|3) =  (2|3)
 *             (2|3) = (2-3|3) = (-1|3)(1|3) = -(3|1) = -(2|1)
 *
 */

/* In principle, the state consists of four variables: e (one bit), a,
   b (two bits each), d (one bit). Collected factors are (-1)^e. a and
   b are the least significant bits of the current remainders. d
   (denominator) is 0 if we're currently subtracting multiplies of a
   from b, and 1 if we're subtracting b from a.

   e is stored in the least significant bit, while a, b and d are
   coded as only 13 distinct values in bits 1-4, according to the
   following table. For rows not mentioning d, the value is either
   implied, or it doesn't matter. */

#if WANT_ASSERT
static const struct
{
  unsigned char a;
  unsigned char b;
} decode_table[13] = {
  /*  0 */ { 0, 1 },
  /*  1 */ { 0, 3 },
  /*  2 */ { 1, 1 },
  /*  3 */ { 1, 3 },
  /*  4 */ { 2, 1 },
  /*  5 */ { 2, 3 },
  /*  6 */ { 3, 1 },
  /*  7 */ { 3, 3 }, /* d = 1 */
  /*  8 */ { 1, 0 },
  /*  9 */ { 1, 2 },
  /* 10 */ { 3, 0 },
  /* 11 */ { 3, 2 },
  /* 12 */ { 3, 3 }, /* d = 0 */
};
#define JACOBI_A(bits) (decode_table[(bits)>>1].a)
#define JACOBI_B(bits) (decode_table[(bits)>>1].b)
#endif /* WANT_ASSERT */

const unsigned char jacobi_table[208] = {
#include "jacobitab.h"
};

#define BITS_FAIL 31

static void
jacobi_hook (void *p, mp_srcptr gp, mp_size_t gn,
	     mp_srcptr qp, mp_size_t qn, int d)
{
  unsigned *bitsp = (unsigned *) p;

  if (gp)
    {
      ASSERT (gn > 0);
      if (gn != 1 || gp[0] != 1)
	{
	  *bitsp = BITS_FAIL;
	  return;
	}
    }

  if (qp)
    {
      ASSERT (qn > 0);
      ASSERT (d >= 0);
      *bitsp = mpn_jacobi_update (*bitsp, d, qp[0] & 3);
    }
}

#define CHOOSE_P(n) (2*(n) / 3)

int
mpn_jacobi_n (mp_ptr ap, mp_ptr bp, mp_size_t n, unsigned bits)
{
  mp_size_t scratch;
  mp_size_t matrix_scratch;
  mp_ptr tp;

  TMP_DECL;

  ASSERT (n > 0);
  ASSERT ( (ap[n-1] | bp[n-1]) > 0);
  ASSERT ( (bp[0] | ap[0]) & 1);

  /* FIXME: Check for small sizes first, before setting up temporary
     storage etc. */
  scratch = MPN_GCD_SUBDIV_STEP_ITCH(n);

  if (ABOVE_THRESHOLD (n, GCD_DC_THRESHOLD))
    {
      mp_size_t hgcd_scratch;
      mp_size_t update_scratch;
      mp_size_t p = CHOOSE_P (n);
      mp_size_t dc_scratch;

      matrix_scratch = MPN_HGCD_MATRIX_INIT_ITCH (n - p);
      hgcd_scratch = mpn_hgcd_itch (n - p);
      update_scratch = p + n - 1;

      dc_scratch = matrix_scratch + MAX(hgcd_scratch, update_scratch);
      if (dc_scratch > scratch)
	scratch = dc_scratch;
    }

  TMP_MARK;
  tp = TMP_ALLOC_LIMBS(scratch);

  while (ABOVE_THRESHOLD (n, JACOBI_DC_THRESHOLD))
    {
      struct hgcd_matrix M;
      mp_size_t p = 2*n/3;
      mp_size_t matrix_scratch = MPN_HGCD_MATRIX_INIT_ITCH (n - p);
      mp_size_t nn;
      mpn_hgcd_matrix_init (&M, n - p, tp);

      nn = mpn_hgcd_jacobi (ap + p, bp + p, n - p, &M, &bits,
			    tp + matrix_scratch);
      if (nn > 0)
	{
	  ASSERT (M.n <= (n - p - 1)/2);
	  ASSERT (M.n + p <= (p + n - 1) / 2);
	  /* Temporary storage 2 (p + M->n) <= p + n - 1. */
	  n = mpn_hgcd_matrix_adjust (&M, p + nn, ap, bp, p, tp + matrix_scratch);
	}
      else
	{
	  /* Temporary storage n */
	  n = mpn_gcd_subdiv_step (ap, bp, n, 0, jacobi_hook, &bits, tp);
	  if (!n)
	    {
	      TMP_FREE;
	      return bits == BITS_FAIL ? 0 : mpn_jacobi_finish (bits);
	    }
	}
    }

  while (n > 2)
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
      if (mpn_hgcd2_jacobi (ah, al, bh, bl, &M, &bits))
	{
	  n = mpn_matrix22_mul1_inverse_vector (&M, tp, ap, bp, n);
	  MP_PTR_SWAP (ap, tp);
	}
      else
	{
	  /* mpn_hgcd2 has failed. Then either one of a or b is very
	     small, or the difference is very small. Perform one
	     subtraction followed by one division. */
	  n = mpn_gcd_subdiv_step (ap, bp, n, 0, &jacobi_hook, &bits, tp);
	  if (!n)
	    {
	      TMP_FREE;
	      return bits == BITS_FAIL ? 0 : mpn_jacobi_finish (bits);
	    }
	}
    }

  if (bits >= 16)
    MP_PTR_SWAP (ap, bp);

  ASSERT (bp[0] & 1);

  if (n == 1)
    {
      mp_limb_t al, bl;
      al = ap[0];
      bl = bp[0];

      TMP_FREE;
      if (bl == 1)
	return 1 - 2*(bits & 1);
      else
	return mpn_jacobi_base (al, bl, bits << 1);
    }

  else
    {
      int res = mpn_jacobi_2 (ap, bp, bits & 1);
      TMP_FREE;
      return res;
    }
}
