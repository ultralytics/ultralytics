/* hgcd_step.c.

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


static void
hgcd_hook (void *p, mp_srcptr gp, mp_size_t gn,
	   mp_srcptr qp, mp_size_t qn, int d)
{
  ASSERT (!gp);
  ASSERT (d >= 0);
  ASSERT (d <= 1);

  MPN_NORMALIZE (qp, qn);
  if (qn > 0)
    {
      struct hgcd_matrix *M = (struct hgcd_matrix *) p;
      /* NOTES: This is a bit ugly. A tp area is passed to
	 gcd_subdiv_step, which stores q at the start of that area. We
	 now use the rest. */
      mp_ptr tp = (mp_ptr) qp + qn;
      mpn_hgcd_matrix_update_q (M, qp, qn, d, tp);
    }
}

/* Perform a few steps, using some of mpn_hgcd2, subtraction and
   division. Reduces the size by almost one limb or more, but never
   below the given size s. Return new size for a and b, or 0 if no
   more steps are possible.

   If hgcd2 succeeds, needs temporary space for hgcd_matrix_mul_1, M->n
   limbs, and hgcd_mul_matrix1_inverse_vector, n limbs. If hgcd2
   fails, needs space for the quotient, qn <= n - s limbs, for and
   hgcd_matrix_update_q, qn + (size of the appropriate column of M) <=
   (resulting size of M) + 1.

   If N is the input size to the calling hgcd, then s = floor(N/2) +
   1, M->n < N, qn + product size <= n - s + n - s + 1 = 2 (n - s) + 1
   <= N.
*/

mp_size_t
mpn_hgcd_step (mp_size_t n, mp_ptr ap, mp_ptr bp, mp_size_t s,
	       struct hgcd_matrix *M, mp_ptr tp)
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
  if (mpn_hgcd2 (ah, al, bh, bl, &M1))
    {
      /* Multiply M <- M * M1 */
      mpn_hgcd_matrix_mul_1 (M, &M1, tp);

      /* Can't swap inputs, so we need to copy. */
      MPN_COPY (tp, ap, n);
      /* Multiply M1^{-1} (a;b) */
      return mpn_matrix22_mul1_inverse_vector (&M1, ap, tp, bp, n);
    }

 subtract:

  return mpn_gcd_subdiv_step (ap, bp, n, s, hgcd_hook, M, tp);
}
