/* matrix22_mul.c.

   Contributed by Niels Möller and Marco Bodrato.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2003-2005, 2008, 2009 Free Software Foundation, Inc.

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

#define MUL(rp, ap, an, bp, bn) do {		\
  if (an >= bn)					\
    mpn_mul (rp, ap, an, bp, bn);		\
  else						\
    mpn_mul (rp, bp, bn, ap, an);		\
} while (0)

/* Inputs are unsigned. */
static int
abs_sub_n (mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  int c;
  MPN_CMP (c, ap, bp, n);
  if (c >= 0)
    {
      mpn_sub_n (rp, ap, bp, n);
      return 0;
    }
  else
    {
      mpn_sub_n (rp, bp, ap, n);
      return 1;
    }
}

static int
add_signed_n (mp_ptr rp,
	      mp_srcptr ap, int as, mp_srcptr bp, int bs, mp_size_t n)
{
  if (as != bs)
    return as ^ abs_sub_n (rp, ap, bp, n);
  else
    {
      ASSERT_NOCARRY (mpn_add_n (rp, ap, bp, n));
      return as;
    }
}

mp_size_t
mpn_matrix22_mul_itch (mp_size_t rn, mp_size_t mn)
{
  if (BELOW_THRESHOLD (rn, MATRIX22_STRASSEN_THRESHOLD)
      || BELOW_THRESHOLD (mn, MATRIX22_STRASSEN_THRESHOLD))
    return 3*rn + 2*mn;
  else
    return 3*(rn + mn) + 5;
}

/* Algorithm:

    / s0 \   /  1  0  0  0 \ / r0 \
    | s1 |   |  0  1  0  1 | | r1 |
    | s2 |   |  0  0 -1  1 | | r2 |
    | s3 | = |  0  1 -1  1 | \ r3 /
    | s4 |   | -1  1 -1  1 |
    | s5 |   |  0  1  0  0 |
    \ s6 /   \  0  0  1  0 /

    / t0 \   /  1  0  0  0 \ / m0 \
    | t1 |   |  0  1  0  1 | | m1 |
    | t2 |   |  0  0 -1  1 | | m2 |
    | t3 | = |  0  1 -1  1 | \ m3 /
    | t4 |   | -1  1 -1  1 |
    | t5 |   |  0  1  0  0 |
    \ t6 /   \  0  0  1  0 /

  Note: the two matrices above are the same, but s_i and t_i are used
  in the same product, only for i<4, see "A Strassen-like Matrix
  Multiplication suited for squaring and higher power computation" by
  M. Bodrato, in Proceedings of ISSAC 2010.

    / r0 \   / 1 0  0  0  0  1  0 \ / s0*t0 \
    | r1 | = | 0 0 -1  1 -1  1  0 | | s1*t1 |
    | r2 |   | 0 1  0 -1  0 -1 -1 | | s2*t2 |
    \ r3 /   \ 0 1  1 -1  0 -1  0 / | s3*t3 |
				    | s4*t5 |
				    | s5*t6 |
				    \ s6*t4 /

  The scheduling uses two temporaries U0 and U1 to store products, and
  two, S0 and T0, to store combinations of entries of the two
  operands.
*/

/* Computes R = R * M. Elements are numbers R = (r0, r1; r2, r3).
 *
 * Resulting elements are of size up to rn + mn + 1.
 *
 * Temporary storage: 3 rn + 3 mn + 5. */
void
mpn_matrix22_mul_strassen (mp_ptr r0, mp_ptr r1, mp_ptr r2, mp_ptr r3, mp_size_t rn,
			   mp_srcptr m0, mp_srcptr m1, mp_srcptr m2, mp_srcptr m3, mp_size_t mn,
			   mp_ptr tp)
{
  mp_ptr s0, t0, u0, u1;
  int r1s, r3s, s0s, t0s, u1s;
  s0 = tp; tp += rn + 1;
  t0 = tp; tp += mn + 1;
  u0 = tp; tp += rn + mn + 1;
  u1 = tp; /* rn + mn + 2 */

  MUL (u0, r1, rn, m2, mn);		/* u5 = s5 * t6 */
  r3s = abs_sub_n (r3, r3, r2, rn);	/* r3 - r2 */
  if (r3s)
    {
      r1s = abs_sub_n (r1, r1, r3, rn);
      r1[rn] = 0;
    }
  else
    {
      r1[rn] = mpn_add_n (r1, r1, r3, rn);
      r1s = 0;				/* r1 - r2 + r3  */
    }
  if (r1s)
    {
      s0[rn] = mpn_add_n (s0, r1, r0, rn);
      s0s = 0;
    }
  else if (r1[rn] != 0)
    {
      s0[rn] = r1[rn] - mpn_sub_n (s0, r1, r0, rn);
      s0s = 1;				/* s4 = -r0 + r1 - r2 + r3 */
					/* Reverse sign! */
    }
  else
    {
      s0s = abs_sub_n (s0, r0, r1, rn);
      s0[rn] = 0;
    }
  MUL (u1, r0, rn, m0, mn);		/* u0 = s0 * t0 */
  r0[rn+mn] = mpn_add_n (r0, u0, u1, rn + mn);
  ASSERT (r0[rn+mn] < 2);		/* u0 + u5 */

  t0s = abs_sub_n (t0, m3, m2, mn);
  u1s = r3s^t0s^1;			/* Reverse sign! */
  MUL (u1, r3, rn, t0, mn);		/* u2 = s2 * t2 */
  u1[rn+mn] = 0;
  if (t0s)
    {
      t0s = abs_sub_n (t0, m1, t0, mn);
      t0[mn] = 0;
    }
  else
    {
      t0[mn] = mpn_add_n (t0, t0, m1, mn);
    }

  /* FIXME: Could be simplified if we had space for rn + mn + 2 limbs
     at r3. I'd expect that for matrices of random size, the high
     words t0[mn] and r1[rn] are non-zero with a pretty small
     probability. If that can be confirmed this should be done as an
     unconditional rn x (mn+1) followed by an if (UNLIKELY (r1[rn]))
     add_n. */
  if (t0[mn] != 0)
    {
      MUL (r3, r1, rn, t0, mn + 1);	/* u3 = s3 * t3 */
      ASSERT (r1[rn] < 2);
      if (r1[rn] != 0)
	mpn_add_n (r3 + rn, r3 + rn, t0, mn + 1);
    }
  else
    {
      MUL (r3, r1, rn + 1, t0, mn);
    }

  ASSERT (r3[rn+mn] < 4);

  u0[rn+mn] = 0;
  if (r1s^t0s)
    {
      r3s = abs_sub_n (r3, u0, r3, rn + mn + 1);
    }
  else
    {
      ASSERT_NOCARRY (mpn_add_n (r3, r3, u0, rn + mn + 1));
      r3s = 0;				/* u3 + u5 */
    }

  if (t0s)
    {
      t0[mn] = mpn_add_n (t0, t0, m0, mn);
    }
  else if (t0[mn] != 0)
    {
      t0[mn] -= mpn_sub_n (t0, t0, m0, mn);
    }
  else
    {
      t0s = abs_sub_n (t0, t0, m0, mn);
    }
  MUL (u0, r2, rn, t0, mn + 1);		/* u6 = s6 * t4 */
  ASSERT (u0[rn+mn] < 2);
  if (r1s)
    {
      ASSERT_NOCARRY (mpn_sub_n (r1, r2, r1, rn));
    }
  else
    {
      r1[rn] += mpn_add_n (r1, r1, r2, rn);
    }
  rn++;
  t0s = add_signed_n (r2, r3, r3s, u0, t0s, rn + mn);
					/* u3 + u5 + u6 */
  ASSERT (r2[rn+mn-1] < 4);
  r3s = add_signed_n (r3, r3, r3s, u1, u1s, rn + mn);
					/* -u2 + u3 + u5  */
  ASSERT (r3[rn+mn-1] < 3);
  MUL (u0, s0, rn, m1, mn);		/* u4 = s4 * t5 */
  ASSERT (u0[rn+mn-1] < 2);
  t0[mn] = mpn_add_n (t0, m3, m1, mn);
  MUL (u1, r1, rn, t0, mn + 1);		/* u1 = s1 * t1 */
  mn += rn;
  ASSERT (u1[mn-1] < 4);
  ASSERT (u1[mn] == 0);
  ASSERT_NOCARRY (add_signed_n (r1, r3, r3s, u0, s0s, mn));
					/* -u2 + u3 - u4 + u5  */
  ASSERT (r1[mn-1] < 2);
  if (r3s)
    {
      ASSERT_NOCARRY (mpn_add_n (r3, u1, r3, mn));
    }
  else
    {
      ASSERT_NOCARRY (mpn_sub_n (r3, u1, r3, mn));
					/* u1 + u2 - u3 - u5  */
    }
  ASSERT (r3[mn-1] < 2);
  if (t0s)
    {
      ASSERT_NOCARRY (mpn_add_n (r2, u1, r2, mn));
    }
  else
    {
      ASSERT_NOCARRY (mpn_sub_n (r2, u1, r2, mn));
					/* u1 - u3 - u5 - u6  */
    }
  ASSERT (r2[mn-1] < 2);
}

void
mpn_matrix22_mul (mp_ptr r0, mp_ptr r1, mp_ptr r2, mp_ptr r3, mp_size_t rn,
		  mp_srcptr m0, mp_srcptr m1, mp_srcptr m2, mp_srcptr m3, mp_size_t mn,
		  mp_ptr tp)
{
  if (BELOW_THRESHOLD (rn, MATRIX22_STRASSEN_THRESHOLD)
      || BELOW_THRESHOLD (mn, MATRIX22_STRASSEN_THRESHOLD))
    {
      mp_ptr p0, p1;
      unsigned i;

      /* Temporary storage: 3 rn + 2 mn */
      p0 = tp + rn;
      p1 = p0 + rn + mn;

      for (i = 0; i < 2; i++)
	{
	  MPN_COPY (tp, r0, rn);

	  if (rn >= mn)
	    {
	      mpn_mul (p0, r0, rn, m0, mn);
	      mpn_mul (p1, r1, rn, m3, mn);
	      mpn_mul (r0, r1, rn, m2, mn);
	      mpn_mul (r1, tp, rn, m1, mn);
	    }
	  else
	    {
	      mpn_mul (p0, m0, mn, r0, rn);
	      mpn_mul (p1, m3, mn, r1, rn);
	      mpn_mul (r0, m2, mn, r1, rn);
	      mpn_mul (r1, m1, mn, tp, rn);
	    }
	  r0[rn+mn] = mpn_add_n (r0, r0, p0, rn + mn);
	  r1[rn+mn] = mpn_add_n (r1, r1, p1, rn + mn);

	  r0 = r2; r1 = r3;
	}
    }
  else
    mpn_matrix22_mul_strassen (r0, r1, r2, r3, rn,
			       m0, m1, m2, m3, mn, tp);
}
