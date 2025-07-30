/* mpn_toom_eval_pm2 -- Evaluate a polynomial in +2 and -2

   Contributed to the GNU project by Niels Möller and Marco Bodrato

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2009 Free Software Foundation, Inc.

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

/* DO_addlsh2(d,a,b,n,cy) computes cy,{d,n} <- {a,n} + 4*(cy,{b,n}), it
   can be used as DO_addlsh2(d,a,d,n,d[n]), for accumulation on {d,n+1}. */
#if HAVE_NATIVE_mpn_addlsh2_n
#define DO_addlsh2(d, a, b, n, cy)	\
do {					\
  (cy) <<= 2;				\
  (cy) += mpn_addlsh2_n(d, a, b, n);	\
} while (0)
#else
#if HAVE_NATIVE_mpn_addlsh_n
#define DO_addlsh2(d, a, b, n, cy)	\
do {					\
  (cy) <<= 2;				\
  (cy) += mpn_addlsh_n(d, a, b, n, 2);	\
} while (0)
#else
/* The following is not a general substitute for addlsh2.
   It is correct if d == b, but it is not if d == a.  */
#define DO_addlsh2(d, a, b, n, cy)	\
do {					\
  (cy) <<= 2;				\
  (cy) += mpn_lshift(d, b, n, 2);	\
  (cy) += mpn_add_n(d, d, a, n);	\
} while (0)
#endif
#endif

/* Evaluates a polynomial of degree 2 < k < GMP_NUMB_BITS, in the
   points +2 and -2. */
int
mpn_toom_eval_pm2 (mp_ptr xp2, mp_ptr xm2, unsigned k,
		   mp_srcptr xp, mp_size_t n, mp_size_t hn, mp_ptr tp)
{
  int i;
  int neg;
  mp_limb_t cy;

  ASSERT (k >= 3);
  ASSERT (k < GMP_NUMB_BITS);

  ASSERT (hn > 0);
  ASSERT (hn <= n);

  /* The degree k is also the number of full-size coefficients, so
   * that last coefficient, of size hn, starts at xp + k*n. */

  cy = 0;
  DO_addlsh2 (xp2, xp + (k-2) * n, xp + k * n, hn, cy);
  if (hn != n)
    cy = mpn_add_1 (xp2 + hn, xp + (k-2) * n + hn, n - hn, cy);
  for (i = k - 4; i >= 0; i -= 2)
    DO_addlsh2 (xp2, xp + i * n, xp2, n, cy);
  xp2[n] = cy;

  k--;

  cy = 0;
  DO_addlsh2 (tp, xp + (k-2) * n, xp + k * n, n, cy);
  for (i = k - 4; i >= 0; i -= 2)
    DO_addlsh2 (tp, xp + i * n, tp, n, cy);
  tp[n] = cy;

  if (k & 1)
    ASSERT_NOCARRY(mpn_lshift (tp , tp , n + 1, 1));
  else
    ASSERT_NOCARRY(mpn_lshift (xp2, xp2, n + 1, 1));

  neg = (mpn_cmp (xp2, tp, n + 1) < 0) ? ~0 : 0;

#if HAVE_NATIVE_mpn_add_n_sub_n
  if (neg)
    mpn_add_n_sub_n (xp2, xm2, tp, xp2, n + 1);
  else
    mpn_add_n_sub_n (xp2, xm2, xp2, tp, n + 1);
#else /* !HAVE_NATIVE_mpn_add_n_sub_n */
  if (neg)
    mpn_sub_n (xm2, tp, xp2, n + 1);
  else
    mpn_sub_n (xm2, xp2, tp, n + 1);

  mpn_add_n (xp2, xp2, tp, n + 1);
#endif /* !HAVE_NATIVE_mpn_add_n_sub_n */

  ASSERT (xp2[n] < (1<<(k+2))-1);
  ASSERT (xm2[n] < ((1<<(k+3))-1 - (1^k&1))/3);

  neg ^= ((k & 1) - 1);

  return neg;
}

#undef DO_addlsh2
