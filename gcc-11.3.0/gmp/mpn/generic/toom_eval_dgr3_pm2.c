/* mpn_toom_eval_dgr3_pm2 -- Evaluate a degree 3 polynomial in +2 and -2

   Contributed to the GNU project by Niels Möller

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

/* Needs n+1 limbs of temporary storage. */
int
mpn_toom_eval_dgr3_pm2 (mp_ptr xp2, mp_ptr xm2,
			mp_srcptr xp, mp_size_t n, mp_size_t x3n, mp_ptr tp)
{
  mp_limb_t cy;
  int neg;

  ASSERT (x3n > 0);
  ASSERT (x3n <= n);

  /* (x0 + 4 * x2) +/- (2 x1 + 8 x_3) */
#if HAVE_NATIVE_mpn_addlsh_n || HAVE_NATIVE_mpn_addlsh2_n
#if HAVE_NATIVE_mpn_addlsh2_n
  xp2[n] = mpn_addlsh2_n (xp2, xp, xp + 2*n, n);

  cy = mpn_addlsh2_n (tp, xp + n, xp + 3*n, x3n);
#else /* HAVE_NATIVE_mpn_addlsh_n */
  xp2[n] = mpn_addlsh_n (xp2, xp, xp + 2*n, n, 2);

  cy = mpn_addlsh_n (tp, xp + n, xp + 3*n, x3n, 2);
#endif
  if (x3n < n)
    cy = mpn_add_1 (tp + x3n, xp + n + x3n, n - x3n, cy);
  tp[n] = cy;
#else
  cy = mpn_lshift (tp, xp + 2*n, n, 2);
  xp2[n] = cy + mpn_add_n (xp2, tp, xp, n);

  tp[x3n] = mpn_lshift (tp, xp + 3*n, x3n, 2);
  if (x3n < n)
    tp[n] = mpn_add (tp, xp + n, n, tp, x3n + 1);
  else
    tp[n] += mpn_add_n (tp, xp + n, tp, n);
#endif
  mpn_lshift (tp, tp, n+1, 1);

  neg = (mpn_cmp (xp2, tp, n + 1) < 0) ? ~0 : 0;

#if HAVE_NATIVE_mpn_add_n_sub_n
  if (neg)
    mpn_add_n_sub_n (xp2, xm2, tp, xp2, n + 1);
  else
    mpn_add_n_sub_n (xp2, xm2, xp2, tp, n + 1);
#else
  if (neg)
    mpn_sub_n (xm2, tp, xp2, n + 1);
  else
    mpn_sub_n (xm2, xp2, tp, n + 1);

  mpn_add_n (xp2, xp2, tp, n + 1);
#endif

  ASSERT (xp2[n] < 15);
  ASSERT (xm2[n] < 10);

  return neg;
}
