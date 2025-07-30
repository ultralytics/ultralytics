/* mpn_toom_eval_pm2exp -- Evaluate a polynomial in +2^k and -2^k

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

/* Evaluates a polynomial of degree k > 2, in the points +2^shift and -2^shift. */
int
mpn_toom_eval_pm2exp (mp_ptr xp2, mp_ptr xm2, unsigned k,
		      mp_srcptr xp, mp_size_t n, mp_size_t hn, unsigned shift,
		      mp_ptr tp)
{
  unsigned i;
  int neg;
#if HAVE_NATIVE_mpn_addlsh_n
  mp_limb_t cy;
#endif

  ASSERT (k >= 3);
  ASSERT (shift*k < GMP_NUMB_BITS);

  ASSERT (hn > 0);
  ASSERT (hn <= n);

  /* The degree k is also the number of full-size coefficients, so
   * that last coefficient, of size hn, starts at xp + k*n. */

#if HAVE_NATIVE_mpn_addlsh_n
  xp2[n] = mpn_addlsh_n (xp2, xp, xp + 2*n, n, 2*shift);
  for (i = 4; i < k; i += 2)
    xp2[n] += mpn_addlsh_n (xp2, xp2, xp + i*n, n, i*shift);

  tp[n] = mpn_lshift (tp, xp+n, n, shift);
  for (i = 3; i < k; i+= 2)
    tp[n] += mpn_addlsh_n (tp, tp, xp+i*n, n, i*shift);

  if (k & 1)
    {
      cy = mpn_addlsh_n (tp, tp, xp+k*n, hn, k*shift);
      MPN_INCR_U (tp + hn, n+1 - hn, cy);
    }
  else
    {
      cy = mpn_addlsh_n (xp2, xp2, xp+k*n, hn, k*shift);
      MPN_INCR_U (xp2 + hn, n+1 - hn, cy);
    }

#else /* !HAVE_NATIVE_mpn_addlsh_n */
  xp2[n] = mpn_lshift (tp, xp+2*n, n, 2*shift);
  xp2[n] += mpn_add_n (xp2, xp, tp, n);
  for (i = 4; i < k; i += 2)
    {
      xp2[n] += mpn_lshift (tp, xp + i*n, n, i*shift);
      xp2[n] += mpn_add_n (xp2, xp2, tp, n);
    }

  tp[n] = mpn_lshift (tp, xp+n, n, shift);
  for (i = 3; i < k; i+= 2)
    {
      tp[n] += mpn_lshift (xm2, xp + i*n, n, i*shift);
      tp[n] += mpn_add_n (tp, tp, xm2, n);
    }

  xm2[hn] = mpn_lshift (xm2, xp + k*n, hn, k*shift);
  if (k & 1)
    mpn_add (tp, tp, n+1, xm2, hn+1);
  else
    mpn_add (xp2, xp2, n+1, xm2, hn+1);
#endif /* !HAVE_NATIVE_mpn_addlsh_n */

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

  /* FIXME: the following asserts are useless if (k+1)*shift >= GMP_LIMB_BITS */
  ASSERT ((k+1)*shift >= GMP_LIMB_BITS ||
	  xp2[n] < ((CNST_LIMB(1)<<((k+1)*shift))-1)/((CNST_LIMB(1)<<shift)-1));
  ASSERT ((k+2)*shift >= GMP_LIMB_BITS ||
	  xm2[n] < ((CNST_LIMB(1)<<((k+2)*shift))-((k&1)?(CNST_LIMB(1)<<shift):1))/((CNST_LIMB(1)<<(2*shift))-1));

  return neg;
}
