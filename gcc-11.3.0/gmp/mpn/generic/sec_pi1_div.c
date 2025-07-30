/* mpn_sec_pi1_div_qr, mpn_sec_pi1_div_r -- Compute Q = floor(U / V), U = U
   mod V.  Side-channel silent under the assumption that the used instructions
   are side-channel silent.

   Contributed to the GNU project by Torbjörn Granlund.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2011-2013 Free Software Foundation, Inc.

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

/* This side-channel silent division algorithm reduces the partial remainder by
   GMP_NUMB_BITS/2 bits at a time, compared to GMP_NUMB_BITS for the main
   division algorithm.  We actually do not insist on reducing by exactly
   GMP_NUMB_BITS/2, but may leave a partial remainder that is D*B^i to 3D*B^i
   too large (B is the limb base, D is the divisor, and i is the induction
   variable); the subsequent step will handle the extra partial remainder bits.

   With that partial remainder reduction, each step generates a quotient "half
   limb".  The outer loop generates two quotient half limbs, an upper (q1h) and
   a lower (q0h) which are stored sparsely in separate limb arrays.  These
   arrays are added at the end; using separate arrays avoids data-dependent
   carry propagation which could else pose a side-channel leakage problem.

   The quotient half limbs may be between -3 to 0 from the accurate value
   ("accurate" being the one which corresponds to a reduction to a principal
   partial remainder).  Too small quotient half limbs correspond to too large
   remainders, which we reduce later, as described above.

   In order to keep quotients from getting too big, corresponding to a negative
   partial remainder, we use an inverse which is slightly smaller than usually.
*/

#if OPERATION_sec_pi1_div_qr
/* Needs (dn + 1) + (nn - dn) + (nn - dn) = 2nn - dn + 1 limbs at tp. */
#define FNAME mpn_sec_pi1_div_qr
#define Q(q) q,
#define RETTYPE mp_limb_t
#endif
#if OPERATION_sec_pi1_div_r
/* Needs (dn + 1) limbs at tp.  */
#define FNAME mpn_sec_pi1_div_r
#define Q(q)
#define RETTYPE void
#endif

RETTYPE
FNAME (Q(mp_ptr qp)
       mp_ptr np, mp_size_t nn,
       mp_srcptr dp, mp_size_t dn,
       mp_limb_t dinv,
       mp_ptr tp)
{
  mp_limb_t nh, cy, q1h, q0h, dummy, cnd;
  mp_size_t i;
  mp_ptr hp;
#if OPERATION_sec_pi1_div_qr
  mp_limb_t qh;
  mp_ptr qlp, qhp;
#endif

  ASSERT (dn >= 1);
  ASSERT (nn >= dn);
  ASSERT ((dp[dn - 1] & GMP_NUMB_HIGHBIT) != 0);

  if (nn == dn)
    {
      cy = mpn_sub_n (np, np, dp, dn);
      mpn_cnd_add_n (cy, np, np, dp, dn);
#if OPERATION_sec_pi1_div_qr
      return 1 - cy;
#else
      return;
#endif
    }

  /* Create a divisor copy shifted half a limb.  */
  hp = tp;					/* (dn + 1) limbs */
  hp[dn] = mpn_lshift (hp, dp, dn, GMP_NUMB_BITS / 2);

#if OPERATION_sec_pi1_div_qr
  qlp = tp + (dn + 1);				/* (nn - dn) limbs */
  qhp = tp + (nn + 1);				/* (nn - dn) limbs */
#endif

  np += nn - dn;
  nh = 0;

  for (i = nn - dn - 1; i >= 0; i--)
    {
      np--;

      nh = (nh << GMP_NUMB_BITS/2) + (np[dn] >> GMP_NUMB_BITS/2);
      umul_ppmm (q1h, dummy, nh, dinv);
      q1h += nh;
#if OPERATION_sec_pi1_div_qr
      qhp[i] = q1h;
#endif
      mpn_submul_1 (np, hp, dn + 1, q1h);

      nh = np[dn];
      umul_ppmm (q0h, dummy, nh, dinv);
      q0h += nh;
#if OPERATION_sec_pi1_div_qr
      qlp[i] = q0h;
#endif
      nh -= mpn_submul_1 (np, dp, dn, q0h);
    }

  /* 1st adjustment depends on extra high remainder limb.  */
  cnd = nh != 0;				/* FIXME: cmp-to-int */
#if OPERATION_sec_pi1_div_qr
  qlp[0] += cnd;
#endif
  nh -= mpn_cnd_sub_n (cnd, np, np, dp, dn);

  /* 2nd adjustment depends on remainder/divisor comparison as well as whether
     extra remainder limb was nullified by previous subtract.  */
  cy = mpn_sub_n (np, np, dp, dn);
  cy = cy - nh;
#if OPERATION_sec_pi1_div_qr
  qlp[0] += 1 - cy;
#endif
  mpn_cnd_add_n (cy, np, np, dp, dn);

  /* 3rd adjustment depends on remainder/divisor comparison.  */
  cy = mpn_sub_n (np, np, dp, dn);
#if OPERATION_sec_pi1_div_qr
  qlp[0] += 1 - cy;
#endif
  mpn_cnd_add_n (cy, np, np, dp, dn);

#if OPERATION_sec_pi1_div_qr
  /* Combine quotient halves into final quotient.  */
  qh = mpn_lshift (qhp, qhp, nn - dn, GMP_NUMB_BITS/2);
  qh += mpn_add_n (qp, qhp, qlp, nn - dn);

  return qh;
#else
  return;
#endif
}
