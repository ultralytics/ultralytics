/* mpn_broot -- Compute hensel sqrt

   Contributed to the GNU project by Niels Möller

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2012 Free Software Foundation, Inc.

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

/* Computes a^e (mod B). Uses right-to-left binary algorithm, since
   typical use will have e small. */
static mp_limb_t
powlimb (mp_limb_t a, mp_limb_t e)
{
  mp_limb_t r = 1;
  mp_limb_t s = a;

  for (r = 1, s = a; e > 0; e >>= 1, s *= s)
    if (e & 1)
      r *= s;

  return r;
}

/* Computes a^{1/k - 1} (mod B^n). Both a and k must be odd.

   Iterates

     r' <-- r - r * (a^{k-1} r^k - 1) / n

   If

     a^{k-1} r^k = 1 (mod 2^m),

   then

     a^{k-1} r'^k = 1 (mod 2^{2m}),

   Compute the update term as

     r' = r - (a^{k-1} r^{k+1} - r) / k

   where we still have cancellation of low limbs.

 */
void
mpn_broot_invm1 (mp_ptr rp, mp_srcptr ap, mp_size_t n, mp_limb_t k)
{
  mp_size_t sizes[GMP_LIMB_BITS * 2];
  mp_ptr akm1, tp, rnp, ep;
  mp_limb_t a0, r0, km1, kp1h, kinv;
  mp_size_t rn;
  unsigned i;

  TMP_DECL;

  ASSERT (n > 0);
  ASSERT (ap[0] & 1);
  ASSERT (k & 1);
  ASSERT (k >= 3);

  TMP_MARK;

  akm1 = TMP_ALLOC_LIMBS (4*n);
  tp = akm1 + n;

  km1 = k-1;
  /* FIXME: Could arrange the iteration so we don't need to compute
     this up front, computing a^{k-1} * r^k as (a r)^{k-1} * r. Note
     that we can use wraparound also for a*r, since the low half is
     unchanged from the previous iteration. Or possibly mulmid. Also,
     a r = a^{1/k}, so we get that value too, for free? */
  mpn_powlo (akm1, ap, &km1, 1, n, tp); /* 3 n scratch space */

  a0 = ap[0];
  binvert_limb (kinv, k);

  /* 4 bits: a^{1/k - 1} (mod 16):

	a % 8
	1 3 5 7
   k%4 +-------
     1 |1 1 1 1
     3 |1 9 9 1
  */
  r0 = 1 + (((k << 2) & ((a0 << 1) ^ (a0 << 2))) & 8);
  r0 = kinv * r0 * (k+1 - akm1[0] * powlimb (r0, k & 0x7f)); /* 8 bits */
  r0 = kinv * r0 * (k+1 - akm1[0] * powlimb (r0, k & 0x7fff)); /* 16 bits */
  r0 = kinv * r0 * (k+1 - akm1[0] * powlimb (r0, k)); /* 32 bits */
#if GMP_NUMB_BITS > 32
  {
    unsigned prec = 32;
    do
      {
	r0 = kinv * r0 * (k+1 - akm1[0] * powlimb (r0, k));
	prec *= 2;
      }
    while (prec < GMP_NUMB_BITS);
  }
#endif

  rp[0] = r0;
  if (n == 1)
    {
      TMP_FREE;
      return;
    }

  /* For odd k, (k+1)/2 = k/2+1, and the latter avoids overflow. */
  kp1h = k/2 + 1;

  /* FIXME: Special case for two limb iteration. */
  rnp = TMP_ALLOC_LIMBS (2*n + 1);
  ep = rnp + n;

  /* FIXME: Possible to this on the fly with some bit fiddling. */
  for (i = 0; n > 1; n = (n + 1)/2)
    sizes[i++] = n;

  rn = 1;

  while (i-- > 0)
    {
      /* Compute x^{k+1}. */
      mpn_sqr (ep, rp, rn); /* For odd n, writes n+1 limbs in the
			       final iteration. */
      mpn_powlo (rnp, ep, &kp1h, 1, sizes[i], tp);

      /* Multiply by a^{k-1}. Can use wraparound; low part equals r. */

      mpn_mullo_n (ep, rnp, akm1, sizes[i]);
      ASSERT (mpn_cmp (ep, rp, rn) == 0);

      ASSERT (sizes[i] <= 2*rn);
      mpn_pi1_bdiv_q_1 (rp + rn, ep + rn, sizes[i] - rn, k, kinv, 0);
      mpn_neg (rp + rn, rp + rn, sizes[i] - rn);
      rn = sizes[i];
    }
  TMP_FREE;
}

/* Computes a^{1/k} (mod B^n). Both a and k must be odd. */
void
mpn_broot (mp_ptr rp, mp_srcptr ap, mp_size_t n, mp_limb_t k)
{
  mp_ptr tp;
  TMP_DECL;

  ASSERT (n > 0);
  ASSERT (ap[0] & 1);
  ASSERT (k & 1);

  if (k == 1)
    {
      MPN_COPY (rp, ap, n);
      return;
    }

  TMP_MARK;
  tp = TMP_ALLOC_LIMBS (n);

  mpn_broot_invm1 (tp, ap, n, k);
  mpn_mullo_n (rp, tp, ap, n);

  TMP_FREE;
}
