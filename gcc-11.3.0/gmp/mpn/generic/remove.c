/* mpn_remove -- divide out all multiples of odd mpn number from another mpn
   number.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2009, 2012-2014 Free Software Foundation, Inc.

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

#if GMP_LIMB_BITS > 50
#define LOG 50
#else
#define LOG GMP_LIMB_BITS
#endif


/* Input: U = {up,un}, V = {vp,vn} must be odd, cap
   Ouput  W = {wp,*wn} allocation need is exactly *wn

   Set W = U / V^k, where k is the largest integer <= cap such that the
   division yields an integer.

   FIXME: We currently allow any operand overlap.  This is quite non mpn-ish
   and might be changed, since it cost significant temporary space.
   * If we require W to have space for un + 1 limbs, we could save qp or qp2
     (but we will still need to copy things into wp 50% of the time).
   * If we allow ourselves to clobber U, we could save the other of qp and qp2,
     and the initial COPY (but also here we would need un + 1 limbs).
*/

/* FIXME: We need to wrap mpn_bdiv_qr due to the itch interface.  This need
   indicates a flaw in the current itch mechanism: Which operands not greater
   than un,un will incur the worst itch?  We need a parallel foo_maxitch set
   of functions.  */
static void
mpn_bdiv_qr_wrap (mp_ptr qp, mp_ptr rp,
		  mp_srcptr np, mp_size_t nn,
		  mp_srcptr dp, mp_size_t dn)
{
  mp_ptr scratch_out;
  TMP_DECL;

  TMP_MARK;
  scratch_out = TMP_ALLOC_LIMBS (mpn_bdiv_qr_itch (nn, dn));
  mpn_bdiv_qr (qp, rp, np, nn, dp, dn, scratch_out);

  TMP_FREE;
}

mp_bitcnt_t
mpn_remove (mp_ptr wp, mp_size_t *wn,
	    mp_srcptr up, mp_size_t un, mp_srcptr vp, mp_size_t vn,
	    mp_bitcnt_t cap)
{
  mp_srcptr pwpsp[LOG];
  mp_size_t pwpsn[LOG];
  mp_size_t npowers;
  mp_ptr tp, qp, np, qp2;
  mp_srcptr pp;
  mp_size_t pn, nn, qn, i;
  mp_bitcnt_t pwr;
  TMP_DECL;

  ASSERT (un > 0);
  ASSERT (vn > 0);
  ASSERT (vp[0] % 2 != 0);	/* 2-adic division wants odd numbers */
  ASSERT (vn > 1 || vp[0] > 1);	/* else we would loop indefinitely */

  TMP_MARK;

  TMP_ALLOC_LIMBS_3 (qp, un + 1,	/* quotient, alternating */
		     qp2, un + 1,	/* quotient, alternating */
		     tp, (un + 1 + vn) / 2); /* remainder */
  pp = vp;
  pn = vn;

  MPN_COPY (qp, up, un);
  qn = un;

  npowers = 0;
  while (qn >= pn)
    {
      qp[qn] = 0;
      mpn_bdiv_qr_wrap (qp2, tp, qp, qn + 1, pp, pn);
      if (!mpn_zero_p (tp, pn))
	break;			/* could not divide by V^npowers */

      MP_PTR_SWAP (qp, qp2);
      qn = qn - pn;
      qn += qp[qn] != 0;

      pwpsp[npowers] = pp;
      pwpsn[npowers] = pn;
      ++npowers;

      if (((mp_bitcnt_t) 2 << npowers) - 1 > cap)
	break;

      nn = 2 * pn - 1;		/* next power will be at least this large */
      if (nn > qn)
	break;			/* next power would be overlarge */

      if (npowers == 1)		/* Alloc once, but only if it's needed */
	np = TMP_ALLOC_LIMBS (qn + LOG);	/* powers of V */
      else
	np += pn;

      mpn_sqr (np, pp, pn);
      pn = nn + (np[nn] != 0);
      pp = np;
    }

  pwr = ((mp_bitcnt_t) 1 << npowers) - 1;

  for (i = npowers; --i >= 0;)
    {
      pn = pwpsn[i];
      if (qn < pn)
	continue;

      if (pwr + ((mp_bitcnt_t) 1 << i) > cap)
	continue;		/* V^i would bring us past cap */

      qp[qn] = 0;
      mpn_bdiv_qr_wrap (qp2, tp, qp, qn + 1, pwpsp[i], pn);
      if (!mpn_zero_p (tp, pn))
	continue;		/* could not divide by V^i */

      MP_PTR_SWAP (qp, qp2);
      qn = qn - pn;
      qn += qp[qn] != 0;

      pwr += (mp_bitcnt_t) 1 << i;
    }

  MPN_COPY (wp, qp, qn);
  *wn = qn;

  TMP_FREE;

  return pwr;
}
