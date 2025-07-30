/* mpf_div -- Divide two floats.

Copyright 1993, 1994, 1996, 2000-2002, 2004, 2005, 2010, 2012 Free Software
Foundation, Inc.

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


/* Not done:

   No attempt is made to identify an overlap u==v.  The result will be
   correct (1.0), but a full actual division is done whereas of course
   x/x==1 needs no work.  Such a call is not a sensible thing to make, and
   it's left to an application to notice and optimize if it might arise
   somehow through pointer aliasing or whatever.

   Enhancements:

   The high quotient limb is non-zero when high{up,vsize} >= {vp,vsize}.  We
   could make that comparison and use qsize==prec instead of qsize==prec+1,
   to save one limb in the division.

   If r==u but the size is enough bigger than prec that there won't be an
   overlap between quotient and dividend in mpn_div_q, then we can avoid
   copying up,usize.  This would only arise from a prec reduced with
   mpf_set_prec_raw and will be pretty unusual, but might be worthwhile if
   it could be worked into the copy_u decision cleanly.  */

void
mpf_div (mpf_ptr r, mpf_srcptr u, mpf_srcptr v)
{
  mp_srcptr up, vp;
  mp_ptr rp, tp, new_vp;
  mp_size_t usize, vsize, rsize, prospective_rsize, tsize, zeros;
  mp_size_t sign_quotient, prec, high_zero, chop;
  mp_exp_t rexp;
  int copy_u;
  TMP_DECL;

  usize = SIZ(u);
  vsize = SIZ(v);

  if (UNLIKELY (vsize == 0))
    DIVIDE_BY_ZERO;

  if (usize == 0)
    {
      SIZ(r) = 0;
      EXP(r) = 0;
      return;
    }

  sign_quotient = usize ^ vsize;
  usize = ABS (usize);
  vsize = ABS (vsize);
  prec = PREC(r);

  TMP_MARK;
  rexp = EXP(u) - EXP(v) + 1;

  rp = PTR(r);
  up = PTR(u);
  vp = PTR(v);

  prospective_rsize = usize - vsize + 1; /* quot from using given u,v sizes */
  rsize = prec + 1;			 /* desired quot */

  zeros = rsize - prospective_rsize;	 /* padding u to give rsize */
  copy_u = (zeros > 0 || rp == up);	 /* copy u if overlap or padding */

  chop = MAX (-zeros, 0);		 /* negative zeros means shorten u */
  up += chop;
  usize -= chop;
  zeros += chop;			 /* now zeros >= 0 */

  tsize = usize + zeros;		 /* size for possible copy of u */

  /* copy and possibly extend u if necessary */
  if (copy_u)
    {
      tp = TMP_ALLOC_LIMBS (tsize + 1);	/* +1 for mpn_div_q's scratch needs */
      MPN_ZERO (tp, zeros);
      MPN_COPY (tp+zeros, up, usize);
      up = tp;
      usize = tsize;
    }
  else
    {
      tp = TMP_ALLOC_LIMBS (usize + 1);
    }

  /* ensure divisor doesn't overlap quotient */
  if (rp == vp)
    {
      new_vp = TMP_ALLOC_LIMBS (vsize);
      MPN_COPY (new_vp, vp, vsize);
      vp = new_vp;
    }

  ASSERT (usize-vsize+1 == rsize);
  mpn_div_q (rp, up, usize, vp, vsize, tp);

  /* strip possible zero high limb */
  high_zero = (rp[rsize-1] == 0);
  rsize -= high_zero;
  rexp -= high_zero;

  SIZ(r) = sign_quotient >= 0 ? rsize : -rsize;
  EXP(r) = rexp;
  TMP_FREE;
}
