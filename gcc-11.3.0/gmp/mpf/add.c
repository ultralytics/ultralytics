/* mpf_add -- Add two floats.

Copyright 1993, 1994, 1996, 2000, 2001, 2005 Free Software Foundation, Inc.

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

void
mpf_add (mpf_ptr r, mpf_srcptr u, mpf_srcptr v)
{
  mp_srcptr up, vp;
  mp_ptr rp, tp;
  mp_size_t usize, vsize, rsize;
  mp_size_t prec;
  mp_exp_t uexp;
  mp_size_t ediff;
  mp_limb_t cy;
  int negate;
  TMP_DECL;

  usize = u->_mp_size;
  vsize = v->_mp_size;

  /* Handle special cases that don't work in generic code below.  */
  if (usize == 0)
    {
    set_r_v_maybe:
      if (r != v)
        mpf_set (r, v);
      return;
    }
  if (vsize == 0)
    {
      v = u;
      goto set_r_v_maybe;
    }

  /* If signs of U and V are different, perform subtraction.  */
  if ((usize ^ vsize) < 0)
    {
      __mpf_struct v_negated;
      v_negated._mp_size = -vsize;
      v_negated._mp_exp = v->_mp_exp;
      v_negated._mp_d = v->_mp_d;
      mpf_sub (r, u, &v_negated);
      return;
    }

  TMP_MARK;

  /* Signs are now known to be the same.  */
  negate = usize < 0;

  /* Make U be the operand with the largest exponent.  */
  if (u->_mp_exp < v->_mp_exp)
    {
      mpf_srcptr t;
      t = u; u = v; v = t;
      usize = u->_mp_size;
      vsize = v->_mp_size;
    }

  usize = ABS (usize);
  vsize = ABS (vsize);
  up = u->_mp_d;
  vp = v->_mp_d;
  rp = r->_mp_d;
  prec = r->_mp_prec;
  uexp = u->_mp_exp;
  ediff = u->_mp_exp - v->_mp_exp;

  /* If U extends beyond PREC, ignore the part that does.  */
  if (usize > prec)
    {
      up += usize - prec;
      usize = prec;
    }

  /* If V extends beyond PREC, ignore the part that does.
     Note that this may make vsize negative.  */
  if (vsize + ediff > prec)
    {
      vp += vsize + ediff - prec;
      vsize = prec - ediff;
    }

#if 0
  /* Locate the least significant non-zero limb in (the needed parts
     of) U and V, to simplify the code below.  */
  while (up[0] == 0)
    up++, usize--;
  while (vp[0] == 0)
    vp++, vsize--;
#endif

  /* Allocate temp space for the result.  Allocate
     just vsize + ediff later???  */
  tp = TMP_ALLOC_LIMBS (prec);

  if (ediff >= prec)
    {
      /* V completely cancelled.  */
      if (rp != up)
	MPN_COPY_INCR (rp, up, usize);
      rsize = usize;
    }
  else
    {
      /* uuuu     |  uuuu     |  uuuu     |  uuuu     |  uuuu    */
      /* vvvvvvv  |  vv       |    vvvvv  |    v      |       vv */

      if (usize > ediff)
	{
	  /* U and V partially overlaps.  */
	  if (vsize + ediff <= usize)
	    {
	      /* uuuu     */
	      /*   v      */
	      mp_size_t size;
	      size = usize - ediff - vsize;
	      MPN_COPY (tp, up, size);
	      cy = mpn_add (tp + size, up + size, usize - size, vp, vsize);
	      rsize = usize;
	    }
	  else
	    {
	      /* uuuu     */
	      /*   vvvvv  */
	      mp_size_t size;
	      size = vsize + ediff - usize;
	      MPN_COPY (tp, vp, size);
	      cy = mpn_add (tp + size, up, usize, vp + size, usize - ediff);
	      rsize = vsize + ediff;
	    }
	}
      else
	{
	  /* uuuu     */
	  /*      vv  */
	  mp_size_t size;
	  size = vsize + ediff - usize;
	  MPN_COPY (tp, vp, vsize);
	  MPN_ZERO (tp + vsize, ediff - usize);
	  MPN_COPY (tp + size, up, usize);
	  cy = 0;
	  rsize = size + usize;
	}

      MPN_COPY (rp, tp, rsize);
      rp[rsize] = cy;
      rsize += cy;
      uexp += cy;
    }

  r->_mp_size = negate ? -rsize : rsize;
  r->_mp_exp = uexp;
  TMP_FREE;
}
