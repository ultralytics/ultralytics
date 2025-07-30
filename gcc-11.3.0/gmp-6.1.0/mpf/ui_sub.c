/* mpf_ui_sub -- Subtract a float from an unsigned long int.

Copyright 1993-1996, 2001, 2002, 2005, 2014 Free Software Foundation, Inc.

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
mpf_ui_sub (mpf_ptr r, unsigned long int u, mpf_srcptr v)
{
#if 1
  __mpf_struct uu;
  mp_limb_t ul;

  if (u == 0)
    {
      mpf_neg (r, v);
      return;
    }

  ul = u;
  uu._mp_size = 1;
  uu._mp_d = &ul;
  uu._mp_exp = 1;
  mpf_sub (r, &uu, v);

#else
  mp_srcptr up, vp;
  mp_ptr rp, tp;
  mp_size_t usize, vsize, rsize;
  mp_size_t prec;
  mp_exp_t uexp;
  mp_size_t ediff;
  int negate;
  mp_limb_t ulimb;
  TMP_DECL;

  vsize = v->_mp_size;

  /* Handle special cases that don't work in generic code below.  */
  if (u == 0)
    {
      mpf_neg (r, v);
      return;
    }
  if (vsize == 0)
    {
      mpf_set_ui (r, u);
      return;
    }

  /* If signs of U and V are different, perform addition.  */
  if (vsize < 0)
    {
      __mpf_struct v_negated;
      v_negated._mp_size = -vsize;
      v_negated._mp_exp = v->_mp_exp;
      v_negated._mp_d = v->_mp_d;
      mpf_add_ui (r, &v_negated, u);
      return;
    }

  /* Signs are now known to be the same.  */
  ASSERT (vsize > 0);
  ulimb = u;
  /* Make U be the operand with the largest exponent.  */
  negate = 1 < v->_mp_exp;
  prec = r->_mp_prec + negate;
  rp = r->_mp_d;
  if (negate)
    {
      usize = vsize;
      vsize = 1;
      up = v->_mp_d;
      vp = &ulimb;
      uexp = v->_mp_exp;
      ediff = uexp - 1;

      /* If U extends beyond PREC, ignore the part that does.  */
      if (usize > prec)
	{
	  up += usize - prec;
	  usize = prec;
	}
      ASSERT (ediff > 0);
    }
  else
    {
      vp = v->_mp_d;
      ediff = 1 - v->_mp_exp;
  /* Ignore leading limbs in U and V that are equal.  Doing
     this helps increase the precision of the result.  */
      if (ediff == 0 && ulimb == vp[vsize - 1])
	{
	  usize = 0;
	  vsize--;
	  uexp = 0;
	  /* Note that V might now have leading zero limbs.
	     In that case we have to adjust uexp.  */
	  for (;;)
	    {
	      if (vsize == 0) {
		rsize = 0;
		uexp = 0;
		goto done;
	      }
	      if ( vp[vsize - 1] != 0)
		break;
	      vsize--, uexp--;
	    }
	}
      else
	{
	  usize = 1;
	  uexp = 1;
	  up = &ulimb;
	}
      ASSERT (usize <= prec);
    }

  if (ediff >= prec)
    {
      /* V completely cancelled.  */
      if (rp != up)
	MPN_COPY (rp, up, usize);
      rsize = usize;
    }
  else
    {
  /* If V extends beyond PREC, ignore the part that does.
     Note that this can make vsize neither zero nor negative.  */
  if (vsize + ediff > prec)
    {
      vp += vsize + ediff - prec;
      vsize = prec - ediff;
    }

      /* Locate the least significant non-zero limb in (the needed
	 parts of) U and V, to simplify the code below.  */
      ASSERT (vsize > 0);
      for (;;)
	{
	  if (vp[0] != 0)
	    break;
	  vp++, vsize--;
	  if (vsize == 0)
	    {
	      MPN_COPY (rp, up, usize);
	      rsize = usize;
	      goto done;
	    }
	}
      for (;;)
	{
	  if (usize == 0)
	    {
	      MPN_COPY (rp, vp, vsize);
	      rsize = vsize;
	      negate ^= 1;
	      goto done;
	    }
	  if (up[0] != 0)
	    break;
	  up++, usize--;
	}

      ASSERT (usize > 0 && vsize > 0);
      TMP_MARK;

      tp = TMP_ALLOC_LIMBS (prec);

      /* uuuu     |  uuuu     |  uuuu     |  uuuu     |  uuuu    */
      /* vvvvvvv  |  vv       |    vvvvv  |    v      |       vv */

      if (usize > ediff)
	{
	  /* U and V partially overlaps.  */
	  if (ediff == 0)
	    {
	      ASSERT (usize == 1 && vsize >= 1 && ulimb == *up); /* usize is 1>ediff, vsize >= 1 */
	      if (1 < vsize)
		{
		  /* u        */
		  /* vvvvvvv  */
		  rsize = vsize;
		  vsize -= 1;
		  /* mpn_cmp (up, vp + vsize - usize, usize) > 0 */
		  if (ulimb > vp[vsize])
		    {
		      tp[vsize] = ulimb - vp[vsize] - 1;
		      ASSERT_CARRY (mpn_neg (tp, vp, vsize));
		    }
		  else
		    {
		      /* vvvvvvv  */  /* Swap U and V. */
		      /* u        */
		      MPN_COPY (tp, vp, vsize);
		      tp[vsize] = vp[vsize] - ulimb;
		      negate = 1;
		    }
		}
	      else /* vsize == usize == 1 */
		{
		  /* u     */
		  /* v     */
		  rsize = 1;
		  negate = ulimb < vp[0];
		  tp[0] = negate ? vp[0] - ulimb: ulimb - vp[0];
		}
	    }
	  else
	    {
	      ASSERT (vsize + ediff <= usize);
	      ASSERT (vsize == 1 && usize >= 2 && ulimb == *vp);
		{
		  /* uuuu     */
		  /*   v      */
		  mp_size_t size;
		  size = usize - ediff - 1;
		  MPN_COPY (tp, up, size);
		  ASSERT_NOCARRY (mpn_sub_1 (tp + size, up + size, usize - size, ulimb));
		  rsize = usize;
		}
		/* Other cases are not possible */
		/* uuuu     */
		/*   vvvvv  */
	    }
	}
      else
	{
	  /* uuuu     */
	  /*      vv  */
	  mp_size_t size, i;
	  ASSERT_CARRY (mpn_neg (tp, vp, vsize));
	  rsize = vsize + ediff;
	  size = rsize - usize;
	  for (i = vsize; i < size; i++)
	    tp[i] = GMP_NUMB_MAX;
	  ASSERT_NOCARRY (mpn_sub_1 (tp + size, up, usize, CNST_LIMB (1)));
	}

      /* Full normalize.  Optimize later.  */
      while (rsize != 0 && tp[rsize - 1] == 0)
	{
	  rsize--;
	  uexp--;
	}
      MPN_COPY (rp, tp, rsize);
      TMP_FREE;
    }

 done:
  r->_mp_size = negate ? -rsize : rsize;
  r->_mp_exp = uexp;
#endif
}
