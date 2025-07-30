/* mpf_add_ui -- Add a float and an unsigned integer.

Copyright 1993, 1994, 1996, 2000, 2001 Free Software Foundation, Inc.

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
mpf_add_ui (mpf_ptr sum, mpf_srcptr u, unsigned long int v)
{
  mp_srcptr up = u->_mp_d;
  mp_ptr sump = sum->_mp_d;
  mp_size_t usize, sumsize;
  mp_size_t prec = sum->_mp_prec;
  mp_exp_t uexp = u->_mp_exp;

  usize = u->_mp_size;
  if (usize <= 0)
    {
      if (usize == 0)
	{
	  mpf_set_ui (sum, v);
	  return;
	}
      else
	{
	  __mpf_struct u_negated;
	  u_negated._mp_size = -usize;
	  u_negated._mp_exp = u->_mp_exp;
	  u_negated._mp_d = u->_mp_d;
	  mpf_sub_ui (sum, &u_negated, v);
	  sum->_mp_size = -(sum->_mp_size);
	  return;
	}
    }

  if (v == 0)
    {
    sum_is_u:
      if (u != sum)
	{
	  sumsize = MIN (usize, prec + 1);
	  MPN_COPY (sum->_mp_d, up + usize - sumsize, sumsize);
	  sum->_mp_size = sumsize;
	  sum->_mp_exp = u->_mp_exp;
	}
      return;
    }

  if (uexp > 0)
    {
      /* U >= 1.  */
      if (uexp > prec)
	{
	  /* U >> V, V is not part of final result.  */
	  goto sum_is_u;
	}
      else
	{
	  /* U's "limb point" is somewhere between the first limb
	     and the PREC:th limb.
	     Both U and V are part of the final result.  */
	  if (uexp > usize)
	    {
	      /*   uuuuuu0000. */
	      /* +          v. */
	      /* We begin with moving U to the top of SUM, to handle
		 samevar(U,SUM).  */
	      MPN_COPY_DECR (sump + uexp - usize, up, usize);
	      sump[0] = v;
	      MPN_ZERO (sump + 1, uexp - usize - 1);
#if 0 /* What is this??? */
	      if (sum == u)
		MPN_COPY (sum->_mp_d, sump, uexp);
#endif
	      sum->_mp_size = uexp;
	      sum->_mp_exp = uexp;
	    }
	  else
	    {
	      /*   uuuuuu.uuuu */
	      /* +      v.     */
	      mp_limb_t cy_limb;
	      if (usize > prec)
		{
		  /* Ignore excess limbs in U.  */
		  up += usize - prec;
		  usize -= usize - prec; /* Eq. usize = prec */
		}
	      if (sump != up)
		MPN_COPY_INCR (sump, up, usize - uexp);
	      cy_limb = mpn_add_1 (sump + usize - uexp, up + usize - uexp,
				   uexp, (mp_limb_t) v);
	      sump[usize] = cy_limb;
	      sum->_mp_size = usize + cy_limb;
	      sum->_mp_exp = uexp + cy_limb;
	    }
	}
    }
  else
    {
      /* U < 1, so V > U for sure.  */
      /* v.         */
      /*  .0000uuuu */
      if ((-uexp) >= prec)
	{
	  sump[0] = v;
	  sum->_mp_size = 1;
	  sum->_mp_exp = 1;
	}
      else
	{
	  if (usize + (-uexp) + 1 > prec)
	    {
	      /* Ignore excess limbs in U.  */
	      up += usize + (-uexp) + 1 - prec;
	      usize -= usize + (-uexp) + 1 - prec;
	    }
	  if (sump != up)
	    MPN_COPY_INCR (sump, up, usize);
	  MPN_ZERO (sump + usize, -uexp);
	  sump[usize + (-uexp)] = v;
	  sum->_mp_size = usize + (-uexp) + 1;
	  sum->_mp_exp = 1;
	}
    }
}
