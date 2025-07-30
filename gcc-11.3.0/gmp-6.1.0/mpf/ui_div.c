/* mpf_ui_div -- Divide an unsigned integer with a float.

Copyright 1993-1996, 2000-2002, 2004, 2005, 2012 Free Software Foundation, Inc.

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

#include <stdio.h>  /* for NULL */
#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"


void
mpf_ui_div (mpf_ptr r, unsigned long int u, mpf_srcptr v)
{
  mp_srcptr vp;
  mp_ptr rp, tp, remp, new_vp;
  mp_size_t vsize;
  mp_size_t rsize, prospective_rsize, zeros, tsize, high_zero;
  mp_size_t sign_quotient;
  mp_size_t prec;
  mp_exp_t rexp;
  TMP_DECL;

  vsize = v->_mp_size;
  sign_quotient = vsize;

  if (UNLIKELY (vsize == 0))
    DIVIDE_BY_ZERO;

  if (UNLIKELY (u == 0))
    {
      r->_mp_size = 0;
      r->_mp_exp = 0;
      return;
    }

  vsize = ABS (vsize);
  prec = r->_mp_prec;

  TMP_MARK;
  rexp = 1 - v->_mp_exp + 1;

  rp = r->_mp_d;
  vp = v->_mp_d;

  prospective_rsize = 1 - vsize + 1;    /* quot from using given u,v sizes */
  rsize = prec + 1;                     /* desired quot size */

  zeros = rsize - prospective_rsize;    /* padding u to give rsize */
  tsize = 1 + zeros;                    /* u with zeros */

  if (WANT_TMP_DEBUG)
    {
      /* separate alloc blocks, for malloc debugging */
      remp = TMP_ALLOC_LIMBS (vsize);
      tp = TMP_ALLOC_LIMBS (tsize);
      new_vp = NULL;
      if (rp == vp)
        new_vp = TMP_ALLOC_LIMBS (vsize);
    }
  else
    {
      /* one alloc with calculated size, for efficiency */
      mp_size_t size = vsize + tsize + (rp == vp ? vsize : 0);
      remp = TMP_ALLOC_LIMBS (size);
      tp = remp + vsize;
      new_vp = tp + tsize;
    }

  /* ensure divisor doesn't overlap quotient */
  if (rp == vp)
    {
      MPN_COPY (new_vp, vp, vsize);
      vp = new_vp;
    }

  MPN_ZERO (tp, tsize-1);

  tp[tsize - 1] = u & GMP_NUMB_MASK;
#if BITS_PER_ULONG > GMP_NUMB_BITS
  if (u > GMP_NUMB_MAX)
    {
      /* tsize-vsize+1 == rsize, so tsize >= rsize.  rsize == prec+1 >= 2,
         so tsize >= 2, hence there's room for 2-limb u with nails */
      ASSERT (tsize >= 2);
      tp[tsize - 1] = u >> GMP_NUMB_BITS;
      tp[tsize - 2] = u & GMP_NUMB_MASK;
      rexp++;
    }
#endif

  ASSERT (tsize-vsize+1 == rsize);
  mpn_tdiv_qr (rp, remp, (mp_size_t) 0, tp, tsize, vp, vsize);

  /* strip possible zero high limb */
  high_zero = (rp[rsize-1] == 0);
  rsize -= high_zero;
  rexp -= high_zero;

  r->_mp_size = sign_quotient >= 0 ? rsize : -rsize;
  r->_mp_exp = rexp;
  TMP_FREE;
}
