/* mpn_preinv_divrem_1 -- mpn by limb division with pre-inverted divisor.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2000-2003 Free Software Foundation, Inc.

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


/* Don't bloat a shared library with unused code. */
#if USE_PREINV_DIVREM_1

/* Same test here for skipping one divide step as in mpn_divrem_1.

   The main reason for a separate shift==0 case is that not all CPUs give
   zero for "n0 >> GMP_LIMB_BITS" which would arise in the general case
   code used on shift==0.  shift==0 is also reasonably common in mp_bases
   big_base, for instance base==10 on a 64-bit limb.

   Under shift!=0 it would be possible to call mpn_lshift to adjust the
   dividend all in one go (into the quotient space say), rather than
   limb-by-limb in the loop.  This might help if mpn_lshift is a lot faster
   than what the compiler can generate for EXTRACT.  But this is left to CPU
   specific implementations to consider, especially since EXTRACT isn't on
   the dependent chain.

   If size==0 then the result is simply xsize limbs of zeros, but nothing
   special is done for that, since it wouldn't be a usual call, and
   certainly never arises from mpn_get_str which is our main caller.  */

mp_limb_t
mpn_preinv_divrem_1 (mp_ptr qp, mp_size_t xsize,
		     mp_srcptr ap, mp_size_t size, mp_limb_t d_unnorm,
		     mp_limb_t dinv, int shift)
{
  mp_limb_t  ahigh, qhigh, r;
  mp_size_t  i;
  mp_limb_t  n1, n0;
  mp_limb_t  d;

  ASSERT (xsize >= 0);
  ASSERT (size >= 1);
  ASSERT (d_unnorm != 0);
#if WANT_ASSERT
  {
    int        want_shift;
    mp_limb_t  want_dinv;
    count_leading_zeros (want_shift, d_unnorm);
    ASSERT (shift == want_shift);
    invert_limb (want_dinv, d_unnorm << shift);
    ASSERT (dinv == want_dinv);
  }
#endif
  /* FIXME: What's the correct overlap rule when xsize!=0? */
  ASSERT (MPN_SAME_OR_SEPARATE_P (qp+xsize, ap, size));

  ahigh = ap[size-1];
  d = d_unnorm << shift;
  qp += (size + xsize - 1);   /* dest high limb */

  if (shift == 0)
    {
      /* High quotient limb is 0 or 1, and skip a divide step. */
      r = ahigh;
      qhigh = (r >= d);
      r = (qhigh ? r-d : r);
      *qp-- = qhigh;
      size--;

      for (i = size-1; i >= 0; i--)
	{
	  n0 = ap[i];
	  udiv_qrnnd_preinv (*qp, r, r, n0, d, dinv);
	  qp--;
	}
    }
  else
    {
      r = 0;
      if (ahigh < d_unnorm)
	{
	  r = ahigh << shift;
	  *qp-- = 0;
	  size--;
	  if (size == 0)
	    goto done_integer;
	}

      n1 = ap[size-1];
      r |= n1 >> (GMP_LIMB_BITS - shift);

      for (i = size-2; i >= 0; i--)
	{
	  ASSERT (r < d);
	  n0 = ap[i];
	  udiv_qrnnd_preinv (*qp, r, r,
			     ((n1 << shift) | (n0 >> (GMP_LIMB_BITS - shift))),
			     d, dinv);
	  qp--;
	  n1 = n0;
	}
      udiv_qrnnd_preinv (*qp, r, r, n1 << shift, d, dinv);
      qp--;
    }

 done_integer:
  for (i = 0; i < xsize; i++)
    {
      udiv_qrnnd_preinv (*qp, r, r, CNST_LIMB(0), d, dinv);
      qp--;
    }

  return r >> shift;
}

#endif /* USE_PREINV_DIVREM_1 */
