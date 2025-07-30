/* UltraSparc 64 mpn_divrem_1 -- mpn by limb division.

Copyright 1991, 1993, 1994, 1996, 1998-2001, 2003 Free Software Foundation,
Inc.

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

#include "mpn/sparc64/sparc64.h"


/*                   64-bit divisor       32-bit divisor
                       cycles/limb          cycles/limb
                        (approx)             (approx)
                   integer  fraction    integer  fraction
   Ultrasparc 2i:    160      160          122      96
*/


/* 32-bit divisors are treated in special case code.  This requires 4 mulx
   per limb instead of 8 in the general case.

   For big endian systems we need HALF_ENDIAN_ADJ included in the src[i]
   addressing, to get the two halves of each limb read in the correct order.
   This is kept in an adj variable.  Doing that measures about 4 c/l faster
   than just writing HALF_ENDIAN_ADJ(i) in the integer loop.  The latter
   shouldn't be 6 cycles worth of work, but perhaps it doesn't schedule well
   (on gcc 3.2.1 at least).  The fraction loop doesn't seem affected, but we
   still use a variable since that ought to work out best.  */

mp_limb_t
mpn_divrem_1 (mp_ptr qp_limbptr, mp_size_t xsize_limbs,
              mp_srcptr ap_limbptr, mp_size_t size_limbs, mp_limb_t d_limb)
{
  mp_size_t  total_size_limbs;
  mp_size_t  i;

  ASSERT (xsize_limbs >= 0);
  ASSERT (size_limbs >= 0);
  ASSERT (d_limb != 0);
  /* FIXME: What's the correct overlap rule when xsize!=0? */
  ASSERT (MPN_SAME_OR_SEPARATE_P (qp_limbptr + xsize_limbs,
                                  ap_limbptr, size_limbs));

  total_size_limbs = size_limbs + xsize_limbs;
  if (UNLIKELY (total_size_limbs == 0))
    return 0;

  /* udivx is good for total_size==1, and no need to bother checking
     limb<divisor, since if that's likely the caller should check */
  if (UNLIKELY (total_size_limbs == 1))
    {
      mp_limb_t  a, q;
      a = (LIKELY (size_limbs != 0) ? ap_limbptr[0] : 0);
      q = a / d_limb;
      qp_limbptr[0] = q;
      return a - q*d_limb;
    }

  if (d_limb <= CNST_LIMB(0xFFFFFFFF))
    {
      mp_size_t  size, xsize, total_size, adj;
      unsigned   *qp, n1, n0, q, r, nshift, norm_rmask;
      mp_limb_t  dinv_limb;
      const unsigned *ap;
      int        norm, norm_rshift;

      size = 2 * size_limbs;
      xsize = 2 * xsize_limbs;
      total_size = size + xsize;

      ap = (unsigned *) ap_limbptr;
      qp = (unsigned *) qp_limbptr;

      qp += xsize;
      r = 0;        /* initial remainder */

      if (LIKELY (size != 0))
        {
          n1 = ap[size-1 + HALF_ENDIAN_ADJ(1)];

          /* If the length of the source is uniformly distributed, then
             there's a 50% chance of the high 32-bits being zero, which we
             can skip.  */
          if (n1 == 0)
            {
              n1 = ap[size-2 + HALF_ENDIAN_ADJ(0)];
              total_size--;
              size--;
              ASSERT (size > 0);  /* because always even */
              qp[size + HALF_ENDIAN_ADJ(1)] = 0;
            }

          /* Skip a division if high < divisor (high quotient 0).  Testing
             here before before normalizing will still skip as often as
             possible.  */
          if (n1 < d_limb)
            {
              r = n1;
              size--;
              qp[size + HALF_ENDIAN_ADJ(size)] = 0;
              total_size--;
              if (total_size == 0)
                return r;
            }
        }

      count_leading_zeros_32 (norm, d_limb);
      norm -= 32;
      d_limb <<= norm;
      r <<= norm;

      norm_rshift = 32 - norm;
      norm_rmask = (norm == 0 ? 0 : 0xFFFFFFFF);

      invert_half_limb (dinv_limb, d_limb);

      if (LIKELY (size != 0))
        {
          i = size - 1;
          adj = HALF_ENDIAN_ADJ (i);
          n1 = ap[i + adj];
          adj = -adj;
          r |= ((n1 >> norm_rshift) & norm_rmask);
          for ( ; i > 0; i--)
            {
              n0 = ap[i-1 + adj];
              adj = -adj;
              nshift = (n1 << norm) | ((n0 >> norm_rshift) & norm_rmask);
              udiv_qrnnd_half_preinv (q, r, r, nshift, d_limb, dinv_limb);
              qp[i + adj] = q;
              n1 = n0;
            }
          nshift = n1 << norm;
          udiv_qrnnd_half_preinv (q, r, r, nshift, d_limb, dinv_limb);
          qp[0 + HALF_ENDIAN_ADJ(0)] = q;
        }
      qp -= xsize;
      adj = HALF_ENDIAN_ADJ (0);
      for (i = xsize-1; i >= 0; i--)
        {
          udiv_qrnnd_half_preinv (q, r, r, 0, d_limb, dinv_limb);
          adj = -adj;
          qp[i + adj] = q;
        }

      return r >> norm;
    }
  else
    {
      mp_srcptr  ap;
      mp_ptr     qp;
      mp_size_t  size, xsize, total_size;
      mp_limb_t  d, n1, n0, q, r, dinv, nshift, norm_rmask;
      int        norm, norm_rshift;

      ap = ap_limbptr;
      qp = qp_limbptr;
      size = size_limbs;
      xsize = xsize_limbs;
      total_size = total_size_limbs;
      d = d_limb;

      qp += total_size;   /* above high limb */
      r = 0;              /* initial remainder */

      if (LIKELY (size != 0))
        {
          /* Skip a division if high < divisor (high quotient 0).  Testing
             here before before normalizing will still skip as often as
             possible.  */
          n1 = ap[size-1];
          if (n1 < d)
            {
              r = n1;
              *--qp = 0;
              total_size--;
              if (total_size == 0)
                return r;
              size--;
            }
        }

      count_leading_zeros (norm, d);
      d <<= norm;
      r <<= norm;

      norm_rshift = GMP_LIMB_BITS - norm;
      norm_rmask = (norm == 0 ? 0 : ~CNST_LIMB(0));

      invert_limb (dinv, d);

      if (LIKELY (size != 0))
        {
          n1 = ap[size-1];
          r |= ((n1 >> norm_rshift) & norm_rmask);
          for (i = size-2; i >= 0; i--)
            {
              n0 = ap[i];
              nshift = (n1 << norm) | ((n0 >> norm_rshift) & norm_rmask);
              udiv_qrnnd_preinv (q, r, r, nshift, d, dinv);
              *--qp = q;
              n1 = n0;
            }
          nshift = n1 << norm;
          udiv_qrnnd_preinv (q, r, r, nshift, d, dinv);
          *--qp = q;
        }
      for (i = 0; i < xsize; i++)
        {
          udiv_qrnnd_preinv (q, r, r, CNST_LIMB(0), d, dinv);
          *--qp = q;
        }
      return r >> norm;
    }
}
