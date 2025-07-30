/* UltraSPARC 64 mpn_mod_1 -- mpn by limb remainder.

Copyright 1991, 1993, 1994, 1999-2001, 2003, 2010 Free Software Foundation,
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


/*                 64-bit divisor   32-bit divisor
                    cycles/limb      cycles/limb
                     (approx)         (approx)
   Ultrasparc 2i:      160               120
*/


/* 32-bit divisors are treated in special case code.  This requires 4 mulx
   per limb instead of 8 in the general case.

   For big endian systems we need HALF_ENDIAN_ADJ included in the src[i]
   addressing, to get the two halves of each limb read in the correct order.
   This is kept in an adj variable.  Doing that measures about 6 c/l faster
   than just writing HALF_ENDIAN_ADJ(i) in the loop.  The latter shouldn't
   be 6 cycles worth of work, but perhaps it doesn't schedule well (on gcc
   3.2.1 at least).

   A simple udivx/umulx loop for the 32-bit case was attempted for small
   sizes, but at size==2 it was only about the same speed and at size==3 was
   slower.  */

static mp_limb_t
mpn_mod_1_anynorm (mp_srcptr src_limbptr, mp_size_t size_limbs, mp_limb_t d_limb)
{
  int        norm, norm_rshift;
  mp_limb_t  src_high_limb;
  mp_size_t  i;

  ASSERT (size_limbs >= 0);
  ASSERT (d_limb != 0);

  if (UNLIKELY (size_limbs == 0))
    return 0;

  src_high_limb = src_limbptr[size_limbs-1];

  /* udivx is good for size==1, and no need to bother checking limb<divisor,
     since if that's likely the caller should check */
  if (UNLIKELY (size_limbs == 1))
    return src_high_limb % d_limb;

  if (d_limb <= CNST_LIMB(0xFFFFFFFF))
    {
      unsigned   *src, n1, n0, r, dummy_q, nshift, norm_rmask;
      mp_size_t  size, adj;
      mp_limb_t  dinv_limb;

      size = 2 * size_limbs;    /* halfwords */
      src = (unsigned *) src_limbptr;

      /* prospective initial remainder, if < d */
      r = src_high_limb >> 32;

      /* If the length of the source is uniformly distributed, then there's
         a 50% chance of the high 32-bits being zero, which we can skip.  */
      if (r == 0)
        {
          r = (unsigned) src_high_limb;
          size--;
          ASSERT (size > 0);  /* because always even */
        }

      /* Skip a division if high < divisor.  Having the test here before
         normalizing will still skip as often as possible.  */
      if (r < d_limb)
        {
          size--;
          ASSERT (size > 0);  /* because size==1 handled above */
        }
      else
        r = 0;

      count_leading_zeros_32 (norm, d_limb);
      norm -= 32;
      d_limb <<= norm;

      norm_rshift = 32 - norm;
      norm_rmask = (norm == 0 ? 0 : 0xFFFFFFFF);
      i = size-1;
      adj = HALF_ENDIAN_ADJ (i);
      n1 = src [i + adj];
      r = (r << norm) | ((n1 >> norm_rshift) & norm_rmask);

      invert_half_limb (dinv_limb, d_limb);
      adj = -adj;

      for (i--; i >= 0; i--)
        {
          n0 = src [i + adj];
          adj = -adj;
          nshift = (n1 << norm) | ((n0 >> norm_rshift) & norm_rmask);
          udiv_qrnnd_half_preinv (dummy_q, r, r, nshift, d_limb, dinv_limb);
          n1 = n0;
        }

      /* same as loop, but without n0 */
      nshift = n1 << norm;
      udiv_qrnnd_half_preinv (dummy_q, r, r, nshift, d_limb, dinv_limb);

      ASSERT ((r & ((1 << norm) - 1)) == 0);
      return r >> norm;
    }
  else
    {
      mp_srcptr  src;
      mp_size_t  size;
      mp_limb_t  n1, n0, r, dinv, dummy_q, nshift, norm_rmask;

      src = src_limbptr;
      size = size_limbs;
      r = src_high_limb;  /* initial remainder */

      /* Skip a division if high < divisor.  Having the test here before
         normalizing will still skip as often as possible.  */
      if (r < d_limb)
        {
          size--;
          ASSERT (size > 0);  /* because size==1 handled above */
        }
      else
        r = 0;

      count_leading_zeros (norm, d_limb);
      d_limb <<= norm;

      norm_rshift = GMP_LIMB_BITS - norm;
      norm_rmask = (norm == 0 ? 0 : 0xFFFFFFFF);

      src += size;
      n1 = *--src;
      r = (r << norm) | ((n1 >> norm_rshift) & norm_rmask);

      invert_limb (dinv, d_limb);

      for (i = size-2; i >= 0; i--)
        {
          n0 = *--src;
          nshift = (n1 << norm) | ((n0 >> norm_rshift) & norm_rmask);
          udiv_qrnnd_preinv (dummy_q, r, r, nshift, d_limb, dinv);
          n1 = n0;
        }

      /* same as loop, but without n0 */
      nshift = n1 << norm;
      udiv_qrnnd_preinv (dummy_q, r, r, nshift, d_limb, dinv);

      ASSERT ((r & ((CNST_LIMB(1) << norm) - 1)) == 0);
      return r >> norm;
    }
}

mp_limb_t
mpn_mod_1 (mp_srcptr ap, mp_size_t n, mp_limb_t b)
{
  ASSERT (n >= 0);
  ASSERT (b != 0);

  /* Should this be handled at all?  Rely on callers?  Note un==0 is currently
     required by mpz/fdiv_r_ui.c and possibly other places.  */
  if (n == 0)
    return 0;

  if (UNLIKELY ((b & GMP_NUMB_HIGHBIT) != 0))
    {
      if (BELOW_THRESHOLD (n, MOD_1N_TO_MOD_1_1_THRESHOLD))
	{
	  return mpn_mod_1_anynorm (ap, n, b);
	}
      else
	{
	  mp_limb_t pre[4];
	  mpn_mod_1_1p_cps (pre, b);
	  return mpn_mod_1_1p (ap, n, b, pre);
	}
    }
  else
    {
      if (BELOW_THRESHOLD (n, MOD_1U_TO_MOD_1_1_THRESHOLD))
	{
	  return mpn_mod_1_anynorm (ap, n, b);
	}
      else if (BELOW_THRESHOLD (n, MOD_1_1_TO_MOD_1_2_THRESHOLD))
	{
	  mp_limb_t pre[4];
	  mpn_mod_1_1p_cps (pre, b);
	  return mpn_mod_1_1p (ap, n, b << pre[1], pre);
	}
      else if (BELOW_THRESHOLD (n, MOD_1_2_TO_MOD_1_4_THRESHOLD) || UNLIKELY (b > GMP_NUMB_MASK / 4))
	{
	  mp_limb_t pre[5];
	  mpn_mod_1s_2p_cps (pre, b);
	  return mpn_mod_1s_2p (ap, n, b << pre[1], pre);
	}
      else
	{
	  mp_limb_t pre[7];
	  mpn_mod_1s_4p_cps (pre, b);
	  return mpn_mod_1s_4p (ap, n, b << pre[1], pre);
	}
    }
}
