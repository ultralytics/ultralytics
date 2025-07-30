/* mpz/gcd.c:   Calculate the greatest common divisor of two integers.

Copyright 1991, 1993, 1994, 1996, 2000-2002, 2005, 2010 Free Software
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
#include "longlong.h"


void
mpz_gcd (mpz_ptr g, mpz_srcptr u, mpz_srcptr v)
{
  unsigned long int g_zero_bits, u_zero_bits, v_zero_bits;
  mp_size_t g_zero_limbs, u_zero_limbs, v_zero_limbs;
  mp_ptr tp;
  mp_ptr up;
  mp_size_t usize;
  mp_ptr vp;
  mp_size_t vsize;
  mp_size_t gsize;
  TMP_DECL;

  up = PTR(u);
  usize = ABSIZ (u);
  vp = PTR(v);
  vsize = ABSIZ (v);
  /* GCD(0, V) == V.  */
  if (usize == 0)
    {
      SIZ (g) = vsize;
      if (g == v)
	return;
      MPZ_REALLOC (g, vsize);
      MPN_COPY (PTR (g), vp, vsize);
      return;
    }

  /* GCD(U, 0) == U.  */
  if (vsize == 0)
    {
      SIZ (g) = usize;
      if (g == u)
	return;
      MPZ_REALLOC (g, usize);
      MPN_COPY (PTR (g), up, usize);
      return;
    }

  if (usize == 1)
    {
      SIZ (g) = 1;
      PTR (g)[0] = mpn_gcd_1 (vp, vsize, up[0]);
      return;
    }

  if (vsize == 1)
    {
      SIZ(g) = 1;
      PTR (g)[0] = mpn_gcd_1 (up, usize, vp[0]);
      return;
    }

  TMP_MARK;

  /*  Eliminate low zero bits from U and V and move to temporary storage.  */
  while (*up == 0)
    up++;
  u_zero_limbs = up - PTR(u);
  usize -= u_zero_limbs;
  count_trailing_zeros (u_zero_bits, *up);
  tp = up;
  up = TMP_ALLOC_LIMBS (usize);
  if (u_zero_bits != 0)
    {
      mpn_rshift (up, tp, usize, u_zero_bits);
      usize -= up[usize - 1] == 0;
    }
  else
    MPN_COPY (up, tp, usize);

  while (*vp == 0)
    vp++;
  v_zero_limbs = vp - PTR (v);
  vsize -= v_zero_limbs;
  count_trailing_zeros (v_zero_bits, *vp);
  tp = vp;
  vp = TMP_ALLOC_LIMBS (vsize);
  if (v_zero_bits != 0)
    {
      mpn_rshift (vp, tp, vsize, v_zero_bits);
      vsize -= vp[vsize - 1] == 0;
    }
  else
    MPN_COPY (vp, tp, vsize);

  if (u_zero_limbs > v_zero_limbs)
    {
      g_zero_limbs = v_zero_limbs;
      g_zero_bits = v_zero_bits;
    }
  else if (u_zero_limbs < v_zero_limbs)
    {
      g_zero_limbs = u_zero_limbs;
      g_zero_bits = u_zero_bits;
    }
  else  /*  Equal.  */
    {
      g_zero_limbs = u_zero_limbs;
      g_zero_bits = MIN (u_zero_bits, v_zero_bits);
    }

  /*  Call mpn_gcd.  The 2nd argument must not have more bits than the 1st.  */
  vsize = (usize < vsize || (usize == vsize && up[usize-1] < vp[vsize-1]))
    ? mpn_gcd (vp, vp, vsize, up, usize)
    : mpn_gcd (vp, up, usize, vp, vsize);

  /*  Here G <-- V << (g_zero_limbs*GMP_LIMB_BITS + g_zero_bits).  */
  gsize = vsize + g_zero_limbs;
  if (g_zero_bits != 0)
    {
      mp_limb_t cy_limb;
      gsize += (vp[vsize - 1] >> (GMP_NUMB_BITS - g_zero_bits)) != 0;
      MPZ_REALLOC (g, gsize);
      MPN_ZERO (PTR (g), g_zero_limbs);

      tp = PTR(g) + g_zero_limbs;
      cy_limb = mpn_lshift (tp, vp, vsize, g_zero_bits);
      if (cy_limb != 0)
	tp[vsize] = cy_limb;
    }
  else
    {
      MPZ_REALLOC (g, gsize);
      MPN_ZERO (PTR (g), g_zero_limbs);
      MPN_COPY (PTR (g) + g_zero_limbs, vp, vsize);
    }

  SIZ (g) = gsize;
  TMP_FREE;
}
