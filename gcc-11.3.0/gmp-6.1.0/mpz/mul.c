/* mpz_mul -- Multiply two integers.

Copyright 1991, 1993, 1994, 1996, 2000, 2001, 2005, 2009, 2011, 2012 Free
Software Foundation, Inc.

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

#include <stdio.h> /* for NULL */
#include "gmp.h"
#include "gmp-impl.h"


void
mpz_mul (mpz_ptr w, mpz_srcptr u, mpz_srcptr v)
{
  mp_size_t usize;
  mp_size_t vsize;
  mp_size_t wsize;
  mp_size_t sign_product;
  mp_ptr up, vp;
  mp_ptr wp;
  mp_ptr free_me;
  size_t free_me_size;
  mp_limb_t cy_limb;
  TMP_DECL;

  usize = SIZ (u);
  vsize = SIZ (v);
  sign_product = usize ^ vsize;
  usize = ABS (usize);
  vsize = ABS (vsize);

  if (usize < vsize)
    {
      MPZ_SRCPTR_SWAP (u, v);
      MP_SIZE_T_SWAP (usize, vsize);
    }

  if (vsize == 0)
    {
      SIZ (w) = 0;
      return;
    }

#if HAVE_NATIVE_mpn_mul_2
  if (vsize <= 2)
    {
      wp = MPZ_REALLOC (w, usize+vsize);
      if (vsize == 1)
	cy_limb = mpn_mul_1 (wp, PTR (u), usize, PTR (v)[0]);
      else
	{
	  cy_limb = mpn_mul_2 (wp, PTR (u), usize, PTR (v));
	  usize++;
	}
      wp[usize] = cy_limb;
      usize += (cy_limb != 0);
      SIZ (w) = (sign_product >= 0 ? usize : -usize);
      return;
    }
#else
  if (vsize == 1)
    {
      wp = MPZ_REALLOC (w, usize+1);
      cy_limb = mpn_mul_1 (wp, PTR (u), usize, PTR (v)[0]);
      wp[usize] = cy_limb;
      usize += (cy_limb != 0);
      SIZ (w) = (sign_product >= 0 ? usize : -usize);
      return;
    }
#endif

  TMP_MARK;
  free_me = NULL;
  up = PTR (u);
  vp = PTR (v);
  wp = PTR (w);

  /* Ensure W has space enough to store the result.  */
  wsize = usize + vsize;
  if (ALLOC (w) < wsize)
    {
      if (wp == up || wp == vp)
	{
	  free_me = wp;
	  free_me_size = ALLOC (w);
	}
      else
	(*__gmp_free_func) (wp, (size_t) ALLOC (w) * GMP_LIMB_BYTES);

      ALLOC (w) = wsize;
      wp = __GMP_ALLOCATE_FUNC_LIMBS (wsize);
      PTR (w) = wp;
    }
  else
    {
      /* Make U and V not overlap with W.  */
      if (wp == up)
	{
	  /* W and U are identical.  Allocate temporary space for U.  */
	  up = TMP_ALLOC_LIMBS (usize);
	  /* Is V identical too?  Keep it identical with U.  */
	  if (wp == vp)
	    vp = up;
	  /* Copy to the temporary space.  */
	  MPN_COPY (up, wp, usize);
	}
      else if (wp == vp)
	{
	  /* W and V are identical.  Allocate temporary space for V.  */
	  vp = TMP_ALLOC_LIMBS (vsize);
	  /* Copy to the temporary space.  */
	  MPN_COPY (vp, wp, vsize);
	}
    }

  if (up == vp)
    {
      mpn_sqr (wp, up, usize);
      cy_limb = wp[wsize - 1];
    }
  else
    {
      cy_limb = mpn_mul (wp, up, usize, vp, vsize);
    }

  wsize -= cy_limb == 0;

  SIZ (w) = sign_product < 0 ? -wsize : wsize;
  if (free_me != NULL)
    (*__gmp_free_func) (free_me, free_me_size * GMP_LIMB_BYTES);
  TMP_FREE;
}
