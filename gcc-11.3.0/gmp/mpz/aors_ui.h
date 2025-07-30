/* mpz_add_ui, mpz_sub_ui -- Add or subtract an mpz_t and an unsigned
   one-word integer.

Copyright 1991, 1993, 1994, 1996, 1999-2002, 2004, 2012, 2013 Free Software
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


#ifdef OPERATION_add_ui
#define FUNCTION          mpz_add_ui
#define FUNCTION2         mpz_add
#define VARIATION_CMP     >=
#define VARIATION_NEG
#define VARIATION_UNNEG   -
#endif

#ifdef OPERATION_sub_ui
#define FUNCTION          mpz_sub_ui
#define FUNCTION2         mpz_sub
#define VARIATION_CMP     <
#define VARIATION_NEG     -
#define VARIATION_UNNEG
#endif

#ifndef FUNCTION
Error, need OPERATION_add_ui or OPERATION_sub_ui
#endif


void
FUNCTION (mpz_ptr w, mpz_srcptr u, unsigned long int vval)
{
  mp_srcptr up;
  mp_ptr wp;
  mp_size_t usize, wsize;
  mp_size_t abs_usize;

#if BITS_PER_ULONG > GMP_NUMB_BITS  /* avoid warnings about shift amount */
  if (vval > GMP_NUMB_MAX)
    {
      mpz_t v;
      mp_limb_t vl[2];
      PTR(v) = vl;
      vl[0] = vval & GMP_NUMB_MASK;
      vl[1] = vval >> GMP_NUMB_BITS;
      SIZ(v) = 2;
      FUNCTION2 (w, u, v);
      return;
    }
#endif

  usize = SIZ (u);
  if (usize == 0)
    {
      PTR (w)[0] = vval;
      SIZ (w) = VARIATION_NEG (vval != 0);
      return;
    }

  abs_usize = ABS (usize);

  /* If not space for W (and possible carry), increase space.  */
  wp = MPZ_REALLOC (w, abs_usize + 1);

  /* These must be after realloc (U may be the same as W).  */
  up = PTR (u);

  if (usize VARIATION_CMP 0)
    {
      mp_limb_t cy;
      cy = mpn_add_1 (wp, up, abs_usize, (mp_limb_t) vval);
      wp[abs_usize] = cy;
      wsize = VARIATION_NEG (abs_usize + cy);
    }
  else
    {
      /* The signs are different.  Need exact comparison to determine
	 which operand to subtract from which.  */
      if (abs_usize == 1 && up[0] < vval)
	{
	  wp[0] = vval - up[0];
	  wsize = VARIATION_NEG 1;
	}
      else
	{
	  mpn_sub_1 (wp, up, abs_usize, (mp_limb_t) vval);
	  /* Size can decrease with at most one limb.  */
	  wsize = VARIATION_UNNEG (abs_usize - (wp[abs_usize - 1] == 0));
	}
    }

  SIZ (w) = wsize;
}
