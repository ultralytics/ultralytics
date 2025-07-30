/* mpz_mul_ui/si (product, multiplier, small_multiplicand) -- Set PRODUCT to
   MULTIPLICATOR times SMALL_MULTIPLICAND.

Copyright 1991, 1993, 1994, 1996, 2000-2002, 2005, 2008, 2012 Free Software
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


#ifdef OPERATION_mul_si
#define FUNCTION               mpz_mul_si
#define MULTIPLICAND_UNSIGNED
#define MULTIPLICAND_ABS(x)    ABS_CAST(unsigned long, (x))
#endif

#ifdef OPERATION_mul_ui
#define FUNCTION               mpz_mul_ui
#define MULTIPLICAND_UNSIGNED  unsigned
#define MULTIPLICAND_ABS(x)    x
#endif

#ifndef FUNCTION
Error, error, unrecognised OPERATION
#endif


void
FUNCTION (mpz_ptr prod, mpz_srcptr mult,
          MULTIPLICAND_UNSIGNED long int small_mult)
{
  mp_size_t size;
  mp_size_t sign_product;
  mp_limb_t sml;
  mp_limb_t cy;
  mp_ptr pp;

  sign_product = SIZ(mult);
  if (sign_product == 0 || small_mult == 0)
    {
      SIZ(prod) = 0;
      return;
    }

  size = ABS (sign_product);

  sml = MULTIPLICAND_ABS (small_mult);

  if (sml <= GMP_NUMB_MAX)
    {
      pp = MPZ_REALLOC (prod, size + 1);
      cy = mpn_mul_1 (pp, PTR(mult), size, sml);
      pp[size] = cy;
      size += cy != 0;
    }
#if GMP_NAIL_BITS != 0
  else
    {
      /* Operand too large for the current nails size.  Use temporary for
	 intermediate products, to allow prod and mult being identical.  */
      mp_ptr tp;
      TMP_DECL;
      TMP_MARK;

      tp = TMP_ALLOC_LIMBS (size + 2);

      /* Use, maybe, mpn_mul_2? */
      cy = mpn_mul_1 (tp, PTR(mult), size, sml & GMP_NUMB_MASK);
      tp[size] = cy;
      cy = mpn_addmul_1 (tp + 1, PTR(mult), size, sml >> GMP_NUMB_BITS);
      tp[size + 1] = cy;
      size += 2;
      MPN_NORMALIZE_NOT_ZERO (tp, size); /* too general, need to trim one or two limb */
      pp = MPZ_REALLOC (prod, size);
      MPN_COPY (pp, tp, size);
      TMP_FREE;
    }
#endif

  SIZ(prod) = ((sign_product < 0) ^ (small_mult < 0)) ? -size : size;
}
