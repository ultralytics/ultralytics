/* mpz_cdiv_qr -- Division rounding the quotient towards +infinity.  The
   remainder gets the opposite sign as the denominator.

Copyright 1994-1996, 2000, 2001, 2005, 2012 Free Software Foundation, Inc.

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
mpz_cdiv_qr (mpz_ptr quot, mpz_ptr rem, mpz_srcptr dividend, mpz_srcptr divisor)
{
  mp_size_t divisor_size = SIZ (divisor);
  mp_size_t xsize;
  mpz_t temp_divisor;		/* N.B.: lives until function returns! */
  TMP_DECL;

  TMP_MARK;

  /* We need the original value of the divisor after the quotient and
     remainder have been preliminary calculated.  We have to copy it to
     temporary space if it's the same variable as either QUOT or REM.  */
  if (quot == divisor || rem == divisor)
    {
      MPZ_TMP_INIT (temp_divisor, ABS (divisor_size));
      mpz_set (temp_divisor, divisor);
      divisor = temp_divisor;
    }

  xsize = SIZ (dividend) ^ divisor_size;;
  mpz_tdiv_qr (quot, rem, dividend, divisor);

  if (xsize >= 0 && SIZ (rem) != 0)
    {
      mpz_add_ui (quot, quot, 1L);
      mpz_sub (rem, rem, divisor);
    }

  TMP_FREE;
}
