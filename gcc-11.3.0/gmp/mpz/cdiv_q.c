/* mpz_cdiv_q -- Division rounding the quotient towards +infinity.  The
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
mpz_cdiv_q (mpz_ptr quot, mpz_srcptr dividend, mpz_srcptr divisor)
{
  mp_size_t dividend_size = SIZ (dividend);
  mp_size_t divisor_size = SIZ (divisor);
  mpz_t rem;
  TMP_DECL;

  TMP_MARK;

  MPZ_TMP_INIT (rem, ABS (divisor_size));

  mpz_tdiv_qr (quot, rem, dividend, divisor);

  if ((divisor_size ^ dividend_size) >= 0 && SIZ (rem) != 0)
    mpz_add_ui (quot, quot, 1L);

  TMP_FREE;
}
