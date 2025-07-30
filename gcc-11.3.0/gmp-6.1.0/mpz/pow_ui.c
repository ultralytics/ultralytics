/* mpz_pow_ui -- mpz raised to ulong.

Copyright 2001, 2008 Free Software Foundation, Inc.

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
mpz_pow_ui (mpz_ptr r, mpz_srcptr b, unsigned long int e)
{
  /* We test some small exponents here, mainly to avoid the overhead of
     mpz_n_pow_ui for small bases and exponents.  */
  switch (e)
    {
    case 0:
      mpz_set_ui (r, 1);
      break;
    case 1:
      mpz_set (r, b);
      break;
    case 2:
      mpz_mul (r, b, b);
      break;
    default:
      mpz_n_pow_ui (r, PTR(b), (mp_size_t) SIZ(b), e);
    }
}
