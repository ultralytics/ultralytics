/* mpf_random2 -- Generate a positive random mpf_t of specified size, with
   long runs of consecutive ones and zeros in the binary representation.
   Intended for testing of other MP routines.

Copyright 1995, 1996, 2001-2003 Free Software Foundation, Inc.

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
mpf_random2 (mpf_ptr x, mp_size_t xs, mp_exp_t exp)
{
  mp_size_t xn;
  mp_size_t prec;
  mp_limb_t elimb;

  xn = ABS (xs);
  prec = PREC(x);

  if (xn == 0)
    {
      EXP(x) = 0;
      SIZ(x) = 0;
      return;
    }

  if (xn > prec + 1)
    xn = prec + 1;

  /* General random mantissa.  */
  mpn_random2 (PTR(x), xn);

  /* Generate random exponent.  */
  _gmp_rand (&elimb, RANDS, GMP_NUMB_BITS);
  exp = ABS (exp);
  exp = elimb % (2 * exp + 1) - exp;

  EXP(x) = exp;
  SIZ(x) = xs < 0 ? -xn : xn;
}
