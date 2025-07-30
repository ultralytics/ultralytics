/* mpfr_set -- copy of a floating-point number

Copyright 1999, 2001-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include "mpfr-impl.h"

/* set a to abs(b) * signb: a=b when signb = SIGN(b), a=abs(b) when signb=1 */
int
mpfr_set4 (mpfr_ptr a, mpfr_srcptr b, mpfr_rnd_t rnd_mode, int signb)
{
  /* Sign is ALWAYS copied */
  MPFR_SET_SIGN (a, signb);

  /* Exponent is also always copied since if the number is singular,
     the exponent field determined the number.
     Can't use MPFR_SET_EXP since the exponent may be singular */
  MPFR_EXP (a) = MPFR_EXP (b);

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (b)))
    {
      /* MPFR_SET_NAN, MPFR_SET_ZERO and MPFR_SET_INF are useless
         since MPFR_EXP (a) = MPFR_EXP (b) does the job */
      if (MPFR_IS_NAN (b))
        MPFR_RET_NAN;
      else
        MPFR_RET (0);
    }
  else if (MPFR_LIKELY (MPFR_PREC (b) == MPFR_PREC (a)))
    {
      /* Same precision and b is not singular:
       * just copy the mantissa, and set the exponent and the sign
       * The result is exact. */
      MPN_COPY (MPFR_MANT (a), MPFR_MANT (b), MPFR_LIMB_SIZE (b));
      MPFR_RET (0);
    }
  else
    {
      int inex;

      /* Else Round B inside a */
      MPFR_RNDRAW (inex, a, MPFR_MANT (b), MPFR_PREC (b), rnd_mode, signb,
                   if (MPFR_UNLIKELY ( ++MPFR_EXP (a) > __gmpfr_emax))
                    return mpfr_overflow (a, rnd_mode, signb) );
      MPFR_RET (inex);
    }
}

/* Set a to b  */
#undef mpfr_set
int
mpfr_set (mpfr_ptr a, mpfr_srcptr b, mpfr_rnd_t rnd_mode)
{
  return mpfr_set4 (a, b, rnd_mode, MPFR_SIGN (b));
}

/* Set a to |b| */
#undef mpfr_abs
int
mpfr_abs (mpfr_ptr a, mpfr_srcptr b, mpfr_rnd_t rnd_mode)
{
  return mpfr_set4 (a, b, rnd_mode, MPFR_SIGN_POS);
}
