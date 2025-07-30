/* mpfr_mul_2ui -- multiply a floating-point number by a power of two

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

int
mpfr_mul_2ui (mpfr_ptr y, mpfr_srcptr x, unsigned long int n, mpfr_rnd_t rnd_mode)
{
  int inexact;

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg n=%lu rnd=%d",
      mpfr_get_prec (x), mpfr_log_prec, x, n, rnd_mode),
     ("y[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (y), mpfr_log_prec, y, inexact));

  inexact = MPFR_UNLIKELY(y != x) ? mpfr_set (y, x, rnd_mode) : 0;

  if (MPFR_LIKELY( MPFR_IS_PURE_FP(y)) )
    {
      /* n will have to be casted to long to make sure that the addition
         and subtraction below (for overflow detection) are signed */
      while (MPFR_UNLIKELY(n > LONG_MAX))
        {
          int inex2;

          n -= LONG_MAX;
          inex2 = mpfr_mul_2ui(y, y, LONG_MAX, rnd_mode);
          if (inex2)
            return inex2; /* overflow */
        }

      /* MPFR_EMIN_MIN + (long) n is signed and doesn't lead to an overflow;
         the first test useful so that the real test can't lead to an
         overflow. */
      {
        mpfr_exp_t exp = MPFR_GET_EXP (y);
        if (MPFR_UNLIKELY( __gmpfr_emax < MPFR_EMIN_MIN + (long) n ||
                           exp > __gmpfr_emax - (long) n))
          return mpfr_overflow (y, rnd_mode, MPFR_SIGN(y));

        MPFR_SET_EXP (y, exp + (long) n);
      }
    }

  return inexact;
}
