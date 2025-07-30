/* mpfr_div_2ui -- divide a floating-point number by a power of two

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
mpfr_div_2ui (mpfr_ptr y, mpfr_srcptr x, unsigned long n, mpfr_rnd_t rnd_mode)
{
  int inexact;

  MPFR_LOG_FUNC (
    ("x[%Pu]=%.*Rg n=%lu rnd=%d", mpfr_get_prec(x), mpfr_log_prec, x, n,
     rnd_mode),
    ("y[%Pu]=%.*Rg inexact=%d", mpfr_get_prec(y), mpfr_log_prec, y, inexact));

  if (MPFR_UNLIKELY (n == 0 || MPFR_IS_SINGULAR (x)))
    return mpfr_set (y, x, rnd_mode);
  else
    {
      mpfr_exp_t exp = MPFR_GET_EXP (x);
      mpfr_uexp_t diffexp;

      MPFR_SETRAW (inexact, y, x, exp, rnd_mode);
      diffexp = (mpfr_uexp_t) exp - (mpfr_uexp_t) (__gmpfr_emin - 1);
      if (MPFR_UNLIKELY (n >= diffexp))  /* exp - n <= emin - 1 */
        {
          if (rnd_mode == MPFR_RNDN &&
              (n > diffexp ||
               ((MPFR_IS_NEG (y) ? inexact <= 0 : inexact >= 0) &&
                mpfr_powerof2_raw (y))))
            rnd_mode = MPFR_RNDZ;
          return mpfr_underflow (y, rnd_mode, MPFR_SIGN (y));
        }
      /* exp - n >= emin (no underflow, no integer overflow) */
      while (n > LONG_MAX)
        {
          n -= LONG_MAX;
          exp -= LONG_MAX;  /* note: signed values */
        }
      MPFR_SET_EXP (y, exp - (long) n);
    }

  MPFR_RET (inexact);
}
