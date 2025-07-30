/*  mpfr_ui_pow_ui -- compute the power beetween two machine integer

Copyright 1999-2017 Free Software Foundation, Inc.
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
mpfr_ui_pow_ui (mpfr_ptr x, unsigned long int y, unsigned long int n,
                mpfr_rnd_t rnd)
{
  mpfr_exp_t err;
  unsigned long m;
  mpfr_t res;
  mpfr_prec_t prec;
  int size_n;
  int inexact;
  MPFR_ZIV_DECL (loop);
  MPFR_SAVE_EXPO_DECL (expo);

  if (MPFR_UNLIKELY (n <= 1))
    {
      if (n == 1)
        return mpfr_set_ui (x, y, rnd);     /* y^1 = y */
      else
        return mpfr_set_ui (x, 1, rnd);     /* y^0 = 1 for any y */
    }
  else if (MPFR_UNLIKELY (y <= 1))
    {
      if (y == 1)
        return mpfr_set_ui (x, 1, rnd);     /* 1^n = 1 for any n > 0 */
      else
        return mpfr_set_ui (x, 0, rnd);     /* 0^n = 0 for any n > 0 */
    }

  for (size_n = 0, m = n; m; size_n++, m >>= 1);

  MPFR_SAVE_EXPO_MARK (expo);
  prec = MPFR_PREC (x) + 3 + size_n;
  mpfr_init2 (res, prec);

  MPFR_ZIV_INIT (loop, prec);
  for (;;)
    {
      int i = size_n;

      inexact = mpfr_set_ui (res, y, MPFR_RNDU);
      err = 1;
      /* now 2^(i-1) <= n < 2^i: i=1+floor(log2(n)) */
      for (i -= 2; i >= 0; i--)
        {
          inexact |= mpfr_mul (res, res, res, MPFR_RNDU);
          err++;
          if (n & (1UL << i))
            inexact |= mpfr_mul_ui (res, res, y, MPFR_RNDU);
        }
      /* since the loop is executed floor(log2(n)) times,
         we have err = 1+floor(log2(n)).
         Since prec >= MPFR_PREC(x) + 4 + floor(log2(n)), prec > err */
      err = prec - err;

      if (MPFR_LIKELY (inexact == 0
                       || MPFR_CAN_ROUND (res, err, MPFR_PREC (x), rnd)))
        break;

      /* Actualisation of the precision */
      MPFR_ZIV_NEXT (loop, prec);
      mpfr_set_prec (res, prec);
    }
  MPFR_ZIV_FREE (loop);

  inexact = mpfr_set (x, res, rnd);

  mpfr_clear (res);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (x, inexact, rnd);
}
