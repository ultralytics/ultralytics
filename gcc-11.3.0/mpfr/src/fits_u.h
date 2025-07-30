/* mpfr_fits_*_p -- test whether an mpfr fits a C unsigned type.

Copyright 2003-2017 Free Software Foundation, Inc.
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
FUNCTION (mpfr_srcptr f, mpfr_rnd_t rnd)
{
  unsigned int saved_flags;
  mpfr_exp_t e;
  int prec;
  TYPE s;
  mpfr_t x;
  int res;

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (f)))
    return MPFR_IS_ZERO (f) ? 1 : 0;  /* Zero always fits */

  e = MPFR_GET_EXP (f);

  if (MPFR_IS_NEG (f))
    return e >= 1 ? 0  /* f <= -1 does not fit */
      : rnd != MPFR_RNDN ? MPFR_IS_LIKE_RNDU (rnd, -1)  /* directed mode */
      : e < 0 ? 1  /* f > -1/2 fits in MPFR_RNDN */
      : mpfr_powerof2_raw(f);  /* -1/2 fits, -1 < f < -1/2 don't */

  /* Now it fits if
     (a) f <= MAXIMUM
     (b) round(f, prec(slong), rnd) <= MAXIMUM */

  /* first compute prec(MAXIMUM); fits in an int */
  for (s = MAXIMUM, prec = 0; s != 0; s /= 2, prec ++);

  /* MAXIMUM needs prec bits, i.e. MAXIMUM = 2^prec - 1 */

  /* if e <= prec - 1, then f < 2^(prec-1) < MAXIMUM */
  if (e <= prec - 1)
    return 1;

  /* if e >= prec + 1, then f >= 2^prec > MAXIMUM */
  if (e >= prec + 1)
    return 0;

  MPFR_ASSERTD (e == prec);

  /* hard case: first round to prec bits, then check */
  saved_flags = __gmpfr_flags;
  mpfr_init2 (x, prec);
  mpfr_set (x, f, rnd);
  /* Warning! Due to the rounding, x can be an infinity. Here we use
     the fact that singular numbers have a special exponent field,
     thus well-defined and different from e, in which case this means
     that the number does not fit. That's why we use MPFR_EXP, not
     MPFR_GET_EXP. */
  res = MPFR_EXP (x) == e;
  mpfr_clear (x);
  __gmpfr_flags = saved_flags;
  return res;
}
