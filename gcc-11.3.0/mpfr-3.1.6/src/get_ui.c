/* mpfr_get_ui -- convert a floating-point number to an unsigned long.

Copyright 2003-2004, 2006-2017 Free Software Foundation, Inc.
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

unsigned long
mpfr_get_ui (mpfr_srcptr f, mpfr_rnd_t rnd)
{
  mpfr_prec_t prec;
  unsigned long s;
  mpfr_t x;
  mp_size_t n;
  mpfr_exp_t exp;

  if (MPFR_UNLIKELY (!mpfr_fits_ulong_p (f, rnd)))
    {
      MPFR_SET_ERANGE ();
      return MPFR_IS_NAN (f) || MPFR_IS_NEG (f) ?
        (unsigned long) 0 : ULONG_MAX;
    }

  if (MPFR_IS_ZERO (f))
    return (unsigned long) 0;

  for (s = ULONG_MAX, prec = 0; s != 0; s /= 2, prec ++)
    { }

  /* first round to prec bits */
  mpfr_init2 (x, prec);
  mpfr_rint (x, f, rnd);

  /* warning: if x=0, taking its exponent is illegal */
  if (MPFR_IS_ZERO(x))
    s = 0;
  else
    {
      /* now the result is in the most significant limb of x */
      exp = MPFR_GET_EXP (x); /* since |x| >= 1, exp >= 1 */
      n = MPFR_LIMB_SIZE(x);
      s = MPFR_MANT(x)[n - 1] >> (GMP_NUMB_BITS - exp);
    }

  mpfr_clear (x);

  return s;
}
