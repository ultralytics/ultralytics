/* mpfr_get_z_2exp -- get a multiple-precision integer and an exponent
                      from a floating-point number

Copyright 2000-2017 Free Software Foundation, Inc.
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

/* puts the significand of f into z, and returns 'exp' such that f = z * 2^exp
 *
 * 0 doesn't have an exponent, therefore the returned exponent in this case
 * isn't really important. We choose to return __gmpfr_emin because
 *   1) it is in the exponent range [__gmpfr_emin,__gmpfr_emax],
 *   2) the smaller a number is (in absolute value), the smaller its
 *      exponent is. In other words, the f -> exp function is monotonous
 *      on nonnegative numbers. --> This is WRONG since the returned
 *      exponent is not necessarily in the exponent range!
 * Note that this is different from the C function frexp().
 *
 * For NaN and infinities, we choose to set z = 0 (neutral value).
 * The exponent doesn't really matter, so let's keep __gmpfr_emin
 * for consistency. The erange flag is set.
 */

mpfr_exp_t
mpfr_get_z_2exp (mpz_ptr z, mpfr_srcptr f)
{
  mp_size_t fn;
  int sh;

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (f)))
    {
      if (MPFR_UNLIKELY (MPFR_NOTZERO (f)))
        MPFR_SET_ERANGE ();
      mpz_set_ui (z, 0);
      return __gmpfr_emin;
    }

  fn = MPFR_LIMB_SIZE(f);

  /* check whether allocated space for z is enough */
  if (MPFR_UNLIKELY (ALLOC (z) < fn))
    MPZ_REALLOC (z, fn);

  MPFR_UNSIGNED_MINUS_MODULO (sh, MPFR_PREC (f));
  if (MPFR_LIKELY (sh))
    mpn_rshift (PTR (z), MPFR_MANT (f), fn, sh);
  else
    MPN_COPY (PTR (z), MPFR_MANT (f), fn);

  SIZ(z) = MPFR_IS_NEG (f) ? -fn : fn;

  if (MPFR_UNLIKELY ((mpfr_uexp_t) MPFR_GET_EXP (f) - MPFR_EXP_MIN
                     < (mpfr_uexp_t) MPFR_PREC (f)))
    {
      /* The exponent isn't representable in an mpfr_exp_t. */
      MPFR_SET_ERANGE ();
      return MPFR_EXP_MIN;
    }

  return MPFR_GET_EXP (f) - MPFR_PREC (f);
}
