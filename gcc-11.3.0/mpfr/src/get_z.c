/* mpfr_get_z -- get a multiple-precision integer from
                 a floating-point number

Copyright 2004, 2006-2017 Free Software Foundation, Inc.
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
mpfr_get_z (mpz_ptr z, mpfr_srcptr f, mpfr_rnd_t rnd)
{
  int inex;
  mpfr_t r;
  mpfr_exp_t exp;

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (f)))
    {
      if (MPFR_UNLIKELY (MPFR_NOTZERO (f)))
        MPFR_SET_ERANGE ();
      mpz_set_ui (z, 0);
      /* The ternary value is 0 even for infinity. Giving the rounding
         direction in this case would not make much sense anyway, and
         the direction would not necessarily match rnd. */
      return 0;
    }

  exp = MPFR_GET_EXP (f);
  /* if exp <= 0, then |f|<1, thus |o(f)|<=1 */
  MPFR_ASSERTN (exp < 0 || exp <= MPFR_PREC_MAX);
  mpfr_init2 (r, (exp < (mpfr_exp_t) MPFR_PREC_MIN ?
                  MPFR_PREC_MIN : (mpfr_prec_t) exp));
  inex = mpfr_rint (r, f, rnd);
  MPFR_ASSERTN (inex != 1 && inex != -1); /* integral part of f is
                                             representable in r */
  MPFR_ASSERTN (MPFR_IS_FP (r));
  exp = mpfr_get_z_2exp (z, r);
  if (exp >= 0)
    mpz_mul_2exp (z, z, exp);
  else
    mpz_fdiv_q_2exp (z, z, -exp);
  mpfr_clear (r);

  return inex;
}
