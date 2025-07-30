/* mpfr_frexp -- convert to integral and fractional parts

Copyright 2011-2017 Free Software Foundation, Inc.
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
mpfr_frexp (mpfr_exp_t *exp, mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  int inex;
  unsigned int saved_flags = __gmpfr_flags;
  MPFR_BLOCK_DECL (flags);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd),
     ("y[%Pu]=%.*Rg exp=%" MPFR_EXP_FSPEC "d inex=%d", mpfr_get_prec (y),
      mpfr_log_prec, y, (mpfr_eexp_t) *exp, inex));

  if (MPFR_UNLIKELY(MPFR_IS_SINGULAR(x)))
    {
      if (MPFR_IS_NAN(x))
        {
          MPFR_SET_NAN(y);
          MPFR_RET_NAN; /* exp is unspecified */
        }
      else if (MPFR_IS_INF(x))
        {
          MPFR_SET_INF(y);
          MPFR_SET_SAME_SIGN(y,x);
          MPFR_RET(0); /* exp is unspecified */
        }
      else
        {
          MPFR_SET_ZERO(y);
          MPFR_SET_SAME_SIGN(y,x);
          *exp = 0;
          MPFR_RET(0);
        }
    }

  MPFR_BLOCK (flags, inex = mpfr_set (y, x, rnd));
  __gmpfr_flags = saved_flags;

  /* Possible overflow due to the rounding, no possible underflow. */

  if (MPFR_UNLIKELY (MPFR_OVERFLOW (flags)))
    {
      int inex2;

      /* An overflow here means that the exponent of y would be larger than
         the one of x, thus x would be rounded to the next power of 2, and
         the returned y should be 1/2 in absolute value, rounded (i.e. with
         possible underflow or overflow). This also implies that x and y are
         different objects, so that the exponent of x has not been lost. */
      MPFR_LOG_MSG (("Internal overflow\n", 0));
      MPFR_ASSERTD (x != y);
      *exp = MPFR_GET_EXP (x) + 1;
      inex2 = mpfr_set_si_2exp (y, MPFR_INT_SIGN (x), -1, rnd);
      MPFR_LOG_MSG (("inex=%d inex2=%d\n", inex, inex2));
      if (inex2 != 0)
        inex = inex2;
      MPFR_RET (inex);
    }

  *exp = MPFR_GET_EXP (y);
  /* Do not use MPFR_SET_EXP because the range has not been checked yet. */
  MPFR_EXP (y) = 0;
  return mpfr_check_range (y, inex, rnd);
}
