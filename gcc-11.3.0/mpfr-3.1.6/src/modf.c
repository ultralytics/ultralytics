/* mpfr_modf -- Integral and fractional part.

Copyright 2007-2017 Free Software Foundation, Inc.
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

#define INEXPOS(y) ((y) == 0 ? 0 : (((y) > 0) ? 1 : 2))
#define INEX(y,z) (INEXPOS(y) | (INEXPOS(z) << 2))

/* Set iop to the integral part of op and fop to its fractional part */
int
mpfr_modf (mpfr_ptr iop, mpfr_ptr fop, mpfr_srcptr op, mpfr_rnd_t rnd_mode)
{
  mpfr_exp_t ope;
  mpfr_prec_t opq;
  int inexi, inexf;

  MPFR_LOG_FUNC
    (("op[%Pu]=%.*Rg rnd=%d",
      mpfr_get_prec (op), mpfr_log_prec, op, rnd_mode),
     ("iop[%Pu]=%.*Rg fop[%Pu]=%.*Rg",
      mpfr_get_prec (iop), mpfr_log_prec, iop,
      mpfr_get_prec (fop), mpfr_log_prec, fop));

  MPFR_ASSERTN (iop != fop);

  if ( MPFR_UNLIKELY (MPFR_IS_SINGULAR (op)) )
    {
      if (MPFR_IS_NAN (op))
        {
          MPFR_SET_NAN (iop);
          MPFR_SET_NAN (fop);
          MPFR_RET_NAN;
        }
      MPFR_SET_SAME_SIGN (iop, op);
      MPFR_SET_SAME_SIGN (fop, op);
      if (MPFR_IS_INF (op))
        {
          MPFR_SET_INF (iop);
          MPFR_SET_ZERO (fop);
          MPFR_RET (0);
        }
      else /* op is zero */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (op));
          MPFR_SET_ZERO (iop);
          MPFR_SET_ZERO (fop);
          MPFR_RET (0);
        }
    }

  ope = MPFR_GET_EXP (op);
  opq = MPFR_PREC (op);

  if (ope <= 0)   /* 0 < |op| < 1 */
    {
      inexf = (fop != op) ? mpfr_set (fop, op, rnd_mode) : 0;
      MPFR_SET_SAME_SIGN (iop, op);
      MPFR_SET_ZERO (iop);
      MPFR_RET (INEX(0, inexf));
    }
  else if (ope >= opq) /* op has no fractional part */
    {
      inexi = (iop != op) ? mpfr_set (iop, op, rnd_mode) : 0;
      MPFR_SET_SAME_SIGN (fop, op);
      MPFR_SET_ZERO (fop);
      MPFR_RET (INEX(inexi, 0));
    }
  else /* op has both integral and fractional parts */
    {
      if (iop != op)
        {
          inexi = mpfr_rint_trunc (iop, op, rnd_mode);
          inexf = mpfr_frac (fop, op, rnd_mode);
        }
      else
        {
          MPFR_ASSERTN (fop != op);
          inexf = mpfr_frac (fop, op, rnd_mode);
          inexi = mpfr_rint_trunc (iop, op, rnd_mode);
        }
      MPFR_RET (INEX(inexi, inexf));
    }
}
