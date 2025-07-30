/* mpfr_stack -- initialize a floating-point number with given allocation area

Copyright 2005-2017 Free Software Foundation, Inc.
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

#undef mpfr_custom_get_size
size_t
mpfr_custom_get_size (mpfr_prec_t prec)
{
  return MPFR_PREC2LIMBS (prec) * MPFR_BYTES_PER_MP_LIMB;
}

#undef mpfr_custom_init
void
mpfr_custom_init (void *mantissa, mpfr_prec_t prec)
{
  return ;
}

#undef mpfr_custom_get_significand
void *
mpfr_custom_get_significand (mpfr_srcptr x)
{
  return (void*) MPFR_MANT (x);
}

#undef mpfr_custom_get_exp
mpfr_exp_t
mpfr_custom_get_exp (mpfr_srcptr x)
{
  return MPFR_EXP (x);
}

#undef mpfr_custom_move
void
mpfr_custom_move (mpfr_ptr x, void *new_position)
{
  MPFR_MANT (x) = (mp_limb_t *) new_position;
}

#undef mpfr_custom_init_set
void
mpfr_custom_init_set (mpfr_ptr x, int kind, mpfr_exp_t exp,
                     mpfr_prec_t prec, void *mantissa)
{
  mpfr_kind_t t;
  int s;
  mpfr_exp_t e;

  if (kind >= 0)
    {
      t = (mpfr_kind_t) kind;
      s = MPFR_SIGN_POS;
    }
  else
    {
      t = (mpfr_kind_t) -kind;
      s = MPFR_SIGN_NEG;
    }
  MPFR_ASSERTD (t <= MPFR_REGULAR_KIND);
  e = MPFR_LIKELY (t == MPFR_REGULAR_KIND) ? exp :
    MPFR_UNLIKELY (t == MPFR_NAN_KIND) ? MPFR_EXP_NAN :
    MPFR_UNLIKELY (t == MPFR_INF_KIND) ? MPFR_EXP_INF : MPFR_EXP_ZERO;

  MPFR_PREC (x) = prec;
  MPFR_SET_SIGN (x, s);
  MPFR_EXP (x) = e;
  MPFR_MANT (x) = (mp_limb_t*) mantissa;
  return;
}

#undef mpfr_custom_get_kind
int
mpfr_custom_get_kind (mpfr_srcptr x)
{
  if (MPFR_LIKELY (!MPFR_IS_SINGULAR (x)))
    return (int) MPFR_REGULAR_KIND * MPFR_INT_SIGN (x);
  if (MPFR_IS_INF (x))
    return (int) MPFR_INF_KIND * MPFR_INT_SIGN (x);
  if (MPFR_IS_NAN (x))
    return (int) MPFR_NAN_KIND;
  MPFR_ASSERTD (MPFR_IS_ZERO (x));
  return (int) MPFR_ZERO_KIND * MPFR_INT_SIGN (x);
}

