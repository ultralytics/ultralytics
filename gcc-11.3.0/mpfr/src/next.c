/* mpfr_nextabove, mpfr_nextbelow, mpfr_nexttoward -- next representable
floating-point number

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

void
mpfr_nexttozero (mpfr_ptr x)
{
  if (MPFR_UNLIKELY(MPFR_IS_INF(x)))
    {
      mpfr_setmax (x, __gmpfr_emax);
      return;
    }
  else if (MPFR_UNLIKELY( MPFR_IS_ZERO(x) ))
    {
      MPFR_CHANGE_SIGN(x);
      mpfr_setmin (x, __gmpfr_emin);
    }
  else
    {
      mp_size_t xn;
      int sh;
      mp_limb_t *xp;

      xn = MPFR_LIMB_SIZE (x);
      MPFR_UNSIGNED_MINUS_MODULO (sh, MPFR_PREC(x));
      xp = MPFR_MANT(x);
      mpn_sub_1 (xp, xp, xn, MPFR_LIMB_ONE << sh);
      if (MPFR_UNLIKELY( MPFR_LIMB_MSB(xp[xn-1]) == 0) )
        { /* was an exact power of two: not normalized any more */
          mpfr_exp_t exp = MPFR_EXP (x);
          if (MPFR_UNLIKELY(exp == __gmpfr_emin))
            MPFR_SET_ZERO(x);
          else
            {
              mp_size_t i;
              MPFR_SET_EXP (x, exp - 1);
              xp[0] = MP_LIMB_T_MAX << sh;
              for (i = 1; i < xn; i++)
                xp[i] = MP_LIMB_T_MAX;
            }
        }
    }
}

void
mpfr_nexttoinf (mpfr_ptr x)
{
  if (MPFR_UNLIKELY(MPFR_IS_INF(x)))
    return;
  else if (MPFR_UNLIKELY(MPFR_IS_ZERO(x)))
    mpfr_setmin (x, __gmpfr_emin);
  else
    {
      mp_size_t xn;
      int sh;
      mp_limb_t *xp;

      xn = MPFR_LIMB_SIZE (x);
      MPFR_UNSIGNED_MINUS_MODULO (sh, MPFR_PREC(x));
      xp = MPFR_MANT(x);
      if (MPFR_UNLIKELY( mpn_add_1 (xp, xp, xn, MPFR_LIMB_ONE << sh)) )
        /* got 1.0000... */
        {
          mpfr_exp_t exp = MPFR_EXP (x);
          if (MPFR_UNLIKELY(exp == __gmpfr_emax))
            MPFR_SET_INF(x);
          else
            {
              MPFR_SET_EXP (x, exp + 1);
              xp[xn-1] = MPFR_LIMB_HIGHBIT;
            }
        }
    }
}

void
mpfr_nextabove (mpfr_ptr x)
{
  if (MPFR_UNLIKELY(MPFR_IS_NAN(x)))
    {
      __gmpfr_flags |= MPFR_FLAGS_NAN;
      return;
    }
  if (MPFR_IS_NEG(x))
    mpfr_nexttozero (x);
  else
    mpfr_nexttoinf (x);
}

void
mpfr_nextbelow (mpfr_ptr x)
{
  if (MPFR_UNLIKELY(MPFR_IS_NAN(x)))
    {
      __gmpfr_flags |= MPFR_FLAGS_NAN;
      return;
    }

  if (MPFR_IS_NEG(x))
    mpfr_nexttoinf (x);
  else
    mpfr_nexttozero (x);
}

void
mpfr_nexttoward (mpfr_ptr x, mpfr_srcptr y)
{
  int s;

  if (MPFR_UNLIKELY(MPFR_IS_NAN(x)))
    {
      __gmpfr_flags |= MPFR_FLAGS_NAN;
      return;
    }
  else if (MPFR_UNLIKELY(MPFR_IS_NAN(x) || MPFR_IS_NAN(y)))
    {
      MPFR_SET_NAN(x);
      __gmpfr_flags |= MPFR_FLAGS_NAN;
      return;
    }

  s = mpfr_cmp (x, y);
  if (s == 0)
    return;
  else if (s < 0)
    mpfr_nextabove (x);
  else
    mpfr_nextbelow (x);
}
