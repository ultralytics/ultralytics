/* Exception flags and utilities.

Copyright 2001-2017 Free Software Foundation, Inc.
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

MPFR_THREAD_VAR (unsigned int, __gmpfr_flags, 0)
MPFR_THREAD_VAR (mpfr_exp_t, __gmpfr_emin, MPFR_EMIN_DEFAULT)
MPFR_THREAD_VAR (mpfr_exp_t, __gmpfr_emax, MPFR_EMAX_DEFAULT)

#undef mpfr_get_emin

mpfr_exp_t
mpfr_get_emin (void)
{
  return __gmpfr_emin;
}

#undef mpfr_set_emin

int
mpfr_set_emin (mpfr_exp_t exponent)
{
  if (exponent >= MPFR_EMIN_MIN && exponent <= MPFR_EMIN_MAX)
    {
      __gmpfr_emin = exponent;
      return 0;
    }
  else
    {
      return 1;
    }
}

mpfr_exp_t
mpfr_get_emin_min (void)
{
  return MPFR_EMIN_MIN;
}

mpfr_exp_t
mpfr_get_emin_max (void)
{
  return MPFR_EMIN_MAX;
}

#undef mpfr_get_emax

mpfr_exp_t
mpfr_get_emax (void)
{
  return __gmpfr_emax;
}

#undef mpfr_set_emax

int
mpfr_set_emax (mpfr_exp_t exponent)
{
  if (exponent >= MPFR_EMAX_MIN && exponent <= MPFR_EMAX_MAX)
    {
      __gmpfr_emax = exponent;
      return 0;
    }
  else
    {
      return 1;
    }
}

mpfr_exp_t
mpfr_get_emax_min (void)
{
  return MPFR_EMAX_MIN;
}
mpfr_exp_t
mpfr_get_emax_max (void)
{
  return MPFR_EMAX_MAX;
}


#undef mpfr_clear_flags

void
mpfr_clear_flags (void)
{
  __gmpfr_flags = 0;
}

#undef mpfr_clear_underflow

void
mpfr_clear_underflow (void)
{
  __gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_UNDERFLOW;
}

#undef mpfr_clear_overflow

void
mpfr_clear_overflow (void)
{
  __gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_OVERFLOW;
}

#undef mpfr_clear_divby0

void
mpfr_clear_divby0 (void)
{
  __gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_DIVBY0;
}

#undef mpfr_clear_nanflag

void
mpfr_clear_nanflag (void)
{
  __gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_NAN;
}

#undef mpfr_clear_inexflag

void
mpfr_clear_inexflag (void)
{
  __gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_INEXACT;
}

#undef mpfr_clear_erangeflag

void
mpfr_clear_erangeflag (void)
{
  __gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_ERANGE;
}

#undef mpfr_set_underflow

void
mpfr_set_underflow (void)
{
  __gmpfr_flags |= MPFR_FLAGS_UNDERFLOW;
}

#undef mpfr_set_overflow

void
mpfr_set_overflow (void)
{
  __gmpfr_flags |= MPFR_FLAGS_OVERFLOW;
}

#undef mpfr_set_divby0

void
mpfr_set_divby0 (void)
{
  __gmpfr_flags |= MPFR_FLAGS_DIVBY0;
}

#undef mpfr_set_nanflag

void
mpfr_set_nanflag (void)
{
  __gmpfr_flags |= MPFR_FLAGS_NAN;
}

#undef mpfr_set_inexflag

void
mpfr_set_inexflag (void)
{
  __gmpfr_flags |= MPFR_FLAGS_INEXACT;
}

#undef mpfr_set_erangeflag

void
mpfr_set_erangeflag (void)
{
  __gmpfr_flags |= MPFR_FLAGS_ERANGE;
}


#undef mpfr_check_range

int
mpfr_check_range (mpfr_ptr x, int t, mpfr_rnd_t rnd_mode)
{
  if (MPFR_LIKELY( MPFR_IS_PURE_FP(x)) )
    { /* x is a non-zero FP */
      mpfr_exp_t exp = MPFR_EXP (x);  /* Do not use MPFR_GET_EXP */
      if (MPFR_UNLIKELY( exp < __gmpfr_emin) )
        {
          /* The following test is necessary because in the rounding to the
           * nearest mode, mpfr_underflow always rounds away from 0. In
           * this rounding mode, we need to round to 0 if:
           *   _ |x| < 2^(emin-2), or
           *   _ |x| = 2^(emin-2) and the absolute value of the exact
           *     result is <= 2^(emin-2).
           */
          if (rnd_mode == MPFR_RNDN &&
              (exp + 1 < __gmpfr_emin ||
               (mpfr_powerof2_raw(x) &&
                (MPFR_IS_NEG(x) ? t <= 0 : t >= 0))))
            rnd_mode = MPFR_RNDZ;
          return mpfr_underflow(x, rnd_mode, MPFR_SIGN(x));
        }
      if (MPFR_UNLIKELY( exp > __gmpfr_emax) )
        return mpfr_overflow (x, rnd_mode, MPFR_SIGN(x));
    }
  else if (MPFR_UNLIKELY (t != 0 && MPFR_IS_INF (x)))
    {
      /* We need to do the following because most MPFR functions are
       * implemented in the following way:
       *   Ziv's loop:
       *   | Compute an approximation to the result and an error bound.
       *   | Possible underflow/overflow detection -> return.
       *   | If can_round, break (exit the loop).
       *   | Otherwise, increase the working precision and loop.
       *   Round the approximation in the target precision.  <== See below
       *   Restore the flags (that could have been set due to underflows
       *   or overflows during the internal computations).
       *   Execute: return mpfr_check_range (...).
       * The problem is that an overflow could be generated when rounding the
       * approximation (in general, such an overflow could not be detected
       * earlier), and the overflow flag is lost when the flags are restored.
       * This can occur only when the rounding yields an exponent change
       * and the new exponent is larger than the maximum exponent, so that
       * an infinity is necessarily obtained.
       * So, the simplest solution is to detect this overflow case here in
       * mpfr_check_range, which is easy to do since the rounded result is
       * necessarily an inexact infinity.
       */
      __gmpfr_flags |= MPFR_FLAGS_OVERFLOW;
    }
  MPFR_RET (t);  /* propagate inexact ternary value, unlike most functions */
}

#undef mpfr_underflow_p

int
mpfr_underflow_p (void)
{
  return __gmpfr_flags & MPFR_FLAGS_UNDERFLOW;
}

#undef mpfr_overflow_p

int
mpfr_overflow_p (void)
{
  return __gmpfr_flags & MPFR_FLAGS_OVERFLOW;
}

#undef mpfr_divby0_p

int
mpfr_divby0_p (void)
{
  return __gmpfr_flags & MPFR_FLAGS_DIVBY0;
}

#undef mpfr_nanflag_p

int
mpfr_nanflag_p (void)
{
  return __gmpfr_flags & MPFR_FLAGS_NAN;
}

#undef mpfr_inexflag_p

int
mpfr_inexflag_p (void)
{
  return __gmpfr_flags & MPFR_FLAGS_INEXACT;
}

#undef mpfr_erangeflag_p

int
mpfr_erangeflag_p (void)
{
  return __gmpfr_flags & MPFR_FLAGS_ERANGE;
}

/* #undef mpfr_underflow */

/* Note: In the rounding to the nearest mode, mpfr_underflow
   always rounds away from 0. In this rounding mode, you must call
   mpfr_underflow with rnd_mode = MPFR_RNDZ if the exact result
   is <= 2^(emin-2) in absolute value. */

int
mpfr_underflow (mpfr_ptr x, mpfr_rnd_t rnd_mode, int sign)
{
  int inex;

  MPFR_ASSERT_SIGN (sign);

  if (MPFR_IS_LIKE_RNDZ(rnd_mode, sign < 0))
    {
      MPFR_SET_ZERO(x);
      inex = -1;
    }
  else
    {
      mpfr_setmin (x, __gmpfr_emin);
      inex = 1;
    }
  MPFR_SET_SIGN(x, sign);
  __gmpfr_flags |= MPFR_FLAGS_INEXACT | MPFR_FLAGS_UNDERFLOW;
  return sign > 0 ? inex : -inex;
}

/* #undef mpfr_overflow */

int
mpfr_overflow (mpfr_ptr x, mpfr_rnd_t rnd_mode, int sign)
{
  int inex;

  MPFR_ASSERT_SIGN(sign);
  if (MPFR_IS_LIKE_RNDZ(rnd_mode, sign < 0))
    {
      mpfr_setmax (x, __gmpfr_emax);
      inex = -1;
    }
  else
    {
      MPFR_SET_INF(x);
      inex = 1;
    }
  MPFR_SET_SIGN(x,sign);
  __gmpfr_flags |= MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW;
  return sign > 0 ? inex : -inex;
}
