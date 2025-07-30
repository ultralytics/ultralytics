/* mpfr_set_uj -- set a MPFR number from a huge machine unsigned integer

Copyright 2004-2017 Free Software Foundation, Inc.
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

#ifdef HAVE_CONFIG_H
# include "config.h"       /* for a build within gmp */
#endif

#define MPFR_NEED_LONGLONG_H
#include "mpfr-intmax.h"
#include "mpfr-impl.h"

#ifdef _MPFR_H_HAVE_INTMAX_T

int
mpfr_set_uj (mpfr_t x, uintmax_t j, mpfr_rnd_t rnd)
{
  return mpfr_set_uj_2exp (x, j, 0, rnd);
}

int
mpfr_set_uj_2exp (mpfr_t x, uintmax_t j, intmax_t e, mpfr_rnd_t rnd)
{
  unsigned int cnt, i;
  mp_size_t k, len;
  mp_limb_t limb;
  mp_limb_t yp[sizeof(uintmax_t) / sizeof(mp_limb_t)];
  mpfr_t y;
  unsigned long uintmax_bit_size = sizeof(uintmax_t) * CHAR_BIT;
  unsigned long bpml = GMP_NUMB_BITS % uintmax_bit_size;

  /* Special case */
  if (j == 0)
    {
      MPFR_SET_POS(x);
      MPFR_SET_ZERO(x);
      MPFR_RET(0);
    }

  MPFR_ASSERTN (sizeof(uintmax_t) % sizeof(mp_limb_t) == 0);

  /* Create an auxillary var */
  MPFR_TMP_INIT1 (yp, y, uintmax_bit_size);
  k = numberof (yp);
  if (k == 1)
    limb = yp[0] = j;
  else
    {
      /* Note: either GMP_NUMB_BITS = uintmax_bit_size, then k = 1 the
         shift j >>= bpml is never done, or GMP_NUMB_BITS < uintmax_bit_size
         and bpml = GMP_NUMB_BITS. */
      for (i = 0; i < k; i++, j >>= bpml)
        yp[i] = j; /* Only the low bits are copied */

      /* Find the first limb not equal to zero. */
      do
        {
          MPFR_ASSERTD (k > 0);
          limb = yp[--k];
        }
      while (limb == 0);
      k++;
    }
  count_leading_zeros(cnt, limb);
  len = numberof (yp) - k;

  /* Normalize it: len = number of last 0 limb, k number of non-zero limbs */
  if (MPFR_LIKELY(cnt))
    mpn_lshift (yp+len, yp, k, cnt);  /* Normalize the High Limb*/
  else if (len != 0)
    MPN_COPY_DECR (yp+len, yp, k);    /* Must use DECR */
  if (len != 0)
    /* Note: when numberof(yp)==1, len is constant and null, so the compiler
       can optimize out this code. */
    {
      if (len == 1)
        yp[0] = (mp_limb_t) 0;
      else
        MPN_ZERO (yp, len);   /* Zeroing the last limbs */
    }
  e += k * GMP_NUMB_BITS - cnt;    /* Update Expo */
  MPFR_ASSERTD (MPFR_LIMB_MSB(yp[numberof (yp) - 1]) != 0);

  /* Check expo underflow / overflow (can't use mpfr_check_range) */
  if (MPFR_UNLIKELY(e < __gmpfr_emin))
    {
      /* The following test is necessary because in the rounding to the
       * nearest mode, mpfr_underflow always rounds away from 0. In
       * this rounding mode, we need to round to 0 if:
       *   _ |x| < 2^(emin-2), or
       *   _ |x| = 2^(emin-2) and the absolute value of the exact
       *     result is <= 2^(emin-2). */
      if (rnd == MPFR_RNDN && (e+1 < __gmpfr_emin || mpfr_powerof2_raw(y)))
        rnd = MPFR_RNDZ;
      return mpfr_underflow (x, rnd, MPFR_SIGN_POS);
    }
  if (MPFR_UNLIKELY(e > __gmpfr_emax))
    return mpfr_overflow (x, rnd, MPFR_SIGN_POS);
  MPFR_SET_EXP (y, e);

  /* Final: set x to y (rounding if necessary) */
  return mpfr_set (x, y, rnd);
}

#endif
