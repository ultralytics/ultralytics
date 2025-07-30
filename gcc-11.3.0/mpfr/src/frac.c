/* mpfr_frac -- Fractional part of a floating-point number.

Copyright 2002-2004, 2006-2017 Free Software Foundation, Inc.
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


#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

/* Optimization note: it is not a good idea to call mpfr_integer_p,
   as some cases will take longer (the number may be parsed twice). */

int
mpfr_frac (mpfr_ptr r, mpfr_srcptr u, mpfr_rnd_t rnd_mode)
{
  mpfr_exp_t re, ue;
  mpfr_prec_t uq;
  mp_size_t un, tn, t0;
  mp_limb_t *up, *tp, k;
  int sh;
  mpfr_t tmp;
  mpfr_ptr t;
  int inex;
  MPFR_SAVE_EXPO_DECL (expo);

  /* Special cases */
  if (MPFR_UNLIKELY(MPFR_IS_NAN(u)))
    {
      MPFR_SET_NAN(r);
      MPFR_RET_NAN;
    }
  else if (MPFR_UNLIKELY(MPFR_IS_INF(u) || mpfr_integer_p (u)))
    {
      MPFR_SET_SAME_SIGN(r, u);
      MPFR_SET_ZERO(r);
      MPFR_RET(0);  /* zero is exact */
    }

  ue = MPFR_GET_EXP (u);
  if (ue <= 0)  /* |u| < 1 */
    return mpfr_set (r, u, rnd_mode);

  /* Now |u| >= 1, meaning that an overflow is not possible. */

  uq = MPFR_PREC(u);
  un = (uq - 1) / GMP_NUMB_BITS;  /* index of most significant limb */
  un -= (mp_size_t) (ue / GMP_NUMB_BITS);
  /* now the index of the MSL containing bits of the fractional part */

  up = MPFR_MANT(u);
  sh = ue % GMP_NUMB_BITS;
  k = up[un] << sh;
  /* the first bit of the fractional part is the MSB of k */

  if (k != 0)
    {
      int cnt;

      count_leading_zeros(cnt, k);
      /* first bit 1 of the fractional part -> MSB of the number */
      re = -cnt;
      sh += cnt;
      MPFR_ASSERTN (sh < GMP_NUMB_BITS);
      k <<= cnt;
    }
  else
    {
      re = sh - GMP_NUMB_BITS;
      /* searching for the first bit 1 (exists since u isn't an integer) */
      while (up[--un] == 0)
        re -= GMP_NUMB_BITS;
      MPFR_ASSERTN(un >= 0);
      k = up[un];
      count_leading_zeros(sh, k);
      re -= sh;
      k <<= sh;
    }
  /* The exponent of r will be re */
  /* un: index of the limb of u that contains the first bit 1 of the FP */

  t = (mp_size_t) (MPFR_PREC(r) - 1) / GMP_NUMB_BITS < un ?
    (mpfr_init2 (tmp, (un + 1) * GMP_NUMB_BITS), tmp) : r;
  /* t has enough precision to contain the fractional part of u */
  /* If we use a temporary variable, we take the non-significant bits
     of u into account, because of the mpn_lshift below. */
  MPFR_SET_SAME_SIGN(t, u);

  /* Put the fractional part of u into t */
  tn = (MPFR_PREC(t) - 1) / GMP_NUMB_BITS;
  MPFR_ASSERTN(tn >= un);
  t0 = tn - un;
  tp = MPFR_MANT(t);
  if (sh == 0)
    MPN_COPY_DECR(tp + t0, up, un + 1);
  else /* warning: un may be 0 here */
    tp[tn] = k | ((un) ? mpn_lshift (tp + t0, up, un, sh) : (mp_limb_t) 0);
  if (t0 > 0)
    MPN_ZERO(tp, t0);

  MPFR_SAVE_EXPO_MARK (expo);

  if (t != r)
    { /* t is tmp */
      MPFR_EXP (t) = 0;  /* should be re, but not necessarily in the range */
      inex = mpfr_set (r, t, rnd_mode);  /* no underflow */
      mpfr_clear (t);
      MPFR_EXP (r) += re;
    }
  else
    { /* There may be remaining non-significant bits in t (= r). */
      int carry;

      MPFR_EXP (r) = re;
      carry = mpfr_round_raw (tp, tp,
                              (mpfr_prec_t) (tn + 1) * GMP_NUMB_BITS,
                              MPFR_IS_NEG (r), MPFR_PREC (r), rnd_mode,
                              &inex);
      if (carry)
        {
          tp[tn] = MPFR_LIMB_HIGHBIT;
          MPFR_EXP (r) ++;
        }
    }

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (r, inex, rnd_mode);
}
