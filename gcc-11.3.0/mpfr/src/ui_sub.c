/* mpfr_ui_sub -- subtract a floating-point number from an integer

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

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

int
mpfr_ui_sub (mpfr_ptr y, unsigned long int u, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  MPFR_LOG_FUNC
    (("u=%lu x[%Pu]=%.*Rg rnd=%d",
      u, mpfr_get_prec(x), mpfr_log_prec, x, rnd_mode),
     ("y[%Pu]=%.*Rg", mpfr_get_prec(y), mpfr_log_prec, y));

  if (MPFR_UNLIKELY (u == 0))
    return mpfr_neg (y, x, rnd_mode);

  if (MPFR_UNLIKELY(MPFR_IS_SINGULAR(x)))
    {
      if (MPFR_IS_NAN(x))
        {
          MPFR_SET_NAN(y);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF(x))
        {
          /*  u - Inf = -Inf and u - -Inf = +Inf  */
          MPFR_SET_INF(y);
          MPFR_SET_OPPOSITE_SIGN(y,x);
          MPFR_RET(0); /* +/-infinity is exact */
        }
      else /* x is zero */
        /* u - 0 = u */
        return mpfr_set_ui(y, u, rnd_mode);
    }
  else
    {
      mpfr_t uu;
      mp_limb_t up[1];
      int cnt;
      int inex;

      MPFR_SAVE_EXPO_DECL (expo);

      MPFR_TMP_INIT1 (up, uu, GMP_NUMB_BITS);
      MPFR_ASSERTN(u == (mp_limb_t) u);
      count_leading_zeros (cnt, (mp_limb_t) u);
      up[0] = (mp_limb_t) u << cnt;

      /* Optimization note: Exponent save/restore operations may be
         removed if mpfr_sub works even when uu is out-of-range. */
      MPFR_SAVE_EXPO_MARK (expo);
      MPFR_SET_EXP (uu, GMP_NUMB_BITS - cnt);
      inex = mpfr_sub (y, uu, x, rnd_mode);
      MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_check_range(y, inex, rnd_mode);
    }
}
