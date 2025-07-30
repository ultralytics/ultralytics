/* mpfr_add_ui -- add a floating-point number with a machine integer

Copyright 2000-2004, 2006-2017 Free Software Foundation, Inc.
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
mpfr_add_ui (mpfr_ptr y, mpfr_srcptr x, unsigned long int u, mpfr_rnd_t rnd_mode)
{
  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg u=%lu rnd=%d",
      mpfr_get_prec(x), mpfr_log_prec, x, u, rnd_mode),
     ("y[%Pu]=%.*Rg", mpfr_get_prec (y), mpfr_log_prec, y));

  if (MPFR_LIKELY(u != 0) )  /* if u=0, do nothing */
    {
      mpfr_t uu;
      mp_limb_t up[1];
      unsigned long cnt;
      int inex;
      MPFR_SAVE_EXPO_DECL (expo);

      MPFR_TMP_INIT1 (up, uu, GMP_NUMB_BITS);
      MPFR_ASSERTD (u == (mp_limb_t) u);
      count_leading_zeros(cnt, (mp_limb_t) u);
      up[0] = (mp_limb_t) u << cnt;

      /* Optimization note: Exponent save/restore operations may be
         removed if mpfr_add works even when uu is out-of-range. */
      MPFR_SAVE_EXPO_MARK (expo);
      MPFR_SET_EXP (uu, GMP_NUMB_BITS - cnt);
      inex = mpfr_add(y, x, uu, rnd_mode);
      MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_check_range(y, inex, rnd_mode);
    }
  else
    /* (unsigned long) 0 is assumed to be a real 0 (unsigned) */
    return mpfr_set (y, x, rnd_mode);
}
