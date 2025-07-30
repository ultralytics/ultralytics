/* generic inverse of a function.

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

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

#ifndef ACTION_SPECIAL
#define ACTION_SPECIAL
#endif

#ifndef ACTION_TINY
#define ACTION_TINY
#endif

/* example of use:
#define FUNCTION mpfr_sec
#define INVERSE  mpfr_cos
#define ACTION_NAN(y) do { MPFR_SET_NAN(y); MPFR_RET_NAN; } while (1)
#define ACTION_INF(y) do { MPFR_SET_NAN(y); MPFR_RET_NAN; } while (1)
#define ACTION_ZERO(y) return mpfr_set_ui (y, 1, MPFR_RNDN)
#include "gen_inverse.h"
*/

int
FUNCTION (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_prec_t precy; /* target precision */
  mpfr_prec_t m;     /* working precision */
  mpfr_t z;        /* temporary variable to store INVERSE(x) */
  int inexact;     /* inexact flag */
  MPFR_ZIV_DECL (loop);
  MPFR_SAVE_EXPO_DECL (expo);

  if (MPFR_UNLIKELY(MPFR_IS_SINGULAR(x)))
    {
      if (MPFR_IS_NAN(x))
        ACTION_NAN(y);
      else if (MPFR_IS_INF(x))
        ACTION_INF(y);
      else /* x = 0 */
        ACTION_ZERO(y,x);
    }

  /* x is neither NaN, Inf nor zero */
  MPFR_SAVE_EXPO_MARK (expo);
  ACTION_TINY (y, x, rnd_mode); /* special case for very small input x */
  precy = MPFR_PREC(y);
  m = precy + MPFR_INT_CEIL_LOG2 (precy) + 3;
  mpfr_init2 (z, m);

  MPFR_ZIV_INIT (loop, m);
  for(;;)
    {
      MPFR_BLOCK_DECL (flags);

      MPFR_BLOCK (flags, INVERSE (z, x, MPFR_RNDZ)); /* error k_u < 1 ulp */
      /* FIXME: the following assumes that if an overflow happens with
         MPFR_EMAX_MAX, then necessarily an underflow happens with
         __gmpfr_emin */
      if (MPFR_OVERFLOW (flags))
        {
          int s = MPFR_SIGN(z);
          MPFR_ZIV_FREE (loop);
          mpfr_clear (z);
          MPFR_SAVE_EXPO_FREE (expo);
          return mpfr_underflow (y, (rnd_mode == MPFR_RNDN) ?
                                 MPFR_RNDZ : rnd_mode, s);
        }
      mpfr_ui_div (z, 1, z, MPFR_RNDN);
      /* the error is less than c_w + 2*c_u*k_u (see algorithms.tex),
         where c_w = 1/2, c_u = 1 since z was rounded toward zero,
         thus 1/2 + 2 < 4 */
      if (MPFR_LIKELY (MPFR_CAN_ROUND (z, m - 2, precy, rnd_mode)))
        break;
      ACTION_SPECIAL;
      MPFR_ZIV_NEXT (loop, m);
      mpfr_set_prec (z, m);
    }
  MPFR_ZIV_FREE (loop);

  inexact = mpfr_set (y, z, rnd_mode);
  mpfr_clear (z);

 end:
  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, inexact, rnd_mode);
}
