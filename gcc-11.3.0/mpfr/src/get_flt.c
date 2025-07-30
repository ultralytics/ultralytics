/* mpfr_get_flt -- convert a mpfr_t to a machine single precision float

Copyright 2009-2017 Free Software Foundation, Inc.
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

#include <float.h>     /* for FLT_MIN */

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

#include "ieee_floats.h"

#define FLT_NEG_ZERO ((float) DBL_NEG_ZERO)
#define MPFR_FLT_INFM ((float) MPFR_DBL_INFM)
#define MPFR_FLT_INFP ((float) MPFR_DBL_INFP)

float
mpfr_get_flt (mpfr_srcptr src, mpfr_rnd_t rnd_mode)
{
  int negative;
  mpfr_exp_t e;
  float d;

  /* in case of NaN, +Inf, -Inf, +0, -0, the conversion from double to float
     is exact */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (src)))
    return (float) mpfr_get_d (src, rnd_mode);

  e = MPFR_GET_EXP (src);
  negative = MPFR_IS_NEG (src);

  if (MPFR_UNLIKELY(rnd_mode == MPFR_RNDA))
    rnd_mode = negative ? MPFR_RNDD : MPFR_RNDU;

  /* FIXME: The code below assumes the IEEE-754 binary32 format
     with subnormal support. */

  /* the smallest positive normal float number is 2^(-126) = 0.5*2^(-125),
     and the smallest positive subnormal number is 2^(-149) = 0.5*2^(-148) */
  if (MPFR_UNLIKELY (e < -148))
    {
      /* |src| < 2^(-149), i.e., |src| is smaller than the smallest positive
         subnormal number.
         In round-to-nearest mode, 2^(-150) is rounded to zero.
      */
      d = negative ?
        (rnd_mode == MPFR_RNDD ||
         (rnd_mode == MPFR_RNDN && mpfr_cmp_si_2exp (src, -1, -150) < 0)
         ? -FLT_MIN : FLT_NEG_ZERO) :
        (rnd_mode == MPFR_RNDU ||
         (rnd_mode == MPFR_RNDN && mpfr_cmp_si_2exp (src, 1, -150) > 0)
         ? FLT_MIN : 0.0);
      if (d != 0.0) /* we multiply FLT_MIN = 2^(-126) by FLT_EPSILON = 2^(-23)
                       to get +-2^(-149) */
        d *= FLT_EPSILON;
    }
  /* the largest normal number is 2^128*(1-2^(-24)) = 0.111...111e128 */
  else if (MPFR_UNLIKELY (e > 128))
    {
      d = negative ?
        (rnd_mode == MPFR_RNDZ || rnd_mode == MPFR_RNDU ?
         -FLT_MAX : MPFR_FLT_INFM) :
        (rnd_mode == MPFR_RNDZ || rnd_mode == MPFR_RNDD ?
         FLT_MAX : MPFR_FLT_INFP);
    }
  else /* -148 <= e <= 127 */
    {
      int nbits;
      mp_size_t np, i;
      mp_limb_t tp[MPFR_LIMBS_PER_FLT];
      int carry;
      double dd;

      nbits = IEEE_FLT_MANT_DIG; /* 24 */
      if (MPFR_UNLIKELY (e < -125))
        /*In the subnormal case, compute the exact number of significant bits*/
        {
          nbits += (125 + e);
          MPFR_ASSERTD (nbits >= 1);
        }
      np = MPFR_PREC2LIMBS (nbits);
      MPFR_ASSERTD(np <= MPFR_LIMBS_PER_FLT);
      carry = mpfr_round_raw_4 (tp, MPFR_MANT(src), MPFR_PREC(src), negative,
                                nbits, rnd_mode);
      /* we perform the reconstruction using the 'double' type here,
         knowing the result is exactly representable as 'float' */
      if (MPFR_UNLIKELY(carry))
        dd = 1.0;
      else
        {
          /* The following computations are exact thanks to the previous
             mpfr_round_raw. */
          dd = (double) tp[0] / MP_BASE_AS_DOUBLE;
          for (i = 1 ; i < np ; i++)
            dd = (dd + tp[i]) / MP_BASE_AS_DOUBLE;
          /* dd is the mantissa (between 1/2 and 1) of the argument rounded
             to 24 bits */
        }
      dd = mpfr_scale2 (dd, e);
      if (negative)
        dd = -dd;

      /* convert (exacly) to float */
      d = (float) dd;
    }

  return d;
}

