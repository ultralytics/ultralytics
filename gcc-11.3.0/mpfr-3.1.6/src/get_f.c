/* mpfr_get_f -- convert a MPFR number to a GNU MPF number

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

/* Since MPFR-3.0, return the usual inexact value.
   The erange flag is set if an error occurred in the conversion
   (y is NaN, +Inf, or -Inf that have no equivalent in mpf)
*/
int
mpfr_get_f (mpf_ptr x, mpfr_srcptr y, mpfr_rnd_t rnd_mode)
{
  int inex;
  mp_size_t sx, sy;
  mpfr_prec_t precx, precy;
  mp_limb_t *xp;
  int sh;

  if (MPFR_UNLIKELY(MPFR_IS_SINGULAR(y)))
    {
      if (MPFR_IS_ZERO(y))
        {
          mpf_set_ui (x, 0);
          return 0;
        }
      else if (MPFR_IS_NAN (y))
        {
          MPFR_SET_ERANGE ();
          return 0;
        }
      else /* y is plus infinity (resp. minus infinity), set x to the maximum
              value (resp. the minimum value) in precision PREC(x) */
        {
          int i;
          mp_limb_t *xp;

          MPFR_SET_ERANGE ();

          /* To this day, [mp_exp_t] and mp_size_t are #defined as the same
             type */
          EXP (x) = MP_SIZE_T_MAX;

          sx = PREC (x);
          SIZ (x) = sx;
          xp = PTR (x);
          for (i = 0; i < sx; i++)
            xp[i] = MP_LIMB_T_MAX;

          if (MPFR_IS_POS (y))
            return -1;
          else
            {
              mpf_neg (x, x);
              return +1;
            }
        }
    }

  sx = PREC(x); /* number of limbs of the mantissa of x */

  precy = MPFR_PREC(y);
  precx = (mpfr_prec_t) sx * GMP_NUMB_BITS;
  sy = MPFR_LIMB_SIZE (y);

  xp = PTR (x);

  /* since mpf numbers are represented in base 2^GMP_NUMB_BITS,
     we loose -EXP(y) % GMP_NUMB_BITS bits in the most significant limb */
  sh = MPFR_GET_EXP(y) % GMP_NUMB_BITS;
  sh = sh <= 0 ? - sh : GMP_NUMB_BITS - sh;
  MPFR_ASSERTD (sh >= 0);
  if (precy + sh <= precx) /* we can copy directly */
    {
      mp_size_t ds;

      MPFR_ASSERTN (sx >= sy);
      ds = sx - sy;

      if (sh != 0)
        {
          mp_limb_t out;
          out = mpn_rshift (xp + ds, MPFR_MANT(y), sy, sh);
          MPFR_ASSERTN (ds > 0 || out == 0);
          if (ds > 0)
            xp[--ds] = out;
        }
      else
        MPN_COPY (xp + ds, MPFR_MANT (y), sy);
      if (ds > 0)
        MPN_ZERO (xp, ds);
      EXP(x) = (MPFR_GET_EXP(y) + sh) / GMP_NUMB_BITS;
      inex = 0;
    }
  else /* we have to round to precx - sh bits */
    {
      mpfr_t z;
      mp_size_t sz;

      /* Recall that precx = (mpfr_prec_t) sx * GMP_NUMB_BITS, thus removing
         sh bits (sh < GMP_NUMB_BITSS) won't reduce the number of limbs. */
      mpfr_init2 (z, precx - sh);
      sz = MPFR_LIMB_SIZE (z);
      MPFR_ASSERTN (sx == sz);

      inex = mpfr_set (z, y, rnd_mode);
      /* warning, sh may change due to rounding, but then z is a power of two,
         thus we can safely ignore its last bit which is 0 */
      sh = MPFR_GET_EXP(z) % GMP_NUMB_BITS;
      sh = sh <= 0 ? - sh : GMP_NUMB_BITS - sh;
      MPFR_ASSERTD (sh >= 0);
      if (sh != 0)
        {
          mp_limb_t out;
          out = mpn_rshift (xp, MPFR_MANT(z), sz, sh);
          /* If sh hasn't changed, it is the number of the non-significant
             bits in the lowest limb of z. Therefore out == 0. */
          MPFR_ASSERTD (out == 0);  (void) out; /* avoid a warning */
        }
      else
        MPN_COPY (xp, MPFR_MANT(z), sz);
      EXP(x) = (MPFR_GET_EXP(z) + sh) / GMP_NUMB_BITS;
      mpfr_clear (z);
    }

  /* set size and sign */
  SIZ(x) = (MPFR_FROM_SIGN_TO_INT(MPFR_SIGN(y)) < 0) ? -sx : sx;

  return inex;
}
