/* mpfr_subnormalize -- Subnormalize a floating point number
   emulating sub-normal numbers.

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

/* For MPFR_RNDN, we can have a problem of double rounding.
   In such a case, this table helps to conclude what to do (y positive):
     Rounding Bit |  Sticky Bit | inexact  | Action    | new inexact
     0            |   ?         |  ?       | Trunc     | sticky
     1            |   0         |  1       | Trunc     |
     1            |   0         |  0       | Trunc if even |
     1            |   0         | -1       | AddOneUlp |
     1            |   1         |  ?       | AddOneUlp |

   For other rounding mode, there isn't such a problem.
   Just round it again and merge the ternary values.

   Set the inexact flag if the returned ternary value is non-zero.
   Set the underflow flag if a second rounding occurred (whether this
   rounding is exact or not). See
     https://sympa.inria.fr/sympa/arc/mpfr/2009-06/msg00000.html
     https://sympa.inria.fr/sympa/arc/mpfr/2009-06/msg00008.html
     https://sympa.inria.fr/sympa/arc/mpfr/2009-06/msg00010.html
*/

int
mpfr_subnormalize (mpfr_ptr y, int old_inexact, mpfr_rnd_t rnd)
{
  int sign;

  /* The subnormal exponent range is [ emin, emin + MPFR_PREC(y) - 2 ] */
  if (MPFR_LIKELY (MPFR_IS_SINGULAR (y)
                   || (MPFR_GET_EXP (y) >=
                       __gmpfr_emin + (mpfr_exp_t) MPFR_PREC (y) - 1)))
    MPFR_RET (old_inexact);

  mpfr_set_underflow ();
  sign = MPFR_SIGN (y);

  /* We have to emulate one bit rounding if EXP(y) = emin */
  if (MPFR_GET_EXP (y) == __gmpfr_emin)
    {
      /* If this is a power of 2, we don't need rounding.
         It handles cases when |y| = 0.1 * 2^emin */
      if (mpfr_powerof2_raw (y))
        MPFR_RET (old_inexact);

      /* We keep the same sign for y.
         Assuming Y is the real value and y the approximation
         and since y is not a power of 2:  0.5*2^emin < Y < 1*2^emin
         We also know the direction of the error thanks to ternary value. */

      if (rnd == MPFR_RNDN)
        {
          mp_limb_t *mant, rb ,sb;
          mp_size_t s;
          /* We need the rounding bit and the sticky bit. Read them
             and use the previous table to conclude. */
          s = MPFR_LIMB_SIZE (y) - 1;
          mant = MPFR_MANT (y) + s;
          rb = *mant & (MPFR_LIMB_HIGHBIT >> 1);
          if (rb == 0)
            goto set_min;
          sb = *mant & ((MPFR_LIMB_HIGHBIT >> 1) - 1);
          while (sb == 0 && s-- != 0)
            sb = *--mant;
          if (sb != 0)
            goto set_min_p1;
          /* Rounding bit is 1 and sticky bit is 0.
             We need to examine old inexact flag to conclude. */
          if ((old_inexact > 0 && sign > 0) ||
              (old_inexact < 0 && sign < 0))
            goto set_min;
          /* If inexact != 0, return 0.1*2^(emin+1).
             Otherwise, rounding bit = 1, sticky bit = 0 and inexact = 0
             So we have 0.1100000000000000000000000*2^emin exactly.
             We return 0.1*2^(emin+1) according to the even-rounding
             rule on subnormals. */
          goto set_min_p1;
        }
      else if (MPFR_IS_LIKE_RNDZ (rnd, MPFR_IS_NEG (y)))
        {
        set_min:
          mpfr_setmin (y, __gmpfr_emin);
          MPFR_RET (-sign);
        }
      else
        {
        set_min_p1:
          /* Note: mpfr_setmin will abort if __gmpfr_emax == __gmpfr_emin. */
          mpfr_setmin (y, __gmpfr_emin + 1);
          MPFR_RET (sign);
        }
    }
  else /* Hard case: It is more or less the same problem than mpfr_cache */
    {
      mpfr_t dest;
      mpfr_prec_t q;
      int inexact, inex2;

      MPFR_ASSERTD (MPFR_GET_EXP (y) > __gmpfr_emin);

      /* Compute the intermediary precision */
      q = (mpfr_uexp_t) MPFR_GET_EXP (y) - __gmpfr_emin + 1;
      MPFR_ASSERTD (q >= MPFR_PREC_MIN && q < MPFR_PREC (y));

      /* TODO: perform the rounding in place. */
      mpfr_init2 (dest, q);
      /* Round y in dest */
      MPFR_SET_EXP (dest, MPFR_GET_EXP (y));
      MPFR_SET_SIGN (dest, sign);
      MPFR_RNDRAW_EVEN (inexact, dest,
                        MPFR_MANT (y), MPFR_PREC (y), rnd, sign,
                        MPFR_SET_EXP (dest, MPFR_GET_EXP (dest) + 1));
      if (MPFR_LIKELY (old_inexact != 0))
        {
          if (MPFR_UNLIKELY (rnd == MPFR_RNDN &&
                             (inexact == MPFR_EVEN_INEX ||
                              inexact == -MPFR_EVEN_INEX)))
            {
              /* if both roundings are in the same direction, we have to go
                 back in the other direction */
              if (SAME_SIGN (inexact, old_inexact))
                {
                  if (SAME_SIGN (inexact, MPFR_INT_SIGN (y)))
                    mpfr_nexttozero (dest);
                  else
                    mpfr_nexttoinf (dest);
                  inexact = -inexact;
                }
            }
          else if (MPFR_UNLIKELY (inexact == 0))
            inexact = old_inexact;
        }

      inex2 = mpfr_set (y, dest, rnd);
      MPFR_ASSERTN (inex2 == 0);
      MPFR_ASSERTN (MPFR_IS_PURE_FP (y));
      mpfr_clear (dest);

      MPFR_RET (inexact);
    }
}
