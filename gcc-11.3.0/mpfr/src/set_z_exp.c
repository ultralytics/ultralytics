/* mpfr_set_z_2exp -- set a floating-point number from a multiple-precision
                      integer and an exponent

Copyright 1999-2017 Free Software Foundation, Inc.
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

/* set f to the integer z multiplied by 2^e */
int
mpfr_set_z_2exp (mpfr_ptr f, mpz_srcptr z, mpfr_exp_t e, mpfr_rnd_t rnd_mode)
{
  mp_size_t fn, zn, dif, en;
  int k, sign_z, inex;
  mp_limb_t *fp, *zp;
  mpfr_exp_t exp;

  sign_z = mpz_sgn (z);
  if (MPFR_UNLIKELY (sign_z == 0)) /* ignore the exponent for 0 */
    {
      MPFR_SET_ZERO(f);
      MPFR_SET_POS(f);
      MPFR_RET(0);
    }
  MPFR_ASSERTD (sign_z == MPFR_SIGN_POS || sign_z == MPFR_SIGN_NEG);

  zn = ABS(SIZ(z)); /* limb size of z */
  /* compute en = floor(e/GMP_NUMB_BITS) */
  en = (e >= 0) ? e / GMP_NUMB_BITS : (e + 1) / GMP_NUMB_BITS - 1;
  MPFR_ASSERTD (zn >= 1);
  if (MPFR_UNLIKELY (zn + en > MPFR_EMAX_MAX / GMP_NUMB_BITS + 1))
    return mpfr_overflow (f, rnd_mode, sign_z);
  /* because zn + en >= MPFR_EMAX_MAX / GMP_NUMB_BITS + 2
     implies (zn + en) * GMP_NUMB_BITS >= MPFR_EMAX_MAX + GMP_NUMB_BITS + 1
     and exp = zn * GMP_NUMB_BITS + e - k
             >= (zn + en) * GMP_NUMB_BITS - k > MPFR_EMAX_MAX */

  fp = MPFR_MANT (f);
  fn = MPFR_LIMB_SIZE (f);
  dif = zn - fn;
  zp = PTR(z);
  count_leading_zeros (k, zp[zn-1]);

  /* now zn + en <= MPFR_EMAX_MAX / GMP_NUMB_BITS + 1
     thus (zn + en) * GMP_NUMB_BITS <= MPFR_EMAX_MAX + GMP_NUMB_BITS
     and exp = zn * GMP_NUMB_BITS + e - k
             <= (zn + en) * GMP_NUMB_BITS - k + GMP_NUMB_BITS - 1
             <= MPFR_EMAX_MAX + 2 * GMP_NUMB_BITS - 1 */
  exp = (mpfr_prec_t) zn * GMP_NUMB_BITS + e - k;
  /* The exponent will be exp or exp + 1 (due to rounding) */
  if (MPFR_UNLIKELY (exp > __gmpfr_emax))
    return mpfr_overflow (f, rnd_mode, sign_z);
  if (MPFR_UNLIKELY (exp + 1 < __gmpfr_emin))
    return mpfr_underflow (f, rnd_mode == MPFR_RNDN ? MPFR_RNDZ : rnd_mode,
                           sign_z);

  if (MPFR_LIKELY (dif >= 0))
    {
      mp_limb_t rb, sb, ulp;
      int sh;

      /* number has to be truncated */
      if (MPFR_LIKELY (k != 0))
        {
          mpn_lshift (fp, &zp[dif], fn, k);
          if (MPFR_LIKELY (dif > 0))
            fp[0] |= zp[dif - 1] >> (GMP_NUMB_BITS - k);
        }
      else
        MPN_COPY (fp, zp + dif, fn);

      /* Compute Rounding Bit and Sticky Bit */
      MPFR_UNSIGNED_MINUS_MODULO (sh, MPFR_PREC (f) );
      if (MPFR_LIKELY (sh != 0))
        {
          mp_limb_t mask = MPFR_LIMB_ONE << (sh-1);
          mp_limb_t limb = fp[0];
          rb = limb & mask;
          sb = limb & (mask-1);
          ulp = 2*mask;
          fp[0] = limb & ~(ulp-1);
        }
      else /* sh == 0 */
        {
          mp_limb_t mask = MPFR_LIMB_ONE << (GMP_NUMB_BITS - 1 - k);
          if (MPFR_LIKELY (dif > 0))
            {
              rb = zp[--dif] & mask;
              sb = zp[dif] & (mask-1);
            }
          else
            rb = sb = 0;
          k = 0;
          ulp = MPFR_LIMB_ONE;
        }
      if (MPFR_UNLIKELY (sb == 0) && MPFR_LIKELY (dif > 0))
        {
          sb = zp[--dif];
          if (MPFR_LIKELY (k != 0))
            sb &= MPFR_LIMB_MASK (GMP_NUMB_BITS - k);
          if (MPFR_UNLIKELY (sb == 0) && MPFR_LIKELY (dif > 0))
            do {
              sb = zp[--dif];
            } while (dif > 0 && sb == 0);
        }

      /* Rounding */
      if (MPFR_LIKELY (rnd_mode == MPFR_RNDN))
        {
          if (rb == 0 || MPFR_UNLIKELY (sb == 0 && (fp[0] & ulp) == 0))
            goto trunc;
          else
            goto addoneulp;
        }
      else /* Not Nearest */
        {
          if (MPFR_LIKELY (MPFR_IS_LIKE_RNDZ (rnd_mode, sign_z < 0))
              || MPFR_UNLIKELY ( (sb | rb) == 0 ))
            goto trunc;
          else
            goto addoneulp;
        }

    trunc:
      inex = MPFR_LIKELY ((sb | rb) != 0) ? -1 : 0;
      goto end;

    addoneulp:
      inex = 1;
      if (MPFR_UNLIKELY (mpn_add_1 (fp, fp, fn, ulp)))
        {
          /* Pow 2 case */
          if (MPFR_UNLIKELY (exp == __gmpfr_emax))
            return mpfr_overflow (f, rnd_mode, sign_z);
          exp ++;
          fp[fn-1] = MPFR_LIMB_HIGHBIT;
        }
    end:
      (void) 0;
    }
  else   /* dif < 0: Mantissa F is strictly bigger than z's one */
    {
      if (MPFR_LIKELY (k != 0))
        mpn_lshift (fp - dif, zp, zn, k);
      else
        MPN_COPY (fp - dif, zp, zn);
      /* fill with zeroes */
      MPN_ZERO (fp, -dif);
      inex = 0; /* result is exact */
    }

  if (MPFR_UNLIKELY (exp < __gmpfr_emin))
    {
      if (rnd_mode == MPFR_RNDN && inex == 0 && mpfr_powerof2_raw (f))
        rnd_mode = MPFR_RNDZ;
      return mpfr_underflow (f, rnd_mode, sign_z);
    }

  MPFR_SET_EXP (f, exp);
  MPFR_SET_SIGN (f, sign_z);
  MPFR_RET (inex*sign_z);
}
