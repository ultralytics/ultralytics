/* mpfr_rint -- Round to an integer.

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

#include "mpfr-impl.h"

/* Merge the following mpfr_rint code with mpfr_round_raw_generic? */

/* For all the round-to-integer functions, we don't need to extend the
 * exponent range. And it is better not to do so, so that we can test
 * the flag setting for intermediate overflow in the test suite without
 * involving huge non-integer numbers (thus in huge precision). This
 * should also be faster.
 *
 * We also need to be careful with the flags.
 */

int
mpfr_rint (mpfr_ptr r, mpfr_srcptr u, mpfr_rnd_t rnd_mode)
{
  int sign;
  int rnd_away;
  mpfr_exp_t exp;

  if (MPFR_UNLIKELY( MPFR_IS_SINGULAR(u) ))
    {
      if (MPFR_IS_NAN(u))
        {
          MPFR_SET_NAN(r);
          MPFR_RET_NAN;
        }
      MPFR_SET_SAME_SIGN(r, u);
      if (MPFR_IS_INF(u))
        {
          MPFR_SET_INF(r);
          MPFR_RET(0);  /* infinity is exact */
        }
      else /* now u is zero */
        {
          MPFR_ASSERTD(MPFR_IS_ZERO(u));
          MPFR_SET_ZERO(r);
          MPFR_RET(0);  /* zero is exact */
        }
    }
  MPFR_SET_SAME_SIGN (r, u); /* Does nothing if r==u */

  sign = MPFR_INT_SIGN (u);
  exp = MPFR_GET_EXP (u);

  rnd_away =
    rnd_mode == MPFR_RNDD ? sign < 0 :
    rnd_mode == MPFR_RNDU ? sign > 0 :
    rnd_mode == MPFR_RNDZ ? 0        :
    rnd_mode == MPFR_RNDA ? 1        :
    -1; /* round to nearest-even (RNDN) or nearest-away (RNDNA) */

  /* rnd_away:
     1 if round away from zero,
     0 if round to zero,
     -1 if not decided yet.
   */

  if (MPFR_UNLIKELY (exp <= 0))  /* 0 < |u| < 1 ==> round |u| to 0 or 1 */
    {
      /* Note: in the MPFR_RNDN mode, 0.5 must be rounded to 0. */
      if (rnd_away != 0 &&
          (rnd_away > 0 ||
           (exp == 0 && (rnd_mode == MPFR_RNDNA ||
                         !mpfr_powerof2_raw (u)))))
        {
          /* The flags will correctly be set and overflow will correctly
             be handled by mpfr_set_si. */
          mpfr_set_si (r, sign, rnd_mode);
          MPFR_RET(sign > 0 ? 2 : -2);
        }
      else
        {
          MPFR_SET_ZERO(r);  /* r = 0 */
          MPFR_RET(sign > 0 ? -2 : 2);
        }
    }
  else  /* exp > 0, |u| >= 1 */
    {
      mp_limb_t *up, *rp;
      mp_size_t un, rn, ui;
      int sh, idiff;
      int uflags;

      /*
       * uflags will contain:
       *   _ 0 if u is an integer representable in r,
       *   _ 1 if u is an integer not representable in r,
       *   _ 2 if u is not an integer.
       */

      up = MPFR_MANT(u);
      rp = MPFR_MANT(r);

      un = MPFR_LIMB_SIZE(u);
      rn = MPFR_LIMB_SIZE(r);
      MPFR_UNSIGNED_MINUS_MODULO (sh, MPFR_PREC (r));

      /* exp is in the current exponent range: obtained from the input. */
      MPFR_SET_EXP (r, exp); /* Does nothing if r==u */

      if ((exp - 1) / GMP_NUMB_BITS >= un)
        {
          ui = un;
          idiff = 0;
          uflags = 0;  /* u is an integer, representable or not in r */
        }
      else
        {
          mp_size_t uj;

          ui = (exp - 1) / GMP_NUMB_BITS + 1;  /* #limbs of the int part */
          MPFR_ASSERTD (un >= ui);
          uj = un - ui;  /* lowest limb of the integer part */
          idiff = exp % GMP_NUMB_BITS;  /* #int-part bits in up[uj] or 0 */

          uflags = idiff == 0 || (up[uj] << idiff) == 0 ? 0 : 2;
          if (uflags == 0)
            while (uj > 0)
              if (up[--uj] != 0)
                {
                  uflags = 2;
                  break;
                }
        }

      if (ui > rn)
        {
          /* More limbs in the integer part of u than in r.
             Just round u with the precision of r. */
          MPFR_ASSERTD (rp != up && un > rn);
          MPN_COPY (rp, up + (un - rn), rn); /* r != u */
          if (rnd_away < 0)
            {
              /* This is a rounding to nearest mode (MPFR_RNDN or MPFR_RNDNA).
                 Decide the rounding direction here. */
              if (rnd_mode == MPFR_RNDN &&
                  (rp[0] & (MPFR_LIMB_ONE << sh)) == 0)
                { /* halfway cases rounded toward zero */
                  mp_limb_t a, b;
                  /* a: rounding bit and some of the following bits */
                  /* b: boundary for a (weight of the rounding bit in a) */
                  if (sh != 0)
                    {
                      a = rp[0] & ((MPFR_LIMB_ONE << sh) - 1);
                      b = MPFR_LIMB_ONE << (sh - 1);
                    }
                  else
                    {
                      a = up[un - rn - 1];
                      b = MPFR_LIMB_HIGHBIT;
                    }
                  rnd_away = a > b;
                  if (a == b)
                    {
                      mp_size_t i;
                      for (i = un - rn - 1 - (sh == 0); i >= 0; i--)
                        if (up[i] != 0)
                          {
                            rnd_away = 1;
                            break;
                          }
                    }
                }
              else  /* halfway cases rounded away from zero */
                rnd_away =  /* rounding bit */
                  ((sh != 0 && (rp[0] & (MPFR_LIMB_ONE << (sh - 1))) != 0) ||
                   (sh == 0 && (up[un - rn - 1] & MPFR_LIMB_HIGHBIT) != 0));
            }
          if (uflags == 0)
            { /* u is an integer; determine if it is representable in r */
              if (sh != 0 && rp[0] << (GMP_NUMB_BITS - sh) != 0)
                uflags = 1;  /* u is not representable in r */
              else
                {
                  mp_size_t i;
                  for (i = un - rn - 1; i >= 0; i--)
                    if (up[i] != 0)
                      {
                        uflags = 1;  /* u is not representable in r */
                        break;
                      }
                }
            }
        }
      else  /* ui <= rn */
        {
          mp_size_t uj, rj;
          int ush;

          uj = un - ui;  /* lowest limb of the integer part in u */
          rj = rn - ui;  /* lowest limb of the integer part in r */

          if (MPFR_LIKELY (rp != up))
            MPN_COPY(rp + rj, up + uj, ui);

          /* Ignore the lowest rj limbs, all equal to zero. */
          rp += rj;
          rn = ui;

          /* number of fractional bits in whole rp[0] */
          ush = idiff == 0 ? 0 : GMP_NUMB_BITS - idiff;

          if (rj == 0 && ush < sh)
            {
              /* If u is an integer (uflags == 0), we need to determine
                 if it is representable in r, i.e. if its sh - ush bits
                 in the non-significant part of r are all 0. */
              if (uflags == 0 && (rp[0] & ((MPFR_LIMB_ONE << sh) -
                                           (MPFR_LIMB_ONE << ush))) != 0)
                uflags = 1;  /* u is an integer not representable in r */
            }
          else  /* The integer part of u fits in r, we'll round to it. */
            sh = ush;

          if (rnd_away < 0)
            {
              /* This is a rounding to nearest mode.
                 Decide the rounding direction here. */
              if (uj == 0 && sh == 0)
                rnd_away = 0; /* rounding bit = 0 (not represented in u) */
              else if (rnd_mode == MPFR_RNDN &&
                       (rp[0] & (MPFR_LIMB_ONE << sh)) == 0)
                { /* halfway cases rounded toward zero */
                  mp_limb_t a, b;
                  /* a: rounding bit and some of the following bits */
                  /* b: boundary for a (weight of the rounding bit in a) */
                  if (sh != 0)
                    {
                      a = rp[0] & ((MPFR_LIMB_ONE << sh) - 1);
                      b = MPFR_LIMB_ONE << (sh - 1);
                    }
                  else
                    {
                      MPFR_ASSERTD (uj >= 1);  /* see above */
                      a = up[uj - 1];
                      b = MPFR_LIMB_HIGHBIT;
                    }
                  rnd_away = a > b;
                  if (a == b)
                    {
                      mp_size_t i;
                      for (i = uj - 1 - (sh == 0); i >= 0; i--)
                        if (up[i] != 0)
                          {
                            rnd_away = 1;
                            break;
                          }
                    }
                }
              else  /* halfway cases rounded away from zero */
                rnd_away =  /* rounding bit */
                  ((sh != 0 && (rp[0] & (MPFR_LIMB_ONE << (sh - 1))) != 0) ||
                   (sh == 0 && (MPFR_ASSERTD (uj >= 1),
                                up[uj - 1] & MPFR_LIMB_HIGHBIT) != 0));
            }
          /* Now we can make the low rj limbs to 0 */
          MPN_ZERO (rp-rj, rj);
        }

      if (sh != 0)
        rp[0] &= MP_LIMB_T_MAX << sh;

      /* If u is a representable integer, there is no rounding. */
      if (uflags == 0)
        MPFR_RET(0);

      MPFR_ASSERTD (rnd_away >= 0);  /* rounding direction is defined */
      if (rnd_away && mpn_add_1(rp, rp, rn, MPFR_LIMB_ONE << sh))
        {
          if (exp == __gmpfr_emax)
            return mpfr_overflow (r, rnd_mode, sign) >= 0 ?
              uflags : -uflags;
          else  /* no overflow */
            {
              MPFR_SET_EXP(r, exp + 1);
              rp[rn-1] = MPFR_LIMB_HIGHBIT;
            }
        }

      MPFR_RET (rnd_away ^ (sign < 0) ? uflags : -uflags);
    }  /* exp > 0, |u| >= 1 */
}

#undef mpfr_round

int
mpfr_round (mpfr_ptr r, mpfr_srcptr u)
{
  return mpfr_rint (r, u, MPFR_RNDNA);
}

#undef mpfr_trunc

int
mpfr_trunc (mpfr_ptr r, mpfr_srcptr u)
{
  return mpfr_rint (r, u, MPFR_RNDZ);
}

#undef mpfr_ceil

int
mpfr_ceil (mpfr_ptr r, mpfr_srcptr u)
{
  return mpfr_rint (r, u, MPFR_RNDU);
}

#undef mpfr_floor

int
mpfr_floor (mpfr_ptr r, mpfr_srcptr u)
{
  return mpfr_rint (r, u, MPFR_RNDD);
}

/* We need to save the flags and restore them after calling the mpfr_round,
 * mpfr_trunc, mpfr_ceil, mpfr_floor functions because these functions set
 * the inexact flag when the argument is not an integer.
 */

#undef mpfr_rint_round

int
mpfr_rint_round (mpfr_ptr r, mpfr_srcptr u, mpfr_rnd_t rnd_mode)
{
  if (MPFR_UNLIKELY( MPFR_IS_SINGULAR(u) ) || mpfr_integer_p (u))
    return mpfr_set (r, u, rnd_mode);
  else
    {
      mpfr_t tmp;
      int inex;
      unsigned int saved_flags = __gmpfr_flags;
      MPFR_BLOCK_DECL (flags);

      mpfr_init2 (tmp, MPFR_PREC (u));
      /* round(u) is representable in tmp unless an overflow occurs */
      MPFR_BLOCK (flags, mpfr_round (tmp, u));
      __gmpfr_flags = saved_flags;
      inex = (MPFR_OVERFLOW (flags)
              ? mpfr_overflow (r, rnd_mode, MPFR_SIGN (u))
              : mpfr_set (r, tmp, rnd_mode));
      mpfr_clear (tmp);
      return inex;
    }
}

#undef mpfr_rint_trunc

int
mpfr_rint_trunc (mpfr_ptr r, mpfr_srcptr u, mpfr_rnd_t rnd_mode)
{
  if (MPFR_UNLIKELY( MPFR_IS_SINGULAR(u) ) || mpfr_integer_p (u))
    return mpfr_set (r, u, rnd_mode);
  else
    {
      mpfr_t tmp;
      int inex;
      unsigned int saved_flags = __gmpfr_flags;

      mpfr_init2 (tmp, MPFR_PREC (u));
      /* trunc(u) is always representable in tmp */
      mpfr_trunc (tmp, u);
      __gmpfr_flags = saved_flags;
      inex = mpfr_set (r, tmp, rnd_mode);
      mpfr_clear (tmp);
      return inex;
    }
}

#undef mpfr_rint_ceil

int
mpfr_rint_ceil (mpfr_ptr r, mpfr_srcptr u, mpfr_rnd_t rnd_mode)
{
  if (MPFR_UNLIKELY( MPFR_IS_SINGULAR(u) ) || mpfr_integer_p (u))
    return mpfr_set (r, u, rnd_mode);
  else
    {
      mpfr_t tmp;
      int inex;
      unsigned int saved_flags = __gmpfr_flags;
      MPFR_BLOCK_DECL (flags);

      mpfr_init2 (tmp, MPFR_PREC (u));
      /* ceil(u) is representable in tmp unless an overflow occurs */
      MPFR_BLOCK (flags, mpfr_ceil (tmp, u));
      __gmpfr_flags = saved_flags;
      inex = (MPFR_OVERFLOW (flags)
              ? mpfr_overflow (r, rnd_mode, MPFR_SIGN_POS)
              : mpfr_set (r, tmp, rnd_mode));
      mpfr_clear (tmp);
      return inex;
    }
}

#undef mpfr_rint_floor

int
mpfr_rint_floor (mpfr_ptr r, mpfr_srcptr u, mpfr_rnd_t rnd_mode)
{
  if (MPFR_UNLIKELY( MPFR_IS_SINGULAR(u) ) || mpfr_integer_p (u))
    return mpfr_set (r, u, rnd_mode);
  else
    {
      mpfr_t tmp;
      int inex;
      unsigned int saved_flags = __gmpfr_flags;
      MPFR_BLOCK_DECL (flags);

      mpfr_init2 (tmp, MPFR_PREC (u));
      /* floor(u) is representable in tmp unless an overflow occurs */
      MPFR_BLOCK (flags, mpfr_floor (tmp, u));
      __gmpfr_flags = saved_flags;
      inex = (MPFR_OVERFLOW (flags)
              ? mpfr_overflow (r, rnd_mode, MPFR_SIGN_NEG)
              : mpfr_set (r, tmp, rnd_mode));
      mpfr_clear (tmp);
      return inex;
    }
}
