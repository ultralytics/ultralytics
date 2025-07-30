/* mpfr_set_ld -- convert a machine long double to
                  a multiple precision floating-point number

Copyright 2002-2017 Free Software Foundation, Inc.
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

#include <float.h>

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

/* Various i386 systems have been seen with <float.h> LDBL constants equal
   to the DBL ones, whereas they ought to be bigger, reflecting the 10-byte
   IEEE extended format on that processor.  gcc 3.2.1 on FreeBSD and Solaris
   has been seen with the problem, and gcc 2.95.4 on FreeBSD 4.7.  */

#if HAVE_LDOUBLE_IEEE_EXT_LITTLE
static const union {
  char         bytes[10];
  long double  d;
} ldbl_max_struct = {
  { '\377','\377','\377','\377',
    '\377','\377','\377','\377',
    '\376','\177' }
};
#define MPFR_LDBL_MAX   (ldbl_max_struct.d)
#else
#define MPFR_LDBL_MAX   LDBL_MAX
#endif

#ifndef HAVE_LDOUBLE_IEEE_EXT_LITTLE

/* Generic code */
int
mpfr_set_ld (mpfr_ptr r, long double d, mpfr_rnd_t rnd_mode)
{
  mpfr_t t, u;
  int inexact, shift_exp;
  long double x;
  MPFR_SAVE_EXPO_DECL (expo);

  /* Check for NAN */
  LONGDOUBLE_NAN_ACTION (d, goto nan);

  /* Check for INF */
  if (d > MPFR_LDBL_MAX)
    {
      mpfr_set_inf (r, 1);
      return 0;
    }
  else if (d < -MPFR_LDBL_MAX)
    {
      mpfr_set_inf (r, -1);
      return 0;
    }
  /* Check for ZERO */
  else if (d == 0.0)
    return mpfr_set_d (r, (double) d, rnd_mode);

  mpfr_init2 (t, MPFR_LDBL_MANT_DIG);
  mpfr_init2 (u, IEEE_DBL_MANT_DIG);

  MPFR_SAVE_EXPO_MARK (expo);

 convert:
  x = d;
  MPFR_SET_ZERO (t);  /* The sign doesn't matter. */
  shift_exp = 0; /* invariant: remainder to deal with is d*2^shift_exp */
  while (x != (long double) 0.0)
    {
      /* Check overflow of double */
      if (x > (long double) DBL_MAX || (-x) > (long double) DBL_MAX)
        {
          long double div9, div10, div11, div12, div13;

#define TWO_64 18446744073709551616.0 /* 2^64 */
#define TWO_128 (TWO_64 * TWO_64)
#define TWO_256 (TWO_128 * TWO_128)
          div9 = (long double) (double) (TWO_256 * TWO_256); /* 2^(2^9) */
          div10 = div9 * div9;
          div11 = div10 * div10; /* 2^(2^11) */
          div12 = div11 * div11; /* 2^(2^12) */
          div13 = div12 * div12; /* 2^(2^13) */
          if (ABS (x) >= div13)
            {
              x /= div13; /* exact */
              shift_exp += 8192;
              mpfr_div_2si (t, t, 8192, MPFR_RNDZ);
            }
          if (ABS (x) >= div12)
            {
              x /= div12; /* exact */
              shift_exp += 4096;
              mpfr_div_2si (t, t, 4096, MPFR_RNDZ);
            }
          if (ABS (x) >= div11)
            {
              x /= div11; /* exact */
              shift_exp += 2048;
              mpfr_div_2si (t, t, 2048, MPFR_RNDZ);
            }
          if (ABS (x) >= div10)
            {
              x /= div10; /* exact */
              shift_exp += 1024;
              mpfr_div_2si (t, t, 1024, MPFR_RNDZ);
            }
          /* warning: we may have DBL_MAX=2^1024*(1-2^(-53)) < x < 2^1024,
             therefore we have one extra exponent reduction step */
          if (ABS (x) >= div9)
            {
              x /= div9; /* exact */
              shift_exp += 512;
              mpfr_div_2si (t, t, 512, MPFR_RNDZ);
            }
        } /* Check overflow of double */
      else /* no overflow on double */
        {
          long double div9, div10, div11;

          div9 = (long double) (double) 7.4583407312002067432909653e-155;
          /* div9 = 2^(-2^9) */
          div10 = div9  * div9;  /* 2^(-2^10) */
          div11 = div10 * div10; /* 2^(-2^11) if extended precision */
          /* since -DBL_MAX <= x <= DBL_MAX, the cast to double should not
             overflow here */
          if (ABS(x) < div10 &&
              div11 != (long double) 0.0 &&
              div11 / div10 == div10) /* possible underflow */
            {
              long double div12, div13;
              /* After the divisions, any bit of x must be >= div10,
                 hence the possible division by div9. */
              div12 = div11 * div11; /* 2^(-2^12) */
              div13 = div12 * div12; /* 2^(-2^13) */
              if (ABS (x) <= div13)
                {
                  x /= div13; /* exact */
                  shift_exp -= 8192;
                  mpfr_mul_2si (t, t, 8192, MPFR_RNDZ);
                }
              if (ABS (x) <= div12)
                {
                  x /= div12; /* exact */
                  shift_exp -= 4096;
                  mpfr_mul_2si (t, t, 4096, MPFR_RNDZ);
                }
              if (ABS (x) <= div11)
                {
                  x /= div11; /* exact */
                  shift_exp -= 2048;
                  mpfr_mul_2si (t, t, 2048, MPFR_RNDZ);
                }
              if (ABS (x) <= div10)
                {
                  x /= div10; /* exact */
                  shift_exp -= 1024;
                  mpfr_mul_2si (t, t, 1024, MPFR_RNDZ);
                }
              if (ABS(x) <= div9)
                {
                  x /= div9;  /* exact */
                  shift_exp -= 512;
                  mpfr_mul_2si (t, t, 512, MPFR_RNDZ);
                }
            }
          else /* no underflow */
            {
              inexact = mpfr_set_d (u, (double) x, MPFR_RNDZ);
              MPFR_ASSERTD (inexact == 0);
              if (mpfr_add (t, t, u, MPFR_RNDZ) != 0)
                {
                  if (!mpfr_number_p (t))
                    break;
                  /* Inexact. This cannot happen unless the C implementation
                     "lies" on the precision or when long doubles are
                     implemented with FP expansions like under Mac OS X. */
                  if (MPFR_PREC (t) != MPFR_PREC (r) + 1)
                    {
                      /* We assume that MPFR_PREC (r) < MPFR_PREC_MAX.
                         The precision MPFR_PREC (r) + 1 allows us to
                         deduce the rounding bit and the sticky bit. */
                      mpfr_set_prec (t, MPFR_PREC (r) + 1);
                      goto convert;
                    }
                  else
                    {
                      mp_limb_t *tp;
                      int rb_mask;

                      /* Since mpfr_add was inexact, the sticky bit is 1. */
                      tp = MPFR_MANT (t);
                      rb_mask = MPFR_LIMB_ONE <<
                        (GMP_NUMB_BITS - 1 -
                         (MPFR_PREC (r) & (GMP_NUMB_BITS - 1)));
                      if (rnd_mode == MPFR_RNDN)
                        rnd_mode = (*tp & rb_mask) ^ MPFR_IS_NEG (t) ?
                          MPFR_RNDU : MPFR_RNDD;
                      *tp |= rb_mask;
                      break;
                    }
                }
              x -= (long double) mpfr_get_d1 (u); /* exact */
            }
        }
    }
  inexact = mpfr_mul_2si (r, t, shift_exp, rnd_mode);
  mpfr_clear (t);
  mpfr_clear (u);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (r, inexact, rnd_mode);

 nan:
  MPFR_SET_NAN(r);
  MPFR_RET_NAN;
}

#else /* IEEE Extended Little Endian Code */

int
mpfr_set_ld (mpfr_ptr r, long double d, mpfr_rnd_t rnd_mode)
{
  int inexact, i, k, cnt;
  mpfr_t tmp;
  mp_limb_t tmpmant[MPFR_LIMBS_PER_LONG_DOUBLE];
  mpfr_long_double_t x;
  mpfr_exp_t exp;
  int signd;
  MPFR_SAVE_EXPO_DECL (expo);

  /* Check for NAN */
  if (MPFR_UNLIKELY (d != d))
    {
      MPFR_SET_NAN (r);
      MPFR_RET_NAN;
    }
  /* Check for INF */
  else if (MPFR_UNLIKELY (d > MPFR_LDBL_MAX))
    {
      MPFR_SET_INF (r);
      MPFR_SET_POS (r);
      return 0;
    }
  else if (MPFR_UNLIKELY (d < -MPFR_LDBL_MAX))
    {
      MPFR_SET_INF (r);
      MPFR_SET_NEG (r);
      return 0;
    }
  /* Check for ZERO */
  else if (MPFR_UNLIKELY (d == 0.0))
    {
      x.ld = d;
      MPFR_SET_ZERO (r);
      if (x.s.sign == 1)
        MPFR_SET_NEG(r);
      else
        MPFR_SET_POS(r);
      return 0;
    }

  /* now d is neither 0, nor NaN nor Inf */
  MPFR_SAVE_EXPO_MARK (expo);

  MPFR_MANT (tmp) = tmpmant;
  MPFR_PREC (tmp) = 64;

  /* Extract sign */
  x.ld = d;
  signd = MPFR_SIGN_POS;
  if (x.ld < 0.0)
    {
      signd = MPFR_SIGN_NEG;
      x.ld = -x.ld;
    }

  /* Extract mantissa */
#if GMP_NUMB_BITS >= 64
  tmpmant[0] = ((mp_limb_t) x.s.manh << 32) | ((mp_limb_t) x.s.manl);
#else
  tmpmant[0] = (mp_limb_t) x.s.manl;
  tmpmant[1] = (mp_limb_t) x.s.manh;
#endif

  /* Normalize mantissa */
  i = MPFR_LIMBS_PER_LONG_DOUBLE;
  MPN_NORMALIZE_NOT_ZERO (tmpmant, i);
  k = MPFR_LIMBS_PER_LONG_DOUBLE - i;
  count_leading_zeros (cnt, tmpmant[i - 1]);
  if (MPFR_LIKELY (cnt != 0))
    mpn_lshift (tmpmant + k, tmpmant, i, cnt);
  else if (k != 0)
    MPN_COPY (tmpmant + k, tmpmant, i);
  if (MPFR_UNLIKELY (k != 0))
    MPN_ZERO (tmpmant, k);

  /* Set exponent */
  exp = (mpfr_exp_t) ((x.s.exph << 8) + x.s.expl);  /* 15-bit unsigned int */
  if (MPFR_UNLIKELY (exp == 0))
    exp -= 0x3FFD;
  else
    exp -= 0x3FFE;

  MPFR_SET_EXP (tmp, exp - cnt - k * GMP_NUMB_BITS);

  /* tmp is exact */
  inexact = mpfr_set4 (r, tmp, rnd_mode, signd);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (r, inexact, rnd_mode);
}

#endif
