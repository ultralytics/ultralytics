/* mpfr_set_d -- convert a machine double precision float to
                 a multiple precision floating-point number

Copyright 1999-2004, 2006-2017 Free Software Foundation, Inc.
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

#include <float.h>  /* For DOUBLE_ISINF and DOUBLE_ISNAN */

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

/* extracts the bits of d in rp[0..n-1] where n=ceil(53/GMP_NUMB_BITS).
   Assumes d is neither 0 nor NaN nor Inf. */
static long
__gmpfr_extract_double (mpfr_limb_ptr rp, double d)
     /* e=0 iff GMP_NUMB_BITS=32 and rp has only one limb */
{
  long exp;
  mp_limb_t manl;
#if GMP_NUMB_BITS == 32
  mp_limb_t manh;
#endif

  /* BUGS
     1. Should handle Inf and NaN in IEEE specific code.
     2. Handle Inf and NaN also in default code, to avoid hangs.
     3. Generalize to handle all GMP_NUMB_BITS.
     4. This lits is incomplete and misspelled.
   */

  MPFR_ASSERTD(!DOUBLE_ISNAN(d));
  MPFR_ASSERTD(!DOUBLE_ISINF(d));
  MPFR_ASSERTD(d != 0.0);

#if _GMP_IEEE_FLOATS

  {
    union ieee_double_extract x;
    x.d = d;

    exp = x.s.exp;
    if (exp)
      {
#if GMP_NUMB_BITS >= 64
        manl = ((MPFR_LIMB_ONE << 63)
                | ((mp_limb_t) x.s.manh << 43) | ((mp_limb_t) x.s.manl << 11));
#else
        manh = (MPFR_LIMB_ONE << 31) | (x.s.manh << 11) | (x.s.manl >> 21);
        manl = x.s.manl << 11;
#endif
      }
    else /* subnormal number */
      {
#if GMP_NUMB_BITS >= 64
        manl = ((mp_limb_t) x.s.manh << 43) | ((mp_limb_t) x.s.manl << 11);
#else
        manh = (x.s.manh << 11) /* high 21 bits */
          | (x.s.manl >> 21); /* middle 11 bits */
        manl = x.s.manl << 11; /* low 21 bits */
#endif
      }

    if (exp)
      exp -= 1022;
    else
      exp = -1021;
  }

#else /* _GMP_IEEE_FLOATS */

  {
    /* Unknown (or known to be non-IEEE) double format.  */
    exp = 0;
    if (d >= 1.0)
      {
        MPFR_ASSERTN (d * 0.5 != d);
        while (d >= 32768.0)
          {
            d *= (1.0 / 65536.0);
            exp += 16;
          }
        while (d >= 1.0)
          {
            d *= 0.5;
            exp += 1;
          }
      }
    else if (d < 0.5)
      {
        while (d < (1.0 / 65536.0))
          {
            d *=  65536.0;
            exp -= 16;
          }
        while (d < 0.5)
          {
            d *= 2.0;
            exp -= 1;
          }
      }

    d *= MP_BASE_AS_DOUBLE;
#if GMP_NUMB_BITS >= 64
    manl = d;
#else
    manh = (mp_limb_t) d;
    manl = (mp_limb_t) ((d - manh) * MP_BASE_AS_DOUBLE);
#endif
  }

#endif /* _GMP_IEEE_FLOATS */

#if GMP_NUMB_BITS >= 64
  rp[0] = manl;
#else
  rp[1] = manh;
  rp[0] = manl;
#endif

  return exp;
}

/* End of part included from gmp-2.0.2 */

int
mpfr_set_d (mpfr_ptr r, double d, mpfr_rnd_t rnd_mode)
{
  int signd, inexact;
  unsigned int cnt;
  mp_size_t i, k;
  mpfr_t tmp;
  mp_limb_t tmpmant[MPFR_LIMBS_PER_DOUBLE];
  MPFR_SAVE_EXPO_DECL (expo);

  if (MPFR_UNLIKELY(DOUBLE_ISNAN(d)))
    {
      MPFR_SET_NAN(r);
      MPFR_RET_NAN;
    }
  else if (MPFR_UNLIKELY(d == 0))
    {
#if _GMP_IEEE_FLOATS
      union ieee_double_extract x;

      MPFR_SET_ZERO(r);
      /* set correct sign */
      x.d = d;
      if (x.s.sig == 1)
        MPFR_SET_NEG(r);
      else
        MPFR_SET_POS(r);
#else /* _GMP_IEEE_FLOATS */
      MPFR_SET_ZERO(r);
      {
        /* This is to get the sign of zero on non-IEEE hardware
           Some systems support +0.0, -0.0 and unsigned zero.
           We can't use d==+0.0 since it should be always true,
           so we check that the memory representation of d is the
           same than +0.0. etc */
        /* FIXME: consider the case where +0.0 or -0.0 may have several
           representations. */
        double poszero = +0.0, negzero = DBL_NEG_ZERO;
        if (memcmp(&d, &poszero, sizeof(double)) == 0)
          MPFR_SET_POS(r);
        else if (memcmp(&d, &negzero, sizeof(double)) == 0)
          MPFR_SET_NEG(r);
        else
          MPFR_SET_POS(r);
      }
#endif
      return 0; /* 0 is exact */
    }
  else if (MPFR_UNLIKELY(DOUBLE_ISINF(d)))
    {
      MPFR_SET_INF(r);
      if (d > 0)
        MPFR_SET_POS(r);
      else
        MPFR_SET_NEG(r);
      return 0; /* infinity is exact */
    }

  /* now d is neither 0, nor NaN nor Inf */

  MPFR_SAVE_EXPO_MARK (expo);

  /* warning: don't use tmp=r here, even if SIZE(r) >= MPFR_LIMBS_PER_DOUBLE,
     since PREC(r) may be different from PREC(tmp), and then both variables
     would have same precision in the mpfr_set4 call below. */
  MPFR_MANT(tmp) = tmpmant;
  MPFR_PREC(tmp) = IEEE_DBL_MANT_DIG;

  signd = (d < 0) ? MPFR_SIGN_NEG : MPFR_SIGN_POS;
  d = ABS (d);

  /* don't use MPFR_SET_EXP here since the exponent may be out of range */
  MPFR_EXP(tmp) = __gmpfr_extract_double (tmpmant, d);

#ifdef MPFR_WANT_ASSERT
  /* Failed assertion if the stored value is 0 (e.g., if the exponent range
     has been reduced at the wrong moment and an underflow to 0 occurred).
     Probably a bug in the C implementation if this happens. */
  i = 0;
  while (tmpmant[i] == 0)
    {
      i++;
      MPFR_ASSERTN(i < MPFR_LIMBS_PER_DOUBLE);
    }
#endif

  /* determine the index i-1 of the most significant non-zero limb
     and the number k of zero high limbs */
  i = MPFR_LIMBS_PER_DOUBLE;
  MPN_NORMALIZE_NOT_ZERO(tmpmant, i);
  k = MPFR_LIMBS_PER_DOUBLE - i;

  count_leading_zeros (cnt, tmpmant[i - 1]);

  if (MPFR_LIKELY(cnt != 0))
    mpn_lshift (tmpmant + k, tmpmant, i, cnt);
  else if (k != 0)
    MPN_COPY (tmpmant + k, tmpmant, i);

  if (MPFR_UNLIKELY(k != 0))
    MPN_ZERO (tmpmant, k);

  /* don't use MPFR_SET_EXP here since the exponent may be out of range */
  MPFR_EXP(tmp) -= (mpfr_exp_t) (cnt + k * GMP_NUMB_BITS);

  /* tmp is exact since PREC(tmp)=53 */
  inexact = mpfr_set4 (r, tmp, rnd_mode, signd);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (r, inexact, rnd_mode);
}



