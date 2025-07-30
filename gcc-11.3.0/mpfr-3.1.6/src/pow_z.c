/* mpfr_pow_z -- power function x^z with z a MPZ

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

/* y <- x^|z| with z != 0
   if cr=1: ensures correct rounding of y
   if cr=0: does not ensure correct rounding, but avoid spurious overflow
   or underflow, and uses the precision of y as working precision (warning,
   y and x might be the same variable). */
static int
mpfr_pow_pos_z (mpfr_ptr y, mpfr_srcptr x, mpz_srcptr z, mpfr_rnd_t rnd, int cr)
{
  mpfr_t res;
  mpfr_prec_t prec, err;
  int inexact;
  mpfr_rnd_t rnd1, rnd2;
  mpz_t absz;
  mp_size_t size_z;
  MPFR_ZIV_DECL (loop);
  MPFR_BLOCK_DECL (flags);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg z=%Zd rnd=%d cr=%d",
      mpfr_get_prec (x), mpfr_log_prec, x, z, rnd, cr),
     ("y[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (y), mpfr_log_prec, y, inexact));

  MPFR_ASSERTD (mpz_sgn (z) != 0);

  if (MPFR_UNLIKELY (mpz_cmpabs_ui (z, 1) == 0))
    return mpfr_set (y, x, rnd);

  absz[0] = z[0];
  SIZ (absz) = ABS(SIZ(absz)); /* Hack to get abs(z) */
  MPFR_MPZ_SIZEINBASE2 (size_z, z);

  /* round toward 1 (or -1) to avoid spurious overflow or underflow,
     i.e. if an overflow or underflow occurs, it is a real exception
     and is not just due to the rounding error. */
  rnd1 = (MPFR_EXP(x) >= 1) ? MPFR_RNDZ
    : (MPFR_IS_POS(x) ? MPFR_RNDU : MPFR_RNDD);
  rnd2 = (MPFR_EXP(x) >= 1) ? MPFR_RNDD : MPFR_RNDU;

  if (cr != 0)
    prec = MPFR_PREC (y) + 3 + size_z + MPFR_INT_CEIL_LOG2 (MPFR_PREC (y));
  else
    prec = MPFR_PREC (y);
  mpfr_init2 (res, prec);

  MPFR_ZIV_INIT (loop, prec);
  for (;;)
    {
      unsigned int inexmul;  /* will be non-zero if res may be inexact */
      mp_size_t i = size_z;

      /* now 2^(i-1) <= z < 2^i */
      /* see below (case z < 0) for the error analysis, which is identical,
         except if z=n, the maximal relative error is here 2(n-1)2^(-prec)
         instead of 2(2n-1)2^(-prec) for z<0. */
      MPFR_ASSERTD (prec > (mpfr_prec_t) i);
      err = prec - 1 - (mpfr_prec_t) i;

      MPFR_BLOCK (flags,
                  inexmul = mpfr_mul (res, x, x, rnd2);
                  MPFR_ASSERTD (i >= 2);
                  if (mpz_tstbit (absz, i - 2))
                    inexmul |= mpfr_mul (res, res, x, rnd1);
                  for (i -= 3; i >= 0 && !MPFR_BLOCK_EXCEP; i--)
                    {
                      inexmul |= mpfr_mul (res, res, res, rnd2);
                      if (mpz_tstbit (absz, i))
                        inexmul |= mpfr_mul (res, res, x, rnd1);
                    });
      if (MPFR_LIKELY (inexmul == 0 || cr == 0
                       || MPFR_OVERFLOW (flags) || MPFR_UNDERFLOW (flags)
                       || MPFR_CAN_ROUND (res, err, MPFR_PREC (y), rnd)))
        break;
      /* Can't decide correct rounding, increase the precision */
      MPFR_ZIV_NEXT (loop, prec);
      mpfr_set_prec (res, prec);
    }
  MPFR_ZIV_FREE (loop);

  /* Check Overflow */
  if (MPFR_OVERFLOW (flags))
    {
      MPFR_LOG_MSG (("overflow\n", 0));
      inexact = mpfr_overflow (y, rnd, mpz_odd_p (absz) ?
                               MPFR_SIGN (x) : MPFR_SIGN_POS);
    }
  /* Check Underflow */
  else if (MPFR_UNDERFLOW (flags))
    {
      MPFR_LOG_MSG (("underflow\n", 0));
      if (rnd == MPFR_RNDN)
        {
          mpfr_t y2, zz;

          /* We cannot decide now whether the result should be rounded
             toward zero or +Inf. So, let's use the general case of
             mpfr_pow, which can do that. But the problem is that the
             result can be exact! However, it is sufficient to try to
             round on 2 bits (the precision does not matter in case of
             underflow, since MPFR does not have subnormals), in which
             case, the result cannot be exact due to previous filtering
             of trivial cases. */
          MPFR_ASSERTD (mpfr_cmp_si_2exp (x, MPFR_SIGN (x),
                                          MPFR_EXP (x) - 1) != 0);
          mpfr_init2 (y2, 2);
          mpfr_init2 (zz, ABS (SIZ (z)) * GMP_NUMB_BITS);
          inexact = mpfr_set_z (zz, z, MPFR_RNDN);
          MPFR_ASSERTN (inexact == 0);
          inexact = mpfr_pow_general (y2, x, zz, rnd, 1,
                                      (mpfr_save_expo_t *) NULL);
          mpfr_clear (zz);
          mpfr_set (y, y2, MPFR_RNDN);
          mpfr_clear (y2);
          __gmpfr_flags = MPFR_FLAGS_INEXACT | MPFR_FLAGS_UNDERFLOW;
        }
      else
        {
          inexact = mpfr_underflow (y, rnd, mpz_odd_p (absz) ?
                                    MPFR_SIGN (x) : MPFR_SIGN_POS);
        }
    }
  else
    inexact = mpfr_set (y, res, rnd);

  mpfr_clear (res);
  return inexact;
}

/* The computation of y = pow(x,z) is done by
 *    y = set_ui(1)      if z = 0
 *    y = pow_ui(x,z)    if z > 0
 *    y = pow_ui(1/x,-z) if z < 0
 *
 * Note: in case z < 0, we could also compute 1/pow_ui(x,-z). However, in
 * case MAX < 1/MIN, where MAX is the largest positive value, i.e.,
 * MAX = nextbelow(+Inf), and MIN is the smallest positive value, i.e.,
 * MIN = nextabove(+0), then x^(-z) might produce an overflow, whereas
 * x^z is representable.
 */

int
mpfr_pow_z (mpfr_ptr y, mpfr_srcptr x, mpz_srcptr z, mpfr_rnd_t rnd)
{
  int   inexact;
  mpz_t tmp;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg z=%Zd rnd=%d",
      mpfr_get_prec (x), mpfr_log_prec, x, z, rnd),
     ("y[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (y), mpfr_log_prec, y, inexact));

  /* x^0 = 1 for any x, even a NaN */
  if (MPFR_UNLIKELY (mpz_sgn (z) == 0))
    return mpfr_set_ui (y, 1, rnd);

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (x))
        {
          /* Inf^n = Inf, (-Inf)^n = Inf for n even, -Inf for n odd */
          /* Inf ^(-n) = 0, sign = + if x>0 or z even */
          if (mpz_sgn (z) > 0)
            MPFR_SET_INF (y);
          else
            MPFR_SET_ZERO (y);
          if (MPFR_UNLIKELY (MPFR_IS_NEG (x) && mpz_odd_p (z)))
            MPFR_SET_NEG (y);
          else
            MPFR_SET_POS (y);
          MPFR_RET (0);
        }
      else /* x is zero */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO(x));
          if (mpz_sgn (z) > 0)
            /* 0^n = +/-0 for any n */
            MPFR_SET_ZERO (y);
          else
            {
              /* 0^(-n) if +/- INF */
              MPFR_SET_INF (y);
              mpfr_set_divby0 ();
            }
          if (MPFR_LIKELY (MPFR_IS_POS (x) || mpz_even_p (z)))
            MPFR_SET_POS (y);
          else
            MPFR_SET_NEG (y);
          MPFR_RET(0);
        }
    }

  MPFR_SAVE_EXPO_MARK (expo);

  /* detect exact powers: x^-n is exact iff x is a power of 2
     Do it if n > 0 too as this is faster and this filtering is
     needed in case of underflow. */
  if (MPFR_UNLIKELY (mpfr_cmp_si_2exp (x, MPFR_SIGN (x),
                                       MPFR_EXP (x) - 1) == 0))
    {
      mpfr_exp_t expx = MPFR_EXP (x); /* warning: x and y may be the same
                                         variable */

      MPFR_LOG_MSG (("x^n with x power of two\n", 0));
      mpfr_set_si (y, mpz_odd_p (z) ? MPFR_INT_SIGN(x) : 1, rnd);
      MPFR_ASSERTD (MPFR_IS_FP (y));
      mpz_init (tmp);
      mpz_mul_si (tmp, z, expx - 1);
      MPFR_ASSERTD (MPFR_GET_EXP (y) == 1);
      mpz_add_ui (tmp, tmp, 1);
      inexact = 0;
      if (MPFR_UNLIKELY (mpz_cmp_si (tmp, __gmpfr_emin) < 0))
        {
          MPFR_LOG_MSG (("underflow\n", 0));
          /* |y| is a power of two, thus |y| <= 2^(emin-2), and in
             rounding to nearest, the value must be rounded to 0. */
          if (rnd == MPFR_RNDN)
            rnd = MPFR_RNDZ;
          inexact = mpfr_underflow (y, rnd, MPFR_SIGN (y));
        }
      else if (MPFR_UNLIKELY (mpz_cmp_si (tmp, __gmpfr_emax) > 0))
        {
          MPFR_LOG_MSG (("overflow\n", 0));
          inexact = mpfr_overflow (y, rnd, MPFR_SIGN (y));
        }
      else
        MPFR_SET_EXP (y, mpz_get_si (tmp));
      mpz_clear (tmp);
      MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);
    }
  else if (mpz_sgn (z) > 0)
    {
      inexact = mpfr_pow_pos_z (y, x, z, rnd, 1);
      MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);
    }
  else
    {
      /* Declaration of the intermediary variable */
      mpfr_t t;
      mpfr_prec_t Nt;   /* Precision of the intermediary variable */
      mpfr_rnd_t rnd1;
      mp_size_t size_z;
      MPFR_ZIV_DECL (loop);

      MPFR_MPZ_SIZEINBASE2 (size_z, z);

      /* initial working precision */
      Nt = MPFR_PREC (y);
      Nt = Nt + size_z + 3 + MPFR_INT_CEIL_LOG2 (Nt);
      /* ensures Nt >= bits(z)+2 */

      /* initialise of intermediary variable */
      mpfr_init2 (t, Nt);

      /* We will compute rnd(rnd1(1/x) ^ (-z)), where rnd1 is the rounding
         toward sign(x), to avoid spurious overflow or underflow. */
      rnd1 = MPFR_EXP (x) < 1 ? MPFR_RNDZ :
        (MPFR_SIGN (x) > 0 ? MPFR_RNDU : MPFR_RNDD);

      MPFR_ZIV_INIT (loop, Nt);
      for (;;)
        {
          MPFR_BLOCK_DECL (flags);

          /* compute (1/x)^(-z), -z>0 */
          /* As emin = -emax, an underflow cannot occur in the division.
             And if an overflow occurs, then this means that x^z overflows
             too (since we have rounded toward 1 or -1). */
          MPFR_BLOCK (flags, mpfr_ui_div (t, 1, x, rnd1));
          MPFR_ASSERTD (! MPFR_UNDERFLOW (flags));
          /* t = (1/x)*(1+theta) where |theta| <= 2^(-Nt) */
          if (MPFR_UNLIKELY (MPFR_OVERFLOW (flags)))
            goto overflow;
          MPFR_BLOCK (flags, mpfr_pow_pos_z (t, t, z, rnd, 0));
          /* Now if z=-n, t = x^z*(1+theta)^(2n-1) where |theta| <= 2^(-Nt),
             with theta maybe different from above. If (2n-1)*2^(-Nt) <= 1/2,
             which is satisfied as soon as Nt >= bits(z)+2, then we can use
             Lemma \ref{lemma_graillat} from algorithms.tex, which yields
             t = x^z*(1+theta) with |theta| <= 2(2n-1)*2^(-Nt), thus the
             error is bounded by 2(2n-1) ulps <= 2^(bits(z)+2) ulps. */
          if (MPFR_UNLIKELY (MPFR_OVERFLOW (flags)))
            {
            overflow:
              MPFR_ZIV_FREE (loop);
              mpfr_clear (t);
              MPFR_SAVE_EXPO_FREE (expo);
              MPFR_LOG_MSG (("overflow\n", 0));
              return mpfr_overflow (y, rnd,
                                    mpz_odd_p (z) ? MPFR_SIGN (x) :
                                    MPFR_SIGN_POS);
            }
          if (MPFR_UNLIKELY (MPFR_UNDERFLOW (flags)))
            {
              MPFR_ZIV_FREE (loop);
              mpfr_clear (t);
              MPFR_LOG_MSG (("underflow\n", 0));
              if (rnd == MPFR_RNDN)
                {
                  mpfr_t y2, zz;

                  /* We cannot decide now whether the result should be
                     rounded toward zero or away from zero. So, like
                     in mpfr_pow_pos_z, let's use the general case of
                     mpfr_pow in precision 2. */
                  MPFR_ASSERTD (mpfr_cmp_si_2exp (x, MPFR_SIGN (x),
                                                  MPFR_EXP (x) - 1) != 0);
                  mpfr_init2 (y2, 2);
                  mpfr_init2 (zz, ABS (SIZ (z)) * GMP_NUMB_BITS);
                  inexact = mpfr_set_z (zz, z, MPFR_RNDN);
                  MPFR_ASSERTN (inexact == 0);
                  inexact = mpfr_pow_general (y2, x, zz, rnd, 1,
                                              (mpfr_save_expo_t *) NULL);
                  mpfr_clear (zz);
                  mpfr_set (y, y2, MPFR_RNDN);
                  mpfr_clear (y2);
                  MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, MPFR_FLAGS_UNDERFLOW);
                  goto end;
                }
              else
                {
                  MPFR_SAVE_EXPO_FREE (expo);
                  return mpfr_underflow (y, rnd, mpz_odd_p (z) ?
                                         MPFR_SIGN (x) : MPFR_SIGN_POS);
                }
            }
          if (MPFR_LIKELY (MPFR_CAN_ROUND (t, Nt - size_z - 2, MPFR_PREC (y),
                                           rnd)))
            break;
          /* actualisation of the precision */
          MPFR_ZIV_NEXT (loop, Nt);
          mpfr_set_prec (t, Nt);
        }
      MPFR_ZIV_FREE (loop);

      inexact = mpfr_set (y, t, rnd);
      mpfr_clear (t);
    }

 end:
  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, inexact, rnd);
}
