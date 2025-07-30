/* mpfr_pow -- power function x^y

Copyright 2001-2017 Free Software Foundation, Inc.
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

/* return non zero iff x^y is exact.
   Assumes x and y are ordinary numbers,
   y is not an integer, x is not a power of 2 and x is positive

   If x^y is exact, it computes it and sets *inexact.
*/
static int
mpfr_pow_is_exact (mpfr_ptr z, mpfr_srcptr x, mpfr_srcptr y,
                   mpfr_rnd_t rnd_mode, int *inexact)
{
  mpz_t a, c;
  mpfr_exp_t d, b;
  unsigned long i;
  int res;

  MPFR_ASSERTD (!MPFR_IS_SINGULAR (y));
  MPFR_ASSERTD (!MPFR_IS_SINGULAR (x));
  MPFR_ASSERTD (!mpfr_integer_p (y));
  MPFR_ASSERTD (mpfr_cmp_si_2exp (x, MPFR_INT_SIGN (x),
                                  MPFR_GET_EXP (x) - 1) != 0);
  MPFR_ASSERTD (MPFR_IS_POS (x));

  if (MPFR_IS_NEG (y))
    return 0; /* x is not a power of two => x^-y is not exact */

  /* compute d such that y = c*2^d with c odd integer */
  mpz_init (c);
  d = mpfr_get_z_2exp (c, y);
  i = mpz_scan1 (c, 0);
  mpz_fdiv_q_2exp (c, c, i);
  d += i;
  /* now y=c*2^d with c odd */
  /* Since y is not an integer, d is necessarily < 0 */
  MPFR_ASSERTD (d < 0);

  /* Compute a,b such that x=a*2^b */
  mpz_init (a);
  b = mpfr_get_z_2exp (a, x);
  i = mpz_scan1 (a, 0);
  mpz_fdiv_q_2exp (a, a, i);
  b += i;
  /* now x=a*2^b with a is odd */

  for (res = 1 ; d != 0 ; d++)
    {
      /* a*2^b is a square iff
            (i)  a is a square when b is even
            (ii) 2*a is a square when b is odd */
      if (b % 2 != 0)
        {
          mpz_mul_2exp (a, a, 1); /* 2*a */
          b --;
        }
      MPFR_ASSERTD ((b % 2) == 0);
      if (!mpz_perfect_square_p (a))
        {
          res = 0;
          goto end;
        }
      mpz_sqrt (a, a);
      b = b / 2;
    }
  /* Now x = (a'*2^b')^(2^-d) with d < 0
     so x^y = ((a'*2^b')^(2^-d))^(c*2^d)
            = ((a'*2^b')^c with c odd integer */
  {
    mpfr_t tmp;
    mpfr_prec_t p;
    MPFR_MPZ_SIZEINBASE2 (p, a);
    mpfr_init2 (tmp, p); /* prec = 1 should not be possible */
    res = mpfr_set_z (tmp, a, MPFR_RNDN);
    MPFR_ASSERTD (res == 0);
    res = mpfr_mul_2si (tmp, tmp, b, MPFR_RNDN);
    MPFR_ASSERTD (res == 0);
    *inexact = mpfr_pow_z (z, tmp, c, rnd_mode);
    mpfr_clear (tmp);
    res = 1;
  }
 end:
  mpz_clear (a);
  mpz_clear (c);
  return res;
}

/* Return 1 if y is an odd integer, 0 otherwise. */
static int
is_odd (mpfr_srcptr y)
{
  mpfr_exp_t expo;
  mpfr_prec_t prec;
  mp_size_t yn;
  mp_limb_t *yp;

  /* NAN, INF or ZERO are not allowed */
  MPFR_ASSERTD (!MPFR_IS_SINGULAR (y));

  expo = MPFR_GET_EXP (y);
  if (expo <= 0)
    return 0;  /* |y| < 1 and not 0 */

  prec = MPFR_PREC(y);
  if ((mpfr_prec_t) expo > prec)
    return 0;  /* y is a multiple of 2^(expo-prec), thus not odd */

  /* 0 < expo <= prec:
     y = 1xxxxxxxxxt.zzzzzzzzzzzzzzzzzz[000]
          expo bits   (prec-expo) bits

     We have to check that:
     (a) the bit 't' is set
     (b) all the 'z' bits are zero
  */

  prec = MPFR_PREC2LIMBS (prec) * GMP_NUMB_BITS - expo;
  /* number of z+0 bits */

  yn = prec / GMP_NUMB_BITS;
  MPFR_ASSERTN(yn >= 0);
  /* yn is the index of limb containing the 't' bit */

  yp = MPFR_MANT(y);
  /* if expo is a multiple of GMP_NUMB_BITS, t is bit 0 */
  if (expo % GMP_NUMB_BITS == 0 ? (yp[yn] & 1) == 0
      : yp[yn] << ((expo % GMP_NUMB_BITS) - 1) != MPFR_LIMB_HIGHBIT)
    return 0;
  while (--yn >= 0)
    if (yp[yn] != 0)
      return 0;
  return 1;
}

/* Assumes that the exponent range has already been extended and if y is
   an integer, then the result is not exact in unbounded exponent range. */
int
mpfr_pow_general (mpfr_ptr z, mpfr_srcptr x, mpfr_srcptr y,
                  mpfr_rnd_t rnd_mode, int y_is_integer, mpfr_save_expo_t *expo)
{
  mpfr_t t, u, k, absx;
  int neg_result = 0;
  int k_non_zero = 0;
  int check_exact_case = 0;
  int inexact;
  /* Declaration of the size variable */
  mpfr_prec_t Nz = MPFR_PREC(z);               /* target precision */
  mpfr_prec_t Nt;                              /* working precision */
  mpfr_exp_t err;                              /* error */
  MPFR_ZIV_DECL (ziv_loop);


  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg y[%Pu]=%.*Rg rnd=%d",
      mpfr_get_prec (x), mpfr_log_prec, x,
      mpfr_get_prec (y), mpfr_log_prec, y, rnd_mode),
     ("z[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (z), mpfr_log_prec, z, inexact));

  /* We put the absolute value of x in absx, pointing to the significand
     of x to avoid allocating memory for the significand of absx. */
  MPFR_ALIAS(absx, x, /*sign=*/ 1, /*EXP=*/ MPFR_EXP(x));

  /* We will compute the absolute value of the result. So, let's
     invert the rounding mode if the result is negative. */
  if (MPFR_IS_NEG (x) && is_odd (y))
    {
      neg_result = 1;
      rnd_mode = MPFR_INVERT_RND (rnd_mode);
    }

  /* compute the precision of intermediary variable */
  /* the optimal number of bits : see algorithms.tex */
  Nt = Nz + 5 + MPFR_INT_CEIL_LOG2 (Nz);

  /* initialise of intermediary variable */
  mpfr_init2 (t, Nt);

  MPFR_ZIV_INIT (ziv_loop, Nt);
  for (;;)
    {
      MPFR_BLOCK_DECL (flags1);

      /* compute exp(y*ln|x|), using MPFR_RNDU to get an upper bound, so
         that we can detect underflows. */
      mpfr_log (t, absx, MPFR_IS_NEG (y) ? MPFR_RNDD : MPFR_RNDU); /* ln|x| */
      mpfr_mul (t, y, t, MPFR_RNDU);                              /* y*ln|x| */
      if (k_non_zero)
        {
          MPFR_LOG_MSG (("subtract k * ln(2)\n", 0));
          mpfr_const_log2 (u, MPFR_RNDD);
          mpfr_mul (u, u, k, MPFR_RNDD);
          /* Error on u = k * log(2): < k * 2^(-Nt) < 1. */
          mpfr_sub (t, t, u, MPFR_RNDU);
          MPFR_LOG_MSG (("t = y * ln|x| - k * ln(2)\n", 0));
          MPFR_LOG_VAR (t);
        }
      /* estimate of the error -- see pow function in algorithms.tex.
         The error on t is at most 1/2 + 3*2^(EXP(t)+1) ulps, which is
         <= 2^(EXP(t)+3) for EXP(t) >= -1, and <= 2 ulps for EXP(t) <= -2.
         Additional error if k_no_zero: treal = t * errk, with
         1 - |k| * 2^(-Nt) <= exp(-|k| * 2^(-Nt)) <= errk <= 1,
         i.e., additional absolute error <= 2^(EXP(k)+EXP(t)-Nt).
         Total error <= 2^err1 + 2^err2 <= 2^(max(err1,err2)+1). */
      err = MPFR_NOTZERO (t) && MPFR_GET_EXP (t) >= -1 ?
        MPFR_GET_EXP (t) + 3 : 1;
      if (k_non_zero)
        {
          if (MPFR_GET_EXP (k) > err)
            err = MPFR_GET_EXP (k);
          err++;
        }
      MPFR_BLOCK (flags1, mpfr_exp (t, t, MPFR_RNDN));  /* exp(y*ln|x|)*/
      /* We need to test */
      if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (t) || MPFR_UNDERFLOW (flags1)))
        {
          mpfr_prec_t Ntmin;
          MPFR_BLOCK_DECL (flags2);

          MPFR_ASSERTN (!k_non_zero);
          MPFR_ASSERTN (!MPFR_IS_NAN (t));

          /* Real underflow? */
          if (MPFR_IS_ZERO (t))
            {
              /* Underflow. We computed rndn(exp(t)), where t >= y*ln|x|.
                 Therefore rndn(|x|^y) = 0, and we have a real underflow on
                 |x|^y. */
              inexact = mpfr_underflow (z, rnd_mode == MPFR_RNDN ? MPFR_RNDZ
                                        : rnd_mode, MPFR_SIGN_POS);
              if (expo != NULL)
                MPFR_SAVE_EXPO_UPDATE_FLAGS (*expo, MPFR_FLAGS_INEXACT
                                             | MPFR_FLAGS_UNDERFLOW);
              break;
            }

          /* Real overflow? */
          if (MPFR_IS_INF (t))
            {
              /* Note: we can probably use a low precision for this test. */
              mpfr_log (t, absx, MPFR_IS_NEG (y) ? MPFR_RNDU : MPFR_RNDD);
              mpfr_mul (t, y, t, MPFR_RNDD);            /* y * ln|x| */
              MPFR_BLOCK (flags2, mpfr_exp (t, t, MPFR_RNDD));
              /* t = lower bound on exp(y * ln|x|) */
              if (MPFR_OVERFLOW (flags2))
                {
                  /* We have computed a lower bound on |x|^y, and it
                     overflowed. Therefore we have a real overflow
                     on |x|^y. */
                  inexact = mpfr_overflow (z, rnd_mode, MPFR_SIGN_POS);
                  if (expo != NULL)
                    MPFR_SAVE_EXPO_UPDATE_FLAGS (*expo, MPFR_FLAGS_INEXACT
                                                 | MPFR_FLAGS_OVERFLOW);
                  break;
                }
            }

          k_non_zero = 1;
          Ntmin = sizeof(mpfr_exp_t) * CHAR_BIT;
          if (Ntmin > Nt)
            {
              Nt = Ntmin;
              mpfr_set_prec (t, Nt);
            }
          mpfr_init2 (u, Nt);
          mpfr_init2 (k, Ntmin);
          mpfr_log2 (k, absx, MPFR_RNDN);
          mpfr_mul (k, y, k, MPFR_RNDN);
          mpfr_round (k, k);
          MPFR_LOG_VAR (k);
          /* |y| < 2^Ntmin, therefore |k| < 2^Nt. */
          continue;
        }
      if (MPFR_LIKELY (MPFR_CAN_ROUND (t, Nt - err, Nz, rnd_mode)))
        {
          inexact = mpfr_set (z, t, rnd_mode);
          break;
        }

      /* check exact power, except when y is an integer (since the
         exact cases for y integer have already been filtered out) */
      if (check_exact_case == 0 && ! y_is_integer)
        {
          if (mpfr_pow_is_exact (z, absx, y, rnd_mode, &inexact))
            break;
          check_exact_case = 1;
        }

      /* reactualisation of the precision */
      MPFR_ZIV_NEXT (ziv_loop, Nt);
      mpfr_set_prec (t, Nt);
      if (k_non_zero)
        mpfr_set_prec (u, Nt);
    }
  MPFR_ZIV_FREE (ziv_loop);

  if (k_non_zero)
    {
      int inex2;
      long lk;

      /* The rounded result in an unbounded exponent range is z * 2^k. As
       * MPFR chooses underflow after rounding, the mpfr_mul_2si below will
       * correctly detect underflows and overflows. However, in rounding to
       * nearest, if z * 2^k = 2^(emin - 2), then the double rounding may
       * affect the result. We need to cope with that before overwriting z.
       * This can occur only if k < 0 (this test is necessary to avoid a
       * potential integer overflow).
       * If inexact >= 0, then the real result is <= 2^(emin - 2), so that
       * o(2^(emin - 2)) = +0 is correct. If inexact < 0, then the real
       * result is > 2^(emin - 2) and we need to round to 2^(emin - 1).
       */
      MPFR_ASSERTN (MPFR_EXP_MAX <= LONG_MAX);
      lk = mpfr_get_si (k, MPFR_RNDN);
      /* Due to early overflow detection, |k| should not be much larger than
       * MPFR_EMAX_MAX, and as MPFR_EMAX_MAX <= MPFR_EXP_MAX/2 <= LONG_MAX/2,
       * an overflow should not be possible in mpfr_get_si (and lk is exact).
       * And one even has the following assertion. TODO: complete proof.
       */
      MPFR_ASSERTD (lk > LONG_MIN && lk < LONG_MAX);
      /* Note: even in case of overflow (lk inexact), the code is correct.
       * Indeed, for the 3 occurrences of lk:
       *   - The test lk < 0 is correct as sign(lk) = sign(k).
       *   - In the test MPFR_GET_EXP (z) == __gmpfr_emin - 1 - lk,
       *     if lk is inexact, then lk = LONG_MIN <= MPFR_EXP_MIN
       *     (the minimum value of the mpfr_exp_t type), and
       *     __gmpfr_emin - 1 - lk >= MPFR_EMIN_MIN - 1 - 2 * MPFR_EMIN_MIN
       *     >= - MPFR_EMIN_MIN - 1 = MPFR_EMAX_MAX - 1. However, from the
       *     choice of k, z has been chosen to be around 1, so that the
       *     result of the test is false, as if lk were exact.
       *   - In the mpfr_mul_2si (z, z, lk, rnd_mode), if lk is inexact,
       *     then |lk| >= LONG_MAX >= MPFR_EXP_MAX, and as z is around 1,
       *     mpfr_mul_2si underflows or overflows in the same way as if
       *     lk were exact.
       * TODO: give a bound on |t|, then on |EXP(z)|.
       */
      if (rnd_mode == MPFR_RNDN && inexact < 0 && lk < 0 &&
          MPFR_GET_EXP (z) == __gmpfr_emin - 1 - lk && mpfr_powerof2_raw (z))
        {
          /* Rounding to nearest, real result > z * 2^k = 2^(emin - 2),
           * underflow case: as the minimum precision is > 1, we will
           * obtain the correct result and exceptions by replacing z by
           * nextabove(z).
           */
          MPFR_ASSERTN (MPFR_PREC_MIN > 1);
          mpfr_nextabove (z);
        }
      mpfr_clear_flags ();
      inex2 = mpfr_mul_2si (z, z, lk, rnd_mode);
      if (inex2)  /* underflow or overflow */
        {
          inexact = inex2;
          if (expo != NULL)
            MPFR_SAVE_EXPO_UPDATE_FLAGS (*expo, __gmpfr_flags);
        }
      mpfr_clears (u, k, (mpfr_ptr) 0);
    }
  mpfr_clear (t);

  /* update the sign of the result if x was negative */
  if (neg_result)
    {
      MPFR_SET_NEG(z);
      inexact = -inexact;
    }

  return inexact;
}

/* The computation of z = pow(x,y) is done by
   z = exp(y * log(x)) = x^y
   For the special cases, see Section F.9.4.4 of the C standard:
     _ pow(±0, y) = ±inf for y an odd integer < 0.
     _ pow(±0, y) = +inf for y < 0 and not an odd integer.
     _ pow(±0, y) = ±0 for y an odd integer > 0.
     _ pow(±0, y) = +0 for y > 0 and not an odd integer.
     _ pow(-1, ±inf) = 1.
     _ pow(+1, y) = 1 for any y, even a NaN.
     _ pow(x, ±0) = 1 for any x, even a NaN.
     _ pow(x, y) = NaN for finite x < 0 and finite non-integer y.
     _ pow(x, -inf) = +inf for |x| < 1.
     _ pow(x, -inf) = +0 for |x| > 1.
     _ pow(x, +inf) = +0 for |x| < 1.
     _ pow(x, +inf) = +inf for |x| > 1.
     _ pow(-inf, y) = -0 for y an odd integer < 0.
     _ pow(-inf, y) = +0 for y < 0 and not an odd integer.
     _ pow(-inf, y) = -inf for y an odd integer > 0.
     _ pow(-inf, y) = +inf for y > 0 and not an odd integer.
     _ pow(+inf, y) = +0 for y < 0.
     _ pow(+inf, y) = +inf for y > 0. */
int
mpfr_pow (mpfr_ptr z, mpfr_srcptr x, mpfr_srcptr y, mpfr_rnd_t rnd_mode)
{
  int inexact;
  int cmp_x_1;
  int y_is_integer;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg y[%Pu]=%.*Rg rnd=%d",
      mpfr_get_prec (x), mpfr_log_prec, x,
      mpfr_get_prec (y), mpfr_log_prec, y, rnd_mode),
     ("z[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (z), mpfr_log_prec, z, inexact));

  if (MPFR_ARE_SINGULAR (x, y))
    {
      /* pow(x, 0) returns 1 for any x, even a NaN. */
      if (MPFR_UNLIKELY (MPFR_IS_ZERO (y)))
        return mpfr_set_ui (z, 1, rnd_mode);
      else if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (z);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_NAN (y))
        {
          /* pow(+1, NaN) returns 1. */
          if (mpfr_cmp_ui (x, 1) == 0)
            return mpfr_set_ui (z, 1, rnd_mode);
          MPFR_SET_NAN (z);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (y))
        {
          if (MPFR_IS_INF (x))
            {
              if (MPFR_IS_POS (y))
                MPFR_SET_INF (z);
              else
                MPFR_SET_ZERO (z);
              MPFR_SET_POS (z);
              MPFR_RET (0);
            }
          else
            {
              int cmp;
              cmp = mpfr_cmpabs (x, __gmpfr_one) * MPFR_INT_SIGN (y);
              MPFR_SET_POS (z);
              if (cmp > 0)
                {
                  /* Return +inf. */
                  MPFR_SET_INF (z);
                  MPFR_RET (0);
                }
              else if (cmp < 0)
                {
                  /* Return +0. */
                  MPFR_SET_ZERO (z);
                  MPFR_RET (0);
                }
              else
                {
                  /* Return 1. */
                  return mpfr_set_ui (z, 1, rnd_mode);
                }
            }
        }
      else if (MPFR_IS_INF (x))
        {
          int negative;
          /* Determine the sign now, in case y and z are the same object */
          negative = MPFR_IS_NEG (x) && is_odd (y);
          if (MPFR_IS_POS (y))
            MPFR_SET_INF (z);
          else
            MPFR_SET_ZERO (z);
          if (negative)
            MPFR_SET_NEG (z);
          else
            MPFR_SET_POS (z);
          MPFR_RET (0);
        }
      else
        {
          int negative;
          MPFR_ASSERTD (MPFR_IS_ZERO (x));
          /* Determine the sign now, in case y and z are the same object */
          negative = MPFR_IS_NEG(x) && is_odd (y);
          if (MPFR_IS_NEG (y))
            {
              MPFR_ASSERTD (! MPFR_IS_INF (y));
              MPFR_SET_INF (z);
              mpfr_set_divby0 ();
            }
          else
            MPFR_SET_ZERO (z);
          if (negative)
            MPFR_SET_NEG (z);
          else
            MPFR_SET_POS (z);
          MPFR_RET (0);
        }
    }

  /* x^y for x < 0 and y not an integer is not defined */
  y_is_integer = mpfr_integer_p (y);
  if (MPFR_IS_NEG (x) && ! y_is_integer)
    {
      MPFR_SET_NAN (z);
      MPFR_RET_NAN;
    }

  /* now the result cannot be NaN:
     (1) either x > 0
     (2) or x < 0 and y is an integer */

  cmp_x_1 = mpfr_cmpabs (x, __gmpfr_one);
  if (cmp_x_1 == 0)
    return mpfr_set_si (z, MPFR_IS_NEG (x) && is_odd (y) ? -1 : 1, rnd_mode);

  /* now we have:
     (1) either x > 0
     (2) or x < 0 and y is an integer
     and in addition |x| <> 1.
  */

  /* detect overflow: an overflow is possible if
     (a) |x| > 1 and y > 0
     (b) |x| < 1 and y < 0.
     FIXME: this assumes 1 is always representable.

     FIXME2: maybe we can test overflow and underflow simultaneously.
     The idea is the following: first compute an approximation to
     y * log2|x|, using rounding to nearest. If |x| is not too near from 1,
     this approximation should be accurate enough, and in most cases enable
     one to prove that there is no underflow nor overflow.
     Otherwise, it should enable one to check only underflow or overflow,
     instead of both cases as in the present case.
  */
  if (cmp_x_1 * MPFR_SIGN (y) > 0)
    {
      mpfr_t t;
      int negative, overflow;

      MPFR_SAVE_EXPO_MARK (expo);
      mpfr_init2 (t, 53);
      /* we want a lower bound on y*log2|x|:
         (i) if x > 0, it suffices to round log2(x) toward zero, and
             to round y*o(log2(x)) toward zero too;
         (ii) if x < 0, we first compute t = o(-x), with rounding toward 1,
              and then follow as in case (1). */
      if (MPFR_SIGN (x) > 0)
        mpfr_log2 (t, x, MPFR_RNDZ);
      else
        {
          mpfr_neg (t, x, (cmp_x_1 > 0) ? MPFR_RNDZ : MPFR_RNDU);
          mpfr_log2 (t, t, MPFR_RNDZ);
        }
      mpfr_mul (t, t, y, MPFR_RNDZ);
      overflow = mpfr_cmp_si (t, __gmpfr_emax) > 0;
      mpfr_clear (t);
      MPFR_SAVE_EXPO_FREE (expo);
      if (overflow)
        {
          MPFR_LOG_MSG (("early overflow detection\n", 0));
          negative = MPFR_SIGN(x) < 0 && is_odd (y);
          return mpfr_overflow (z, rnd_mode, negative ? -1 : 1);
        }
    }

  /* Basic underflow checking. One has:
   *   - if y > 0, |x^y| < 2^(EXP(x) * y);
   *   - if y < 0, |x^y| <= 2^((EXP(x) - 1) * y);
   * so that one can compute a value ebound such that |x^y| < 2^ebound.
   * If we have ebound <= emin - 2 (emin - 1 in directed rounding modes),
   * then there is an underflow and we can decide the return value.
   */
  if (MPFR_IS_NEG (y) ? (MPFR_GET_EXP (x) > 1) : (MPFR_GET_EXP (x) < 0))
    {
      mpfr_t tmp;
      mpfr_eexp_t ebound;
      int inex2;

      /* We must restore the flags. */
      MPFR_SAVE_EXPO_MARK (expo);
      mpfr_init2 (tmp, sizeof (mpfr_exp_t) * CHAR_BIT);
      inex2 = mpfr_set_exp_t (tmp, MPFR_GET_EXP (x), MPFR_RNDN);
      MPFR_ASSERTN (inex2 == 0);
      if (MPFR_IS_NEG (y))
        {
          inex2 = mpfr_sub_ui (tmp, tmp, 1, MPFR_RNDN);
          MPFR_ASSERTN (inex2 == 0);
        }
      mpfr_mul (tmp, tmp, y, MPFR_RNDU);
      if (MPFR_IS_NEG (y))
        mpfr_nextabove (tmp);
      /* tmp doesn't necessarily fit in ebound, but that doesn't matter
         since we get the minimum value in such a case. */
      ebound = mpfr_get_exp_t (tmp, MPFR_RNDU);
      mpfr_clear (tmp);
      MPFR_SAVE_EXPO_FREE (expo);
      if (MPFR_UNLIKELY (ebound <=
                         __gmpfr_emin - (rnd_mode == MPFR_RNDN ? 2 : 1)))
        {
          /* warning: mpfr_underflow rounds away from 0 for MPFR_RNDN */
          MPFR_LOG_MSG (("early underflow detection\n", 0));
          return mpfr_underflow (z,
                                 rnd_mode == MPFR_RNDN ? MPFR_RNDZ : rnd_mode,
                                 MPFR_SIGN (x) < 0 && is_odd (y) ? -1 : 1);
        }
    }

  /* If y is an integer, we can use mpfr_pow_z (based on multiplications),
     but if y is very large (I'm not sure about the best threshold -- VL),
     we shouldn't use it, as it can be very slow and take a lot of memory
     (and even crash or make other programs crash, as several hundred of
     MBs may be necessary). Note that in such a case, either x = +/-2^b
     (this case is handled below) or x^y cannot be represented exactly in
     any precision supported by MPFR (the general case uses this property).
  */
  if (y_is_integer && (MPFR_GET_EXP (y) <= 256))
    {
      mpz_t zi;

      MPFR_LOG_MSG (("special code for y not too large integer\n", 0));
      mpz_init (zi);
      mpfr_get_z (zi, y, MPFR_RNDN);
      inexact = mpfr_pow_z (z, x, zi, rnd_mode);
      mpz_clear (zi);
      return inexact;
    }

  /* Special case (+/-2^b)^Y which could be exact. If x is negative, then
     necessarily y is a large integer. */
  {
    mpfr_exp_t b = MPFR_GET_EXP (x) - 1;

    MPFR_ASSERTN (b >= LONG_MIN && b <= LONG_MAX);  /* FIXME... */
    if (mpfr_cmp_si_2exp (x, MPFR_SIGN(x), b) == 0)
      {
        mpfr_t tmp;
        int sgnx = MPFR_SIGN (x);

        MPFR_LOG_MSG (("special case (+/-2^b)^Y\n", 0));
        /* now x = +/-2^b, so x^y = (+/-1)^y*2^(b*y) is exact whenever b*y is
           an integer */
        MPFR_SAVE_EXPO_MARK (expo);
        mpfr_init2 (tmp, MPFR_PREC (y) + sizeof (long) * CHAR_BIT);
        inexact = mpfr_mul_si (tmp, y, b, MPFR_RNDN); /* exact */
        MPFR_ASSERTN (inexact == 0);
        /* Note: as the exponent range has been extended, an overflow is not
           possible (due to basic overflow and underflow checking above, as
           the result is ~ 2^tmp), and an underflow is not possible either
           because b is an integer (thus either 0 or >= 1). */
        mpfr_clear_flags ();
        inexact = mpfr_exp2 (z, tmp, rnd_mode);
        mpfr_clear (tmp);
        if (sgnx < 0 && is_odd (y))
          {
            mpfr_neg (z, z, rnd_mode);
            inexact = -inexact;
          }
        /* Without the following, the overflows3 test in tpow.c fails. */
        MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);
        MPFR_SAVE_EXPO_FREE (expo);
        return mpfr_check_range (z, inexact, rnd_mode);
      }
  }

  MPFR_SAVE_EXPO_MARK (expo);

  /* Case where |y * log(x)| is very small. Warning: x can be negative, in
     that case y is a large integer. */
  {
    mpfr_t t;
    mpfr_exp_t err;

    /* We need an upper bound on the exponent of y * log(x). */
    mpfr_init2 (t, 16);
    if (MPFR_IS_POS(x))
      mpfr_log (t, x, cmp_x_1 < 0 ? MPFR_RNDD : MPFR_RNDU); /* away from 0 */
    else
      {
        /* if x < -1, round to +Inf, else round to zero */
        mpfr_neg (t, x, (mpfr_cmp_si (x, -1) < 0) ? MPFR_RNDU : MPFR_RNDD);
        mpfr_log (t, t, (mpfr_cmp_ui (t, 1) < 0) ? MPFR_RNDD : MPFR_RNDU);
      }
    MPFR_ASSERTN (MPFR_IS_PURE_FP (t));
    err = MPFR_GET_EXP (y) + MPFR_GET_EXP (t);
    mpfr_clear (t);
    mpfr_clear_flags ();
    MPFR_SMALL_INPUT_AFTER_SAVE_EXPO (z, __gmpfr_one, - err, 0,
                                      (MPFR_SIGN (y) > 0) ^ (cmp_x_1 < 0),
                                      rnd_mode, expo, {});
  }

  /* General case */
  inexact = mpfr_pow_general (z, x, y, rnd_mode, y_is_integer, &expo);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (z, inexact, rnd_mode);
}
