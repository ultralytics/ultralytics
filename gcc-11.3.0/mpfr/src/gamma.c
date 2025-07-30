/* mpfr_gamma -- gamma function

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

#define IS_GAMMA
#include "lngamma.c"
#undef IS_GAMMA

/* return a sufficient precision such that 2-x is exact, assuming x < 0 */
static mpfr_prec_t
mpfr_gamma_2_minus_x_exact (mpfr_srcptr x)
{
  /* Since x < 0, 2-x = 2+y with y := -x.
     If y < 2, a precision w >= PREC(y) + EXP(2)-EXP(y) = PREC(y) + 2 - EXP(y)
     is enough, since no overlap occurs in 2+y, so no carry happens.
     If y >= 2, either ULP(y) <= 2, and we need w >= PREC(y)+1 since a
     carry can occur, or ULP(y) > 2, and we need w >= EXP(y)-1:
     (a) if EXP(y) <= 1, w = PREC(y) + 2 - EXP(y)
     (b) if EXP(y) > 1 and EXP(y)-PREC(y) <= 1, w = PREC(y) + 1
     (c) if EXP(y) > 1 and EXP(y)-PREC(y) > 1, w = EXP(y) - 1 */
  return (MPFR_GET_EXP(x) <= 1) ? MPFR_PREC(x) + 2 - MPFR_GET_EXP(x)
    : ((MPFR_GET_EXP(x) <= MPFR_PREC(x) + 1) ? MPFR_PREC(x) + 1
       : MPFR_GET_EXP(x) - 1);
}

/* return a sufficient precision such that 1-x is exact, assuming x < 1 */
static mpfr_prec_t
mpfr_gamma_1_minus_x_exact (mpfr_srcptr x)
{
  if (MPFR_IS_POS(x))
    return MPFR_PREC(x) - MPFR_GET_EXP(x);
  else if (MPFR_GET_EXP(x) <= 0)
    return MPFR_PREC(x) + 1 - MPFR_GET_EXP(x);
  else if (MPFR_PREC(x) >= MPFR_GET_EXP(x))
    return MPFR_PREC(x) + 1;
  else
    return MPFR_GET_EXP(x);
}

/* returns a lower bound of the number of significant bits of n!
   (not counting the low zero bits).
   We know n! >= (n/e)^n*sqrt(2*Pi*n) for n >= 1, and the number of zero bits
   is floor(n/2) + floor(n/4) + floor(n/8) + ...
   This approximation is exact for n <= 500000, except for n = 219536, 235928,
   298981, 355854, 464848, 493725, 498992 where it returns a value 1 too small.
*/
static unsigned long
bits_fac (unsigned long n)
{
  mpfr_t x, y;
  unsigned long r, k;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_SAVE_EXPO_MARK (expo);
  mpfr_init2 (x, 38);
  mpfr_init2 (y, 38);
  mpfr_set_ui (x, n, MPFR_RNDZ);
  mpfr_set_str_binary (y, "10.101101111110000101010001011000101001"); /* upper bound of e */
  mpfr_div (x, x, y, MPFR_RNDZ);
  mpfr_pow_ui (x, x, n, MPFR_RNDZ);
  mpfr_const_pi (y, MPFR_RNDZ);
  mpfr_mul_ui (y, y, 2 * n, MPFR_RNDZ);
  mpfr_sqrt (y, y, MPFR_RNDZ);
  mpfr_mul (x, x, y, MPFR_RNDZ);
  mpfr_log2 (x, x, MPFR_RNDZ);
  r = mpfr_get_ui (x, MPFR_RNDU);
  for (k = 2; k <= n; k *= 2)
    r -= n / k;
  mpfr_clear (x);
  mpfr_clear (y);
  MPFR_SAVE_EXPO_FREE (expo);

  return r;
}

/* We use the reflection formula
  Gamma(1+t) Gamma(1-t) = - Pi t / sin(Pi (1 + t))
  in order to treat the case x <= 1,
  i.e. with x = 1-t, then Gamma(x) = -Pi*(1-x)/sin(Pi*(2-x))/GAMMA(2-x)
*/
int
mpfr_gamma (mpfr_ptr gamma, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_t xp, GammaTrial, tmp, tmp2;
  mpz_t fact;
  mpfr_prec_t realprec;
  int compared, is_integer;
  int inex = 0;  /* 0 means: result gamma not set yet */
  MPFR_GROUP_DECL (group);
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_ZIV_DECL (loop);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd_mode),
     ("gamma[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (gamma), mpfr_log_prec, gamma, inex));

  /* Trivial cases */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (gamma);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (x))
        {
          if (MPFR_IS_NEG (x))
            {
              MPFR_SET_NAN (gamma);
              MPFR_RET_NAN;
            }
          else
            {
              MPFR_SET_INF (gamma);
              MPFR_SET_POS (gamma);
              MPFR_RET (0);  /* exact */
            }
        }
      else /* x is zero */
        {
          MPFR_ASSERTD(MPFR_IS_ZERO(x));
          MPFR_SET_INF(gamma);
          MPFR_SET_SAME_SIGN(gamma, x);
          mpfr_set_divby0 ();
          MPFR_RET (0);  /* exact */
        }
    }

  /* Check for tiny arguments, where gamma(x) ~ 1/x - euler + ....
     We know from "Bound on Runs of Zeros and Ones for Algebraic Functions",
     Proceedings of Arith15, T. Lang and J.-M. Muller, 2001, that the maximal
     number of consecutive zeroes or ones after the round bit is n-1 for an
     input of n bits. But we need a more precise lower bound. Assume x has
     n bits, and 1/x is near a floating-point number y of n+1 bits. We can
     write x = X*2^e, y = Y/2^f with X, Y integers of n and n+1 bits.
     Thus X*Y^2^(e-f) is near from 1, i.e., X*Y is near from 2^(f-e).
     Two cases can happen:
     (i) either X*Y is exactly 2^(f-e), but this can happen only if X and Y
         are themselves powers of two, i.e., x is a power of two;
     (ii) or X*Y is at distance at least one from 2^(f-e), thus
          |xy-1| >= 2^(e-f), or |y-1/x| >= 2^(e-f)/x = 2^(-f)/X >= 2^(-f-n).
          Since ufp(y) = 2^(n-f) [ufp = unit in first place], this means
          that the distance |y-1/x| >= 2^(-2n) ufp(y).
          Now assuming |gamma(x)-1/x| <= 1, which is true for x <= 1,
          if 2^(-2n) ufp(y) >= 2, the error is at most 2^(-2n-1) ufp(y),
          and round(1/x) with precision >= 2n+2 gives the correct result.
          If x < 2^E, then y > 2^(-E), thus ufp(y) > 2^(-E-1).
          A sufficient condition is thus EXP(x) + 2 <= -2 MAX(PREC(x),PREC(Y)).
  */
  if (MPFR_GET_EXP (x) + 2
      <= -2 * (mpfr_exp_t) MAX(MPFR_PREC(x), MPFR_PREC(gamma)))
    {
      int sign = MPFR_SIGN (x); /* retrieve sign before possible override */
      int special;
      MPFR_BLOCK_DECL (flags);

      MPFR_SAVE_EXPO_MARK (expo);

      /* for overflow cases, see below; this needs to be done
         before x possibly gets overridden. */
      special =
        MPFR_GET_EXP (x) == 1 - MPFR_EMAX_MAX &&
        MPFR_IS_POS_SIGN (sign) &&
        MPFR_IS_LIKE_RNDD (rnd_mode, sign) &&
        mpfr_powerof2_raw (x);

      MPFR_BLOCK (flags, inex = mpfr_ui_div (gamma, 1, x, rnd_mode));
      if (inex == 0) /* x is a power of two */
        {
          /* return RND(1/x - euler) = RND(+/- 2^k - eps) with eps > 0 */
          if (rnd_mode == MPFR_RNDN || MPFR_IS_LIKE_RNDU (rnd_mode, sign))
            inex = 1;
          else
            {
              mpfr_nextbelow (gamma);
              inex = -1;
            }
        }
      else if (MPFR_UNLIKELY (MPFR_OVERFLOW (flags)))
        {
          /* Overflow in the division 1/x. This is a real overflow, except
             in RNDZ or RNDD when 1/x = 2^emax, i.e. x = 2^(-emax): due to
             the "- euler", the rounded value in unbounded exponent range
             is 0.111...11 * 2^emax (not an overflow). */
          if (!special)
            MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, flags);
        }
      MPFR_SAVE_EXPO_FREE (expo);
      /* Note: an overflow is possible with an infinite result;
         in this case, the overflow flag will automatically be
         restored by mpfr_check_range. */
      return mpfr_check_range (gamma, inex, rnd_mode);
    }

  is_integer = mpfr_integer_p (x);
  /* gamma(x) for x a negative integer gives NaN */
  if (is_integer && MPFR_IS_NEG(x))
    {
      MPFR_SET_NAN (gamma);
      MPFR_RET_NAN;
    }

  compared = mpfr_cmp_ui (x, 1);
  if (compared == 0)
    return mpfr_set_ui (gamma, 1, rnd_mode);

  /* if x is an integer that fits into an unsigned long, use mpfr_fac_ui
     if argument is not too large.
     If precision is p, fac_ui costs O(u*p), whereas gamma costs O(p*M(p)),
     so for u <= M(p), fac_ui should be faster.
     We approximate here M(p) by p*log(p)^2, which is not a bad guess.
     Warning: since the generic code does not handle exact cases,
     we want all cases where gamma(x) is exact to be treated here.
  */
  if (is_integer && mpfr_fits_ulong_p (x, MPFR_RNDN))
    {
      unsigned long int u;
      mpfr_prec_t p = MPFR_PREC(gamma);
      u = mpfr_get_ui (x, MPFR_RNDN);
      if (u < 44787929UL && bits_fac (u - 1) <= p + (rnd_mode == MPFR_RNDN))
        /* bits_fac: lower bound on the number of bits of m,
           where gamma(x) = (u-1)! = m*2^e with m odd. */
        return mpfr_fac_ui (gamma, u - 1, rnd_mode);
      /* if bits_fac(...) > p (resp. p+1 for rounding to nearest),
         then gamma(x) cannot be exact in precision p (resp. p+1).
         FIXME: remove the test u < 44787929UL after changing bits_fac
         to return a mpz_t or mpfr_t. */
    }

  MPFR_SAVE_EXPO_MARK (expo);

  /* check for overflow: according to (6.1.37) in Abramowitz & Stegun,
     gamma(x) >= exp(-x) * x^(x-1/2) * sqrt(2*Pi)
              >= 2 * (x/e)^x / x for x >= 1 */
  if (compared > 0)
    {
      mpfr_t yp;
      mpfr_exp_t expxp;
      MPFR_BLOCK_DECL (flags);

      /* 1/e rounded down to 53 bits */
#define EXPM1_STR "0.010111100010110101011000110110001011001110111100111"
      mpfr_init2 (xp, 53);
      mpfr_init2 (yp, 53);
      mpfr_set_str_binary (xp, EXPM1_STR);
      mpfr_mul (xp, x, xp, MPFR_RNDZ);
      mpfr_sub_ui (yp, x, 2, MPFR_RNDZ);
      mpfr_pow (xp, xp, yp, MPFR_RNDZ); /* (x/e)^(x-2) */
      mpfr_set_str_binary (yp, EXPM1_STR);
      mpfr_mul (xp, xp, yp, MPFR_RNDZ); /* x^(x-2) / e^(x-1) */
      mpfr_mul (xp, xp, yp, MPFR_RNDZ); /* x^(x-2) / e^x */
      mpfr_mul (xp, xp, x, MPFR_RNDZ); /* lower bound on x^(x-1) / e^x */
      MPFR_BLOCK (flags, mpfr_mul_2ui (xp, xp, 1, MPFR_RNDZ));
      expxp = MPFR_GET_EXP (xp);
      mpfr_clear (xp);
      mpfr_clear (yp);
      MPFR_SAVE_EXPO_FREE (expo);
      return MPFR_OVERFLOW (flags) || expxp > __gmpfr_emax ?
        mpfr_overflow (gamma, rnd_mode, 1) :
        mpfr_gamma_aux (gamma, x, rnd_mode);
    }

  /* now compared < 0 */

  /* check for underflow: for x < 1,
     gamma(x) = Pi*(x-1)/sin(Pi*(2-x))/gamma(2-x).
     Since gamma(2-x) >= 2 * ((2-x)/e)^(2-x) / (2-x), we have
     |gamma(x)| <= Pi*(1-x)*(2-x)/2/((2-x)/e)^(2-x) / |sin(Pi*(2-x))|
                <= 12 * ((2-x)/e)^x / |sin(Pi*(2-x))|.
     To avoid an underflow in ((2-x)/e)^x, we compute the logarithm.
  */
  if (MPFR_IS_NEG(x))
    {
      int underflow = 0, sgn, ck;
      mpfr_prec_t w;

      mpfr_init2 (xp, 53);
      mpfr_init2 (tmp, 53);
      mpfr_init2 (tmp2, 53);
      /* we want an upper bound for x * [log(2-x)-1].
         since x < 0, we need a lower bound on log(2-x) */
      mpfr_ui_sub (xp, 2, x, MPFR_RNDD);
      mpfr_log (xp, xp, MPFR_RNDD);
      mpfr_sub_ui (xp, xp, 1, MPFR_RNDD);
      mpfr_mul (xp, xp, x, MPFR_RNDU);

      /* we need an upper bound on 1/|sin(Pi*(2-x))|,
         thus a lower bound on |sin(Pi*(2-x))|.
         If 2-x is exact, then the error of Pi*(2-x) is (1+u)^2 with u = 2^(-p)
         thus the error on sin(Pi*(2-x)) is less than 1/2ulp + 3Pi(2-x)u,
         assuming u <= 1, thus <= u + 3Pi(2-x)u */

      w = mpfr_gamma_2_minus_x_exact (x); /* 2-x is exact for prec >= w */
      w += 17; /* to get tmp2 small enough */
      mpfr_set_prec (tmp, w);
      mpfr_set_prec (tmp2, w);
      ck = mpfr_ui_sub (tmp, 2, x, MPFR_RNDN);
      MPFR_ASSERTD (ck == 0);  (void) ck; /* use ck to avoid a warning */
      mpfr_const_pi (tmp2, MPFR_RNDN);
      mpfr_mul (tmp2, tmp2, tmp, MPFR_RNDN); /* Pi*(2-x) */
      mpfr_sin (tmp, tmp2, MPFR_RNDN); /* sin(Pi*(2-x)) */
      sgn = mpfr_sgn (tmp);
      mpfr_abs (tmp, tmp, MPFR_RNDN);
      mpfr_mul_ui (tmp2, tmp2, 3, MPFR_RNDU); /* 3Pi(2-x) */
      mpfr_add_ui (tmp2, tmp2, 1, MPFR_RNDU); /* 3Pi(2-x)+1 */
      mpfr_div_2ui (tmp2, tmp2, mpfr_get_prec (tmp), MPFR_RNDU);
      /* if tmp2<|tmp|, we get a lower bound */
      if (mpfr_cmp (tmp2, tmp) < 0)
        {
          mpfr_sub (tmp, tmp, tmp2, MPFR_RNDZ); /* low bnd on |sin(Pi*(2-x))| */
          mpfr_ui_div (tmp, 12, tmp, MPFR_RNDU); /* upper bound */
          mpfr_log2 (tmp, tmp, MPFR_RNDU);
          mpfr_add (xp, tmp, xp, MPFR_RNDU);
          /* The assert below checks that expo.saved_emin - 2 always
             fits in a long. FIXME if we want to allow mpfr_exp_t to
             be a long long, for instance. */
          MPFR_ASSERTN (MPFR_EMIN_MIN - 2 >= LONG_MIN);
          underflow = mpfr_cmp_si (xp, expo.saved_emin - 2) <= 0;
        }

      mpfr_clear (xp);
      mpfr_clear (tmp);
      mpfr_clear (tmp2);
      if (underflow) /* the sign is the opposite of that of sin(Pi*(2-x)) */
        {
          MPFR_SAVE_EXPO_FREE (expo);
          return mpfr_underflow (gamma, (rnd_mode == MPFR_RNDN) ? MPFR_RNDZ : rnd_mode, -sgn);
        }
    }

  realprec = MPFR_PREC (gamma);
  /* we want both 1-x and 2-x to be exact */
  {
    mpfr_prec_t w;
    w = mpfr_gamma_1_minus_x_exact (x);
    if (realprec < w)
      realprec = w;
    w = mpfr_gamma_2_minus_x_exact (x);
    if (realprec < w)
      realprec = w;
  }
  realprec = realprec + MPFR_INT_CEIL_LOG2 (realprec) + 20;
  MPFR_ASSERTD(realprec >= 5);

  MPFR_GROUP_INIT_4 (group, realprec + MPFR_INT_CEIL_LOG2 (realprec) + 20,
                     xp, tmp, tmp2, GammaTrial);
  mpz_init (fact);
  MPFR_ZIV_INIT (loop, realprec);
  for (;;)
    {
      mpfr_exp_t err_g;
      int ck;
      MPFR_GROUP_REPREC_4 (group, realprec, xp, tmp, tmp2, GammaTrial);

      /* reflection formula: gamma(x) = Pi*(x-1)/sin(Pi*(2-x))/gamma(2-x) */

      ck = mpfr_ui_sub (xp, 2, x, MPFR_RNDN); /* 2-x, exact */
      MPFR_ASSERTD(ck == 0);  (void) ck; /* use ck to avoid a warning */
      mpfr_gamma (tmp, xp, MPFR_RNDN);   /* gamma(2-x), error (1+u) */
      mpfr_const_pi (tmp2, MPFR_RNDN);   /* Pi, error (1+u) */
      mpfr_mul (GammaTrial, tmp2, xp, MPFR_RNDN); /* Pi*(2-x), error (1+u)^2 */
      err_g = MPFR_GET_EXP(GammaTrial);
      mpfr_sin (GammaTrial, GammaTrial, MPFR_RNDN); /* sin(Pi*(2-x)) */
      /* If tmp is +Inf, we compute exp(lngamma(x)). */
      if (mpfr_inf_p (tmp))
        {
          inex = mpfr_explgamma (gamma, x, &expo, tmp, tmp2, rnd_mode);
          if (inex)
            goto end;
          else
            goto ziv_next;
        }
      err_g = err_g + 1 - MPFR_GET_EXP(GammaTrial);
      /* let g0 the true value of Pi*(2-x), g the computed value.
         We have g = g0 + h with |h| <= |(1+u^2)-1|*g.
         Thus sin(g) = sin(g0) + h' with |h'| <= |(1+u^2)-1|*g.
         The relative error is thus bounded by |(1+u^2)-1|*g/sin(g)
         <= |(1+u^2)-1|*2^err_g. <= 2.25*u*2^err_g for |u|<=1/4.
         With the rounding error, this gives (0.5 + 2.25*2^err_g)*u. */
      ck = mpfr_sub_ui (xp, x, 1, MPFR_RNDN); /* x-1, exact */
      MPFR_ASSERTD(ck == 0);  (void) ck; /* use ck to avoid a warning */
      mpfr_mul (xp, tmp2, xp, MPFR_RNDN); /* Pi*(x-1), error (1+u)^2 */
      mpfr_mul (GammaTrial, GammaTrial, tmp, MPFR_RNDN);
      /* [1 + (0.5 + 2.25*2^err_g)*u]*(1+u)^2 = 1 + (2.5 + 2.25*2^err_g)*u
         + (0.5 + 2.25*2^err_g)*u*(2u+u^2) + u^2.
         For err_g <= realprec-2, we have (0.5 + 2.25*2^err_g)*u <=
         0.5*u + 2.25/4 <= 0.6875 and u^2 <= u/4, thus
         (0.5 + 2.25*2^err_g)*u*(2u+u^2) + u^2 <= 0.6875*(2u+u/4) + u/4
         <= 1.8*u, thus the rel. error is bounded by (4.5 + 2.25*2^err_g)*u. */
      mpfr_div (GammaTrial, xp, GammaTrial, MPFR_RNDN);
      /* the error is of the form (1+u)^3/[1 + (4.5 + 2.25*2^err_g)*u].
         For realprec >= 5 and err_g <= realprec-2, [(4.5 + 2.25*2^err_g)*u]^2
         <= 0.71, and for |y|<=0.71, 1/(1-y) can be written 1+a*y with a<=4.
         (1+u)^3 * (1+4*(4.5 + 2.25*2^err_g)*u)
         = 1 + (21 + 9*2^err_g)*u + (57+27*2^err_g)*u^2 + (55+27*2^err_g)*u^3
             + (18+9*2^err_g)*u^4
         <= 1 + (21 + 9*2^err_g)*u + (57+27*2^err_g)*u^2 + (56+28*2^err_g)*u^3
         <= 1 + (21 + 9*2^err_g)*u + (59+28*2^err_g)*u^2
         <= 1 + (23 + 10*2^err_g)*u.
         The final error is thus bounded by (23 + 10*2^err_g) ulps,
         which is <= 2^6 for err_g<=2, and <= 2^(err_g+4) for err_g >= 2. */
      err_g = (err_g <= 2) ? 6 : err_g + 4;

      if (MPFR_LIKELY (MPFR_CAN_ROUND (GammaTrial, realprec - err_g,
                                       MPFR_PREC(gamma), rnd_mode)))
        break;

    ziv_next:
      MPFR_ZIV_NEXT (loop, realprec);
    }

 end:
  MPFR_ZIV_FREE (loop);

  if (inex == 0)
    inex = mpfr_set (gamma, GammaTrial, rnd_mode);
  MPFR_GROUP_CLEAR (group);
  mpz_clear (fact);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (gamma, inex, rnd_mode);
}
