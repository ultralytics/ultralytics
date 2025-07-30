/* mpfr_root -- kth root.

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

 /* The computation of y = x^(1/k) is done as follows, except for large
    values of k, for which this would be inefficient or yield internal
    integer overflows:

    Let x = sign * m * 2^(k*e) where m is an integer

    with 2^(k*(n-1)) <= m < 2^(k*n) where n = PREC(y)

    and m = s^k + t where 0 <= t and m < (s+1)^k

    we want that s has n bits i.e. s >= 2^(n-1), or m >= 2^(k*(n-1))
    i.e. m must have at least k*(n-1)+1 bits

    then, not taking into account the sign, the result will be
    x^(1/k) = s * 2^e or (s+1) * 2^e according to the rounding mode.
 */

static int
mpfr_root_aux (mpfr_ptr y, mpfr_srcptr x, unsigned long k,
               mpfr_rnd_t rnd_mode);

int
mpfr_root (mpfr_ptr y, mpfr_srcptr x, unsigned long k, mpfr_rnd_t rnd_mode)
{
  mpz_t m;
  mpfr_exp_t e, r, sh, f;
  mpfr_prec_t n, size_m, tmp;
  int inexact, negative;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg k=%lu rnd=%d",
      mpfr_get_prec (x), mpfr_log_prec, x, k, rnd_mode),
     ("y[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (y), mpfr_log_prec, y, inexact));

  if (MPFR_UNLIKELY (k <= 1))
    {
      if (k == 0)
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else /* y = x^(1/1) = x */
        return mpfr_set (y, x, rnd_mode);
    }

  /* Singular values */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (y); /* NaN^(1/k) = NaN */
          MPFR_RET_NAN;
        }

      if (MPFR_IS_INF (x)) /* +Inf^(1/k) = +Inf
                              -Inf^(1/k) = -Inf if k odd
                              -Inf^(1/k) = NaN if k even */
        {
          if (MPFR_IS_NEG(x) && (k % 2 == 0))
            {
              MPFR_SET_NAN (y);
              MPFR_RET_NAN;
            }
          MPFR_SET_INF (y);
        }
      else /* x is necessarily 0: (+0)^(1/k) = +0
                                  (-0)^(1/k) = -0 */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (x));
          MPFR_SET_ZERO (y);
        }
      MPFR_SET_SAME_SIGN (y, x);
      MPFR_RET (0);
    }

  /* Returns NAN for x < 0 and k even */
  if (MPFR_UNLIKELY (MPFR_IS_NEG (x) && (k % 2 == 0)))
    {
      MPFR_SET_NAN (y);
      MPFR_RET_NAN;
    }

  /* General case */

  /* For large k, use exp(log(x)/k). The threshold of 100 seems to be quite
     good when the precision goes to infinity. */
  if (k > 100)
    return mpfr_root_aux (y, x, k, rnd_mode);

  MPFR_SAVE_EXPO_MARK (expo);
  mpz_init (m);

  e = mpfr_get_z_2exp (m, x);                /* x = m * 2^e */
  if ((negative = MPFR_IS_NEG(x)))
    mpz_neg (m, m);
  r = e % (mpfr_exp_t) k;
  if (r < 0)
    r += k; /* now r = e (mod k) with 0 <= r < k */
  MPFR_ASSERTD (0 <= r && r < k);
  /* x = (m*2^r) * 2^(e-r) where e-r is a multiple of k */

  MPFR_MPZ_SIZEINBASE2 (size_m, m);
  /* for rounding to nearest, we want the round bit to be in the root */
  n = MPFR_PREC (y) + (rnd_mode == MPFR_RNDN);

  /* we now multiply m by 2^sh so that root(m,k) will give
     exactly n bits: we want k*(n-1)+1 <= size_m + sh <= k*n
     i.e. sh = k*f + r with f = max(floor((k*n-size_m-r)/k),0) */
  if ((mpfr_exp_t) size_m + r >= k * (mpfr_exp_t) n)
    f = 0; /* we already have too many bits */
  else
    f = (k * (mpfr_exp_t) n - (mpfr_exp_t) size_m - r) / k;
  sh = k * f + r;
  mpz_mul_2exp (m, m, sh);
  e = e - sh;

  /* invariant: x = m*2^e, with e divisible by k */

  /* we reuse the variable m to store the kth root, since it is not needed
     any more: we just need to know if the root is exact */
  inexact = mpz_root (m, m, k) == 0;

  MPFR_MPZ_SIZEINBASE2 (tmp, m);
  sh = tmp - n;
  if (sh > 0) /* we have to flush to 0 the last sh bits from m */
    {
      inexact = inexact || ((mpfr_exp_t) mpz_scan1 (m, 0) < sh);
      mpz_fdiv_q_2exp (m, m, sh);
      e += k * sh;
    }

  if (inexact)
    {
      if (negative)
        rnd_mode = MPFR_INVERT_RND (rnd_mode);
      if (rnd_mode == MPFR_RNDU || rnd_mode == MPFR_RNDA
          || (rnd_mode == MPFR_RNDN && mpz_tstbit (m, 0)))
        inexact = 1, mpz_add_ui (m, m, 1);
      else
        inexact = -1;
    }

  /* either inexact is not zero, and the conversion is exact, i.e. inexact
     is not changed; or inexact=0, and inexact is set only when
     rnd_mode=MPFR_RNDN and bit (n+1) from m is 1 */
  inexact += mpfr_set_z (y, m, MPFR_RNDN);
  MPFR_SET_EXP (y, MPFR_GET_EXP (y) + e / (mpfr_exp_t) k);

  if (negative)
    {
      MPFR_CHANGE_SIGN (y);
      inexact = -inexact;
    }

  mpz_clear (m);
  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, inexact, rnd_mode);
}

/* Compute y <- x^(1/k) using exp(log(x)/k).
   Assume all special cases have been eliminated before.
   In the extended exponent range, overflows/underflows are not possible.
   Assume x > 0, or x < 0 and k odd.
*/
static int
mpfr_root_aux (mpfr_ptr y, mpfr_srcptr x, unsigned long k, mpfr_rnd_t rnd_mode)
{
  int inexact, exact_root = 0;
  mpfr_prec_t w; /* working precision */
  mpfr_t absx, t;
  MPFR_GROUP_DECL(group);
  MPFR_TMP_DECL(marker);
  MPFR_ZIV_DECL(loop);
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_TMP_INIT_ABS (absx, x);

  MPFR_TMP_MARK(marker);
  w = MPFR_PREC(y) + 10;
  /* Take some guard bits to prepare for the 'expt' lost bits below.
     If |x| < 2^k, then log|x| < k, thus taking log2(k) bits should be fine. */
  if (MPFR_GET_EXP(x) > 0)
    w += MPFR_INT_CEIL_LOG2 (MPFR_GET_EXP(x));
  MPFR_GROUP_INIT_1(group, w, t);
  MPFR_SAVE_EXPO_MARK (expo);
  MPFR_ZIV_INIT (loop, w);
  for (;;)
    {
      mpfr_exp_t expt;
      unsigned int err;

      mpfr_log (t, absx, MPFR_RNDN);
      /* t = log|x| * (1 + theta) with |theta| <= 2^(-w) */
      mpfr_div_ui (t, t, k, MPFR_RNDN);
      expt = MPFR_GET_EXP (t);
      /* t = log|x|/k * (1 + theta) + eps with |theta| <= 2^(-w)
         and |eps| <= 1/2 ulp(t), thus the total error is bounded
         by 1.5 * 2^(expt - w) */
      mpfr_exp (t, t, MPFR_RNDN);
      /* t = |x|^(1/k) * exp(tau) * (1 + theta1) with
         |tau| <= 1.5 * 2^(expt - w) and |theta1| <= 2^(-w).
         For |tau| <= 0.5 we have |exp(tau)-1| < 4/3*tau, thus
         for w >= expt + 2 we have:
         t = |x|^(1/k) * (1 + 2^(expt+2)*theta2) * (1 + theta1) with
         |theta1|, |theta2| <= 2^(-w).
         If expt+2 > 0, as long as w >= 1, we have:
         t = |x|^(1/k) * (1 + 2^(expt+3)*theta3) with |theta3| < 2^(-w).
         For expt+2 = 0, we have:
         t = |x|^(1/k) * (1 + 2^2*theta3) with |theta3| < 2^(-w).
         Finally for expt+2 < 0 we have:
         t = |x|^(1/k) * (1 + 2*theta3) with |theta3| < 2^(-w).
      */
      err = (expt + 2 > 0) ? expt + 3
        : (expt + 2 == 0) ? 2 : 1;
      /* now t = |x|^(1/k) * (1 + 2^(err-w)) thus the error is at most
         2^(EXP(t) - w + err) */
      if (MPFR_LIKELY (MPFR_CAN_ROUND(t, w - err, MPFR_PREC(y), rnd_mode)))
        break;

      /* If we fail to round correctly, check for an exact result or a
         midpoint result with MPFR_RNDN (regarded as hard-to-round in
         all precisions in order to determine the ternary value). */
      {
        mpfr_t z, zk;

        mpfr_init2 (z, MPFR_PREC(y) + (rnd_mode == MPFR_RNDN));
        mpfr_init2 (zk, MPFR_PREC(x));
        mpfr_set (z, t, MPFR_RNDN);
        inexact = mpfr_pow_ui (zk, z, k, MPFR_RNDN);
        exact_root = !inexact && mpfr_equal_p (zk, absx);
        if (exact_root) /* z is the exact root, thus round z directly */
          inexact = mpfr_set4 (y, z, rnd_mode, MPFR_SIGN (x));
        mpfr_clear (zk);
        mpfr_clear (z);
        if (exact_root)
          break;
      }

      MPFR_ZIV_NEXT (loop, w);
      MPFR_GROUP_REPREC_1(group, w, t);
    }
  MPFR_ZIV_FREE (loop);

  if (!exact_root)
    inexact = mpfr_set4 (y, t, rnd_mode, MPFR_SIGN (x));

  MPFR_GROUP_CLEAR(group);
  MPFR_TMP_FREE(marker);
  MPFR_SAVE_EXPO_FREE (expo);

  return mpfr_check_range (y, inexact, rnd_mode);
}
