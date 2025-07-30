/* mpfr_sin_cos -- sine and cosine of a floating-point number

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

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

/* (y, z) <- (sin(x), cos(x)), return value is 0 iff both results are exact
   ie, iff x = 0 */
int
mpfr_sin_cos (mpfr_ptr y, mpfr_ptr z, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_prec_t prec, m;
  int neg, reduce;
  mpfr_t c, xr;
  mpfr_srcptr xx;
  mpfr_exp_t err, expx;
  int inexy, inexz;
  MPFR_ZIV_DECL (loop);
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_ASSERTN (y != z);

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN(x) || MPFR_IS_INF(x))
        {
          MPFR_SET_NAN (y);
          MPFR_SET_NAN (z);
          MPFR_RET_NAN;
        }
      else /* x is zero */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (x));
          MPFR_SET_ZERO (y);
          MPFR_SET_SAME_SIGN (y, x);
          /* y = 0, thus exact, but z is inexact in case of underflow
             or overflow */
          inexy = 0; /* y is exact */
          inexz = mpfr_set_ui (z, 1, rnd_mode);
          return INEX(inexy,inexz);
        }
    }

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd_mode),
     ("sin[%Pu]=%.*Rg cos[%Pu]=%.*Rg", mpfr_get_prec(y), mpfr_log_prec, y,
      mpfr_get_prec (z), mpfr_log_prec, z));

  MPFR_SAVE_EXPO_MARK (expo);

  prec = MAX (MPFR_PREC (y), MPFR_PREC (z));
  m = prec + MPFR_INT_CEIL_LOG2 (prec) + 13;
  expx = MPFR_GET_EXP (x);

  /* When x is close to 0, say 2^(-k), then there is a cancellation of about
     2k bits in 1-cos(x)^2. FIXME: in that case, it would be more efficient
     to compute sin(x) directly. VL: This is partly done by using
     MPFR_FAST_COMPUTE_IF_SMALL_INPUT from the mpfr_sin and mpfr_cos
     functions. Moreover, any overflow on m is avoided. */
  if (expx < 0)
    {
      /* Warning: in case y = x, and the first call to
         MPFR_FAST_COMPUTE_IF_SMALL_INPUT succeeds but the second fails,
         we will have clobbered the original value of x.
         The workaround is to first compute z = cos(x) in that case, since
         y and z are different. */
      if (y != x)
        /* y and x differ, thus we can safely try to compute y first */
        {
          MPFR_FAST_COMPUTE_IF_SMALL_INPUT (
            y, x, -2 * expx, 2, 0, rnd_mode,
            { inexy = _inexact;
              goto small_input; });
          if (0)
            {
            small_input:
              /* we can go here only if we can round sin(x) */
              MPFR_FAST_COMPUTE_IF_SMALL_INPUT (
                z, __gmpfr_one, -2 * expx, 1, 0, rnd_mode,
                { inexz = _inexact;
                  MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);
                  goto end; });
            }

          /* if we go here, one of the two MPFR_FAST_COMPUTE_IF_SMALL_INPUT
             calls failed */
        }
      else /* y and x are the same variable: try to compute z first, which
              necessarily differs */
        {
          MPFR_FAST_COMPUTE_IF_SMALL_INPUT (
            z, __gmpfr_one, -2 * expx, 1, 0, rnd_mode,
            { inexz = _inexact;
              goto small_input2; });
          if (0)
            {
            small_input2:
              /* we can go here only if we can round cos(x) */
              MPFR_FAST_COMPUTE_IF_SMALL_INPUT (
                y, x, -2 * expx, 2, 0, rnd_mode,
                { inexy = _inexact;
                  MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);
                  goto end; });
            }
        }
      m += 2 * (-expx);
    }

  if (prec >= MPFR_SINCOS_THRESHOLD)
    {
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_sincos_fast (y, z, x, rnd_mode);
    }

  mpfr_init (c);
  mpfr_init (xr);

  MPFR_ZIV_INIT (loop, m);
  for (;;)
    {
      /* the following is copied from sin.c */
      if (expx >= 2) /* reduce the argument */
        {
          reduce = 1;
          mpfr_set_prec (c, expx + m - 1);
          mpfr_set_prec (xr, m);
          mpfr_const_pi (c, MPFR_RNDN);
          mpfr_mul_2ui (c, c, 1, MPFR_RNDN);
          mpfr_remainder (xr, x, c, MPFR_RNDN);
          mpfr_div_2ui (c, c, 1, MPFR_RNDN);
          if (MPFR_SIGN (xr) > 0)
            mpfr_sub (c, c, xr, MPFR_RNDZ);
          else
            mpfr_add (c, c, xr, MPFR_RNDZ);
          if (MPFR_IS_ZERO(xr)
              || MPFR_EXP(xr) < (mpfr_exp_t) 3 - (mpfr_exp_t) m
              || MPFR_EXP(c) < (mpfr_exp_t) 3 - (mpfr_exp_t) m)
            goto next_step;
          xx = xr;
        }
      else /* the input argument is already reduced */
        {
          reduce = 0;
          xx = x;
        }

      neg = MPFR_IS_NEG (xx); /* gives sign of sin(x) */
      mpfr_set_prec (c, m);
      mpfr_cos (c, xx, MPFR_RNDZ);
      /* If no argument reduction was performed, the error is at most ulp(c),
         otherwise it is at most ulp(c) + 2^(2-m). Since |c| < 1, we have
         ulp(c) <= 2^(-m), thus the error is bounded by 2^(3-m) in that later
         case. */
      if (reduce == 0)
        err = m;
      else
        err = MPFR_GET_EXP (c) + (mpfr_exp_t) (m - 3);
      if (!mpfr_can_round (c, err, MPFR_RNDN, MPFR_RNDZ,
                           MPFR_PREC (z) + (rnd_mode == MPFR_RNDN)))
        goto next_step;

      /* we can't set z now, because in case z = x, and the mpfr_can_round()
         call below fails, we will have clobbered the input */
      mpfr_set_prec (xr, MPFR_PREC(c));
      mpfr_swap (xr, c); /* save the approximation of the cosine in xr */
      mpfr_sqr (c, xr, MPFR_RNDU); /* the absolute error is bounded by
                                      2^(5-m) if reduce=1, and by 2^(2-m)
                                      otherwise */
      mpfr_ui_sub (c, 1, c, MPFR_RNDN); /* error bounded by 2^(6-m) if reduce
                                           is 1, and 2^(3-m) otherwise */
      mpfr_sqrt (c, c, MPFR_RNDN); /* the absolute error is bounded by
                                      2^(6-m-Exp(c)) if reduce=1, and
                                      2^(3-m-Exp(c)) otherwise */
      err = 3 + 3 * reduce - MPFR_GET_EXP (c);
      if (neg)
        MPFR_CHANGE_SIGN (c);

      /* the absolute error on c is at most 2^(err-m), which we must put
         in the form 2^(EXP(c)-err). */
      err = MPFR_GET_EXP (c) + (mpfr_exp_t) m - err;
      if (mpfr_can_round (c, err, MPFR_RNDN, MPFR_RNDZ,
                          MPFR_PREC (y) + (rnd_mode == MPFR_RNDN)))
        break;
      /* check for huge cancellation */
      if (err < (mpfr_exp_t) MPFR_PREC (y))
        m += MPFR_PREC (y) - err;
      /* Check if near 1 */
      if (MPFR_GET_EXP (c) == 1
          && MPFR_MANT (c)[MPFR_LIMB_SIZE (c)-1] == MPFR_LIMB_HIGHBIT)
        m += m;

    next_step:
      MPFR_ZIV_NEXT (loop, m);
      mpfr_set_prec (c, m);
    }
  MPFR_ZIV_FREE (loop);

  inexy = mpfr_set (y, c, rnd_mode);
  inexz = mpfr_set (z, xr, rnd_mode);

  mpfr_clear (c);
  mpfr_clear (xr);

 end:
  MPFR_SAVE_EXPO_FREE (expo);
  /* FIXME: add a test for bug before revision 7355 */
  inexy = mpfr_check_range (y, inexy, rnd_mode);
  inexz = mpfr_check_range (z, inexz, rnd_mode);
  MPFR_RET (INEX(inexy,inexz));
}

/*************** asymptotically fast implementation below ********************/

/* truncate Q from R to at most prec bits.
   Return the number of truncated bits.
 */
static mpfr_prec_t
reduce (mpz_t Q, mpz_srcptr R, mpfr_prec_t prec)
{
  mpfr_prec_t l = mpz_sizeinbase (R, 2);

  l = (l > prec) ? l - prec : 0;
  mpz_fdiv_q_2exp (Q, R, l);
  return l;
}

/* truncate S and C so that the smaller has prec bits.
   Return the number of truncated bits.
 */
static unsigned long
reduce2 (mpz_t S, mpz_t C, mpfr_prec_t prec)
{
  unsigned long ls = mpz_sizeinbase (S, 2);
  unsigned long lc = mpz_sizeinbase (C, 2);
  unsigned long l;

  l = (ls < lc) ? ls : lc; /* smaller length */
  l = (l > prec) ? l - prec : 0;
  mpz_fdiv_q_2exp (S, S, l);
  mpz_fdiv_q_2exp (C, C, l);
  return l;
}

/* return in S0/Q0 a rational approximation of sin(X) with absolute error
                     bounded by 9*2^(-prec), where 0 <= X=p/2^r <= 1/2,
   and in    C0/Q0 a rational approximation of cos(X), with relative error
                     bounded by 9*2^(-prec) (and also absolute error, since
                     |cos(X)| <= 1).
   We have sin(X)/X = sum((-1)^i*(p/2^r)^i/(2i+1)!, i=0..infinity).
   We use the following binary splitting formula:
   P(a,b) = (-p)^(b-a)
   Q(a,b) = (2a)*(2a+1)*2^r if a+1=b [except Q(0,1)=1], Q(a,c)*Q(c,b) otherwise
   T(a,b) = 1 if a+1=b, Q(c,b)*T(a,c)+P(a,c)*T(c,b) otherwise.

   Since we use P(a,b) for b-a=2^k only, we compute only p^(2^k).
   We do not store the factor 2^r in Q().

   Then sin(X)/X ~ T(0,i)/Q(0,i) for i so that (p/2^r)^i/i! is small enough.

   Return l such that Q0 has to be multiplied by 2^l.

   Assumes prec >= 10.
*/
static unsigned long
sin_bs_aux (mpz_t Q0, mpz_t S0, mpz_t C0, mpz_srcptr p, mpfr_prec_t r,
            mpfr_prec_t prec)
{
  mpz_t T[GMP_NUMB_BITS], Q[GMP_NUMB_BITS], ptoj[GMP_NUMB_BITS], pp;
  mpfr_prec_t log2_nb_terms[GMP_NUMB_BITS], mult[GMP_NUMB_BITS];
  mpfr_prec_t accu[GMP_NUMB_BITS], size_ptoj[GMP_NUMB_BITS];
  mpfr_prec_t prec_i_have, r0 = r;
  unsigned long alloc, i, j, k;
  mpfr_prec_t l;

  if (MPFR_UNLIKELY(mpz_cmp_ui (p, 0) == 0)) /* sin(x)/x -> 1 */
    {
      mpz_set_ui (Q0, 1);
      mpz_set_ui (S0, 1);
      mpz_set_ui (C0, 1);
      return 0;
    }

  /* check that X=p/2^r <= 1/2 */
  MPFR_ASSERTN(mpz_sizeinbase (p, 2) - (mpfr_exp_t) r <= -1);

  mpz_init (pp);

  /* normalize p (non-zero here) */
  l = mpz_scan1 (p, 0);
  mpz_fdiv_q_2exp (pp, p, l); /* p = pp * 2^l */
  mpz_mul (pp, pp, pp);
  r = 2 * (r - l);            /* x^2 = (p/2^r0)^2 = pp / 2^r */

  /* now p is odd */
  alloc = 2;
  mpz_init_set_ui (T[0], 6);
  mpz_init_set_ui (Q[0], 6);
  mpz_init_set (ptoj[0], pp); /* ptoj[i] = pp^(2^i) */
  mpz_init (T[1]);
  mpz_init (Q[1]);
  mpz_init (ptoj[1]);
  mpz_mul (ptoj[1], pp, pp);  /* ptoj[1] = pp^2 */
  size_ptoj[1] = mpz_sizeinbase (ptoj[1], 2);

  mpz_mul_2exp (T[0], T[0], r);
  mpz_sub (T[0], T[0], pp);      /* 6*2^r - pp = 6*2^r*(1 - x^2/6) */
  log2_nb_terms[0] = 1;

  /* already take into account the factor x=p/2^r in sin(x) = x * (...) */
  mult[0] = r  - mpz_sizeinbase (pp, 2) + r0 - mpz_sizeinbase (p, 2);
  /* we have x^3 < 1/2^mult[0] */

  for (i = 2, k = 0, prec_i_have = mult[0]; prec_i_have < prec; i += 2)
    {
      /* i is even here */
      /* invariant: Q[0]*Q[1]*...*Q[k] equals (2i-1)!,
         we have already summed terms of index < i
         in S[0]/Q[0], ..., S[k]/Q[k] */
      k ++;
      if (k + 1 >= alloc) /* necessarily k + 1 = alloc */
        {
          alloc ++;
          mpz_init (T[k+1]);
          mpz_init (Q[k+1]);
          mpz_init (ptoj[k+1]);
          mpz_mul (ptoj[k+1], ptoj[k], ptoj[k]); /* pp^(2^(k+1)) */
          size_ptoj[k+1] = mpz_sizeinbase (ptoj[k+1], 2);
        }
      /* for i even, we have Q[k] = (2*i)*(2*i+1), T[k] = 1,
         then                Q[k+1] = (2*i+2)*(2*i+3), T[k+1] = 1,
         which reduces to T[k] = (2*i+2)*(2*i+3)*2^r-pp,
         Q[k] = (2*i)*(2*i+1)*(2*i+2)*(2*i+3). */
      log2_nb_terms[k] = 1;
      mpz_set_ui (Q[k], 2 * i + 2);
      mpz_mul_ui (Q[k], Q[k], 2 * i + 3);
      mpz_mul_2exp (T[k], Q[k], r);
      mpz_sub (T[k], T[k], pp);
      mpz_mul_ui (Q[k], Q[k], 2 * i);
      mpz_mul_ui (Q[k], Q[k], 2 * i + 1);
      /* the next term of the series is divided by Q[k] and multiplied
         by pp^2/2^(2r), thus the mult. factor < 1/2^mult[k] */
      mult[k] = mpz_sizeinbase (Q[k], 2) + 2 * r - size_ptoj[1] - 1;
      /* the absolute contribution of the next term is 1/2^accu[k] */
      accu[k] = (k == 0) ? mult[k] : mult[k] + accu[k-1];
      prec_i_have = accu[k]; /* the current term is < 1/2^accu[k] */
      j = (i + 2) / 2;
      l = 1;
      while ((j & 1) == 0) /* combine and reduce */
        {
          mpz_mul (T[k], T[k], ptoj[l]);
          mpz_mul (T[k-1], T[k-1], Q[k]);
          mpz_mul_2exp (T[k-1], T[k-1], r << l);
          mpz_add (T[k-1], T[k-1], T[k]);
          mpz_mul (Q[k-1], Q[k-1], Q[k]);
          log2_nb_terms[k-1] ++; /* number of terms in S[k-1]
                                    is a power of 2 by construction */
          prec_i_have = mpz_sizeinbase (Q[k], 2);
          mult[k-1] += prec_i_have + (r << l) - size_ptoj[l] - 1;
          accu[k-1] = (k == 1) ? mult[k-1] : mult[k-1] + accu[k-2];
          prec_i_have = accu[k-1];
          l ++;
          j >>= 1;
          k --;
        }
    }

  /* accumulate all products in T[0] and Q[0]. Warning: contrary to above,
     here we do not have log2_nb_terms[k-1] = log2_nb_terms[k]+1. */
  l = 0; /* number of accumulated terms in the right part T[k]/Q[k] */
  while (k > 0)
    {
      j = log2_nb_terms[k-1];
      mpz_mul (T[k], T[k], ptoj[j]);
      mpz_mul (T[k-1], T[k-1], Q[k]);
      l += 1 << log2_nb_terms[k];
      mpz_mul_2exp (T[k-1], T[k-1], r * l);
      mpz_add (T[k-1], T[k-1], T[k]);
      mpz_mul (Q[k-1], Q[k-1], Q[k]);
      k--;
    }

  l = r0 + r * (i - 1); /* implicit multiplier 2^r for Q0 */
  /* at this point T[0]/(2^l*Q[0]) is an approximation of sin(x) where the 1st
     neglected term has contribution < 1/2^prec, thus since the series has
     alternate signs, the error is < 1/2^prec */

  /* we truncate Q0 to prec bits: the relative error is at most 2^(1-prec),
     which means that Q0 = Q[0] * (1+theta) with |theta| <= 2^(1-prec)
     [up to a power of two] */
  l += reduce (Q0, Q[0], prec);
  l -= reduce (T[0], T[0], prec);
  /* multiply by x = p/2^l */
  mpz_mul (S0, T[0], p);
  l -= reduce (S0, S0, prec); /* S0 = T[0] * (1 + theta)^2 up to power of 2 */
  /* sin(X) ~ S0/Q0*(1 + theta)^3 + err with |theta| <= 2^(1-prec) and
              |err| <= 2^(-prec), thus since |S0/Q0| <= 1:
     |sin(X) - S0/Q0| <= 4*|theta*S0/Q0| + |err| <= 9*2^(-prec) */

  mpz_clear (pp);
  for (j = 0; j < alloc; j ++)
    {
      mpz_clear (T[j]);
      mpz_clear (Q[j]);
      mpz_clear (ptoj[j]);
    }

  /* compute cos(X) from sin(X): sqrt(1-(S/Q)^2) = sqrt(Q^2-S^2)/Q
     = sqrt(Q0^2*2^(2l)-S0^2)/Q0.
     Write S/Q = sin(X) + eps with |eps| <= 9*2^(-prec),
     then sqrt(Q^2-S^2) = sqrt(Q^2-Q^2*(sin(X)+eps)^2)
                        = sqrt(Q^2*cos(X)^2-Q^2*(2*sin(X)*eps+eps^2))
                        = sqrt(Q^2*cos(X)^2-Q^2*eps1) with |eps1|<=9*2^(-prec)
                          [using X<=1/2 and eps<=9*2^(-prec) and prec>=10]

                        Since we truncate the square root, we get:
                          sqrt(Q^2*cos(X)^2-Q^2*eps1)+eps2 with |eps2|<1
                        = Q*sqrt(cos(X)^2-eps1)+eps2
                        = Q*cos(X)*(1+eps3)+eps2 with |eps3| <= 6*2^(-prec)
                        = Q*cos(X)*(1+eps3+eps2/(Q*cos(X)))
                        = Q*cos(X)*(1+eps4) with |eps4| <= 9*2^(-prec)
                          since |Q| >= 2^(prec-1) */
  /* we assume that Q0*2^l >= 2^(prec-1) */
  MPFR_ASSERTN(l + mpz_sizeinbase (Q0, 2) >= prec);
  mpz_mul (C0, Q0, Q0);
  mpz_mul_2exp (C0, C0, 2 * l);
  mpz_submul (C0, S0, S0);
  mpz_sqrt (C0, C0);

  return l;
}

/* Put in s and c approximations of sin(x) and cos(x) respectively.
   Assumes 0 < x < Pi/4 and PREC(s) = PREC(c) >= 10.
   Return err such that the relative error is bounded by 2^err ulps.
*/
static int
sincos_aux (mpfr_t s, mpfr_t c, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_prec_t prec_s, sh;
  mpz_t Q, S, C, Q2, S2, C2, y;
  mpfr_t x2;
  unsigned long l, l2, j, err;

  MPFR_ASSERTD(MPFR_PREC(s) == MPFR_PREC(c));

  prec_s = MPFR_PREC(s);

  mpfr_init2 (x2, MPFR_PREC(x));
  mpz_init (Q);
  mpz_init (S);
  mpz_init (C);
  mpz_init (Q2);
  mpz_init (S2);
  mpz_init (C2);
  mpz_init (y);

  mpfr_set (x2, x, MPFR_RNDN); /* exact */
  mpz_set_ui (Q, 1);
  l = 0;
  mpz_set_ui (S, 0); /* sin(0) = S/(2^l*Q), exact */
  mpz_set_ui (C, 1); /* cos(0) = C/(2^l*Q), exact */

  /* Invariant: x = X + x2/2^(sh-1), where the part X was already treated,
     S/(2^l*Q) ~ sin(X), C/(2^l*Q) ~ cos(X), and x2/2^(sh-1) < Pi/4.
     'sh-1' is the number of already shifted bits in x2.
  */

  for (sh = 1, j = 0; mpfr_cmp_ui (x2, 0) != 0 && sh <= prec_s; sh <<= 1, j++)
    {
      if (sh > prec_s / 2) /* sin(x) = x + O(x^3), cos(x) = 1 + O(x^2) */
        {
          l2 = -mpfr_get_z_2exp (S2, x2); /* S2/2^l2 = x2 */
          l2 += sh - 1;
          mpz_set_ui (Q2, 1);
          mpz_set_ui (C2, 1);
          mpz_mul_2exp (C2, C2, l2);
          mpfr_set_ui (x2, 0, MPFR_RNDN);
        }
      else
        {
          /* y <- trunc(x2 * 2^sh) = trunc(x * 2^(2*sh-1)) */
          mpfr_mul_2exp (x2, x2, sh, MPFR_RNDN); /* exact */
          mpfr_get_z (y, x2, MPFR_RNDZ); /* round towards zero: now
                                           0 <= x2 < 2^sh, thus
                                           0 <= x2/2^(sh-1) < 2^(1-sh) */
          if (mpz_cmp_ui (y, 0) == 0)
            continue;
          mpfr_sub_z (x2, x2, y, MPFR_RNDN); /* should be exact */
          l2 = sin_bs_aux (Q2, S2, C2, y, 2 * sh - 1, prec_s);
          /* we now have |S2/Q2/2^l2 - sin(X)| <= 9*2^(prec_s)
             and |C2/Q2/2^l2 - cos(X)| <= 6*2^(prec_s), with X=y/2^(2sh-1) */
        }
      if (sh == 1) /* S=0, C=1 */
        {
          l = l2;
          mpz_swap (Q, Q2);
          mpz_swap (S, S2);
          mpz_swap (C, C2);
        }
      else
        {
          /* s <- s*c2+c*s2, c <- c*c2-s*s2, using Karatsuba:
             a = s+c, b = s2+c2, t = a*b, d = s*s2, e = c*c2,
             s <- t - d - e, c <- e - d */
          mpz_add (y, S, C); /* a */
          mpz_mul (C, C, C2); /* e */
          mpz_add (C2, C2, S2); /* b */
          mpz_mul (S2, S, S2); /* d */
          mpz_mul (y, y, C2); /* a*b */
          mpz_sub (S, y, S2); /* t - d */
          mpz_sub (S, S, C); /* t - d - e */
          mpz_sub (C, C, S2); /* e - d */
          mpz_mul (Q, Q, Q2);
          /* after j loops, the error is <= (11j-2)*2^(prec_s) */
          l += l2;
          /* reduce Q to prec_s bits */
          l += reduce (Q, Q, prec_s);
          /* reduce S,C to prec_s bits, error <= 11*j*2^(prec_s) */
          l -= reduce2 (S, C, prec_s);
        }
    }

  j = 11 * j;
  for (err = 0; j > 1; j = (j + 1) / 2, err ++);

  mpfr_set_z (s, S, MPFR_RNDN);
  mpfr_div_z (s, s, Q, MPFR_RNDN);
  mpfr_div_2exp (s, s, l, MPFR_RNDN);

  mpfr_set_z (c, C, MPFR_RNDN);
  mpfr_div_z (c, c, Q, MPFR_RNDN);
  mpfr_div_2exp (c, c, l, MPFR_RNDN);

  mpz_clear (Q);
  mpz_clear (S);
  mpz_clear (C);
  mpz_clear (Q2);
  mpz_clear (S2);
  mpz_clear (C2);
  mpz_clear (y);
  mpfr_clear (x2);
  return err;
}

/* Assumes x is neither NaN, +/-Inf, nor +/- 0.
   One of s and c might be NULL, in which case the corresponding value is
   not computed.
   Assumes s differs from c.
 */
int
mpfr_sincos_fast (mpfr_t s, mpfr_t c, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  int inexs, inexc;
  mpfr_t x_red, ts, tc;
  mpfr_prec_t w;
  mpfr_exp_t err, errs, errc;
  MPFR_ZIV_DECL (loop);

  MPFR_ASSERTN(s != c);
  if (s == NULL)
    w = MPFR_PREC(c);
  else if (c == NULL)
    w = MPFR_PREC(s);
  else
    w = MPFR_PREC(s) >= MPFR_PREC(c) ? MPFR_PREC(s) : MPFR_PREC(c);
  w += MPFR_INT_CEIL_LOG2(w) + 9; /* ensures w >= 10 (needed by sincos_aux) */
  mpfr_init2 (ts, w);
  mpfr_init2 (tc, w);

  MPFR_ZIV_INIT (loop, w);
  for (;;)
    {
      /* if 0 < x <= Pi/4, we can call sincos_aux directly */
      if (MPFR_IS_POS(x) && mpfr_cmp_ui_2exp (x, 1686629713, -31) <= 0)
        {
          err = sincos_aux (ts, tc, x, MPFR_RNDN);
        }
      /* if -Pi/4 <= x < 0, use sin(-x)=-sin(x) */
      else if (MPFR_IS_NEG(x) && mpfr_cmp_si_2exp (x, -1686629713, -31) >= 0)
        {
          mpfr_init2 (x_red, MPFR_PREC(x));
          mpfr_neg (x_red, x, rnd); /* exact */
          err = sincos_aux (ts, tc, x_red, MPFR_RNDN);
          mpfr_neg (ts, ts, MPFR_RNDN);
          mpfr_clear (x_red);
        }
      else /* argument reduction is needed */
        {
          long q;
          mpfr_t pi;
          int neg = 0;

          mpfr_init2 (x_red, w);
          mpfr_init2 (pi, (MPFR_EXP(x) > 0) ? w + MPFR_EXP(x) : w);
          mpfr_const_pi (pi, MPFR_RNDN);
          mpfr_div_2exp (pi, pi, 1, MPFR_RNDN); /* Pi/2 */
          mpfr_remquo (x_red, &q, x, pi, MPFR_RNDN);
          /* x = q * (Pi/2 + eps1) + x_red + eps2,
             where |eps1| <= 1/2*ulp(Pi/2) = 2^(-w-MAX(0,EXP(x))),
             and eps2 <= 1/2*ulp(x_red) <= 1/2*ulp(Pi/2) = 2^(-w)
             Since |q| <= x/(Pi/2) <= |x|, we have
             q*|eps1| <= 2^(-w), thus
             |x - q * Pi/2 - x_red| <= 2^(1-w) */
          /* now -Pi/4 <= x_red <= Pi/4: if x_red < 0, consider -x_red */
          if (MPFR_IS_NEG(x_red))
            {
              mpfr_neg (x_red, x_red, MPFR_RNDN);
              neg = 1;
            }
          err = sincos_aux (ts, tc, x_red, MPFR_RNDN);
          err ++; /* to take into account the argument reduction */
          if (neg) /* sin(-x) = -sin(x), cos(-x) = cos(x) */
            mpfr_neg (ts, ts, MPFR_RNDN);
          if (q & 2) /* sin(x+Pi) = -sin(x), cos(x+Pi) = -cos(x) */
            {
              mpfr_neg (ts, ts, MPFR_RNDN);
              mpfr_neg (tc, tc, MPFR_RNDN);
            }
          if (q & 1) /* sin(x+Pi/2) = cos(x), cos(x+Pi/2) = -sin(x) */
            {
              mpfr_neg (ts, ts, MPFR_RNDN);
              mpfr_swap (ts, tc);
            }
          mpfr_clear (x_red);
          mpfr_clear (pi);
        }
      /* adjust errors with respect to absolute values */
      errs = err - MPFR_EXP(ts);
      errc = err - MPFR_EXP(tc);
      if ((s == NULL || MPFR_CAN_ROUND (ts, w - errs, MPFR_PREC(s), rnd)) &&
          (c == NULL || MPFR_CAN_ROUND (tc, w - errc, MPFR_PREC(c), rnd)))
        break;
      MPFR_ZIV_NEXT (loop, w);
      mpfr_set_prec (ts, w);
      mpfr_set_prec (tc, w);
    }
  MPFR_ZIV_FREE (loop);

  inexs = (s == NULL) ? 0 : mpfr_set (s, ts, rnd);
  inexc = (c == NULL) ? 0 : mpfr_set (c, tc, rnd);

  mpfr_clear (ts);
  mpfr_clear (tc);
  return INEX(inexs,inexc);
}
