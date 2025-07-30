/* mpfr_exp -- exponential of a floating-point number

Copyright 1999, 2001-2017 Free Software Foundation, Inc.
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

#define MPFR_NEED_LONGLONG_H /* for MPFR_MPZ_SIZEINBASE2 */
#include "mpfr-impl.h"

/* y <- exp(p/2^r) within 1 ulp, using 2^m terms from the series
   Assume |p/2^r| < 1.
   We use the following binary splitting formula:
   P(a,b) = p if a+1=b, P(a,c)*P(c,b) otherwise
   Q(a,b) = a*2^r if a+1=b [except Q(0,1)=1], Q(a,c)*Q(c,b) otherwise
   T(a,b) = P(a,b) if a+1=b, Q(c,b)*T(a,c)+P(a,c)*T(c,b) otherwise
   Then exp(p/2^r) ~ T(0,i)/Q(0,i) for i so that (p/2^r)^i/i! is small enough.

   Since P(a,b) = p^(b-a), and we consider only values of b-a of the form 2^j,
   we don't need to compute P(), we only precompute p^(2^j) in the ptoj[] array
   below.

   Since Q(a,b) is divisible by 2^(r*(b-a-1)), we don't compute the power of
   two part.
*/
static void
mpfr_exp_rational (mpfr_ptr y, mpz_ptr p, long r, int m,
                   mpz_t *Q, mpfr_prec_t *mult)
{
  unsigned long n, i, j;
  mpz_t *S, *ptoj;
  mpfr_prec_t *log2_nb_terms;
  mpfr_exp_t diff, expo;
  mpfr_prec_t precy = MPFR_PREC(y), prec_i_have, prec_ptoj;
  int k, l;

  MPFR_ASSERTN ((size_t) m < sizeof (long) * CHAR_BIT - 1);

  S    = Q + (m+1);
  ptoj = Q + 2*(m+1);                     /* ptoj[i] = mantissa^(2^i) */
  log2_nb_terms = mult + (m+1);

  /* Normalize p */
  MPFR_ASSERTD (mpz_cmp_ui (p, 0) != 0);
  n = mpz_scan1 (p, 0); /* number of trailing zeros in p */
  mpz_tdiv_q_2exp (p, p, n);
  r -= n; /* since |p/2^r| < 1 and p >= 1, r >= 1 */

  /* Set initial var */
  mpz_set (ptoj[0], p);
  for (k = 1; k < m; k++)
    mpz_mul (ptoj[k], ptoj[k-1], ptoj[k-1]); /* ptoj[k] = p^(2^k) */
  mpz_set_ui (Q[0], 1);
  mpz_set_ui (S[0], 1);
  k = 0;
  mult[0] = 0; /* the multiplier P[k]/Q[k] for the remaining terms
                  satisfies P[k]/Q[k] <= 2^(-mult[k]) */
  log2_nb_terms[0] = 0; /* log2(#terms) [exact in 1st loop where 2^k] */
  prec_i_have = 0;

  /* Main Loop */
  n = 1UL << m;
  for (i = 1; (prec_i_have < precy) && (i < n); i++)
    {
      /* invariant: Q[0]*Q[1]*...*Q[k] equals i! */
      k++;
      log2_nb_terms[k] = 0; /* 1 term */
      mpz_set_ui (Q[k], i + 1);
      mpz_set_ui (S[k], i + 1);
      j = i + 1; /* we have computed j = i+1 terms so far */
      l = 0;
      while ((j & 1) == 0) /* combine and reduce */
        {
          /* invariant: S[k] corresponds to 2^l consecutive terms */
          mpz_mul (S[k], S[k], ptoj[l]);
          mpz_mul (S[k-1], S[k-1], Q[k]);
          /* Q[k] corresponds to 2^l consecutive terms too.
             Since it does not contains the factor 2^(r*2^l),
             when going from l to l+1 we need to multiply
             by 2^(r*2^(l+1))/2^(r*2^l) = 2^(r*2^l) */
          mpz_mul_2exp (S[k-1], S[k-1], r << l);
          mpz_add (S[k-1], S[k-1], S[k]);
          mpz_mul (Q[k-1], Q[k-1], Q[k]);
          log2_nb_terms[k-1] ++; /* number of terms in S[k-1]
                                    is a power of 2 by construction */
          MPFR_MPZ_SIZEINBASE2 (prec_i_have, Q[k]);
          MPFR_MPZ_SIZEINBASE2 (prec_ptoj, ptoj[l]);
          mult[k-1] += prec_i_have + (r << l) - prec_ptoj - 1;
          prec_i_have = mult[k] = mult[k-1];
          /* since mult[k] >= mult[k-1] + nbits(Q[k]),
             we have Q[0]*...*Q[k] <= 2^mult[k] = 2^prec_i_have */
          l ++;
          j >>= 1;
          k --;
        }
    }

  /* accumulate all products in S[0] and Q[0]. Warning: contrary to above,
     here we do not have log2_nb_terms[k-1] = log2_nb_terms[k]+1. */
  l = 0; /* number of accumulated terms in the right part S[k]/Q[k] */
  while (k > 0)
    {
      j = log2_nb_terms[k-1];
      mpz_mul (S[k], S[k], ptoj[j]);
      mpz_mul (S[k-1], S[k-1], Q[k]);
      l += 1 << log2_nb_terms[k];
      mpz_mul_2exp (S[k-1], S[k-1], r * l);
      mpz_add (S[k-1], S[k-1], S[k]);
      mpz_mul (Q[k-1], Q[k-1], Q[k]);
      k--;
    }

  /* Q[0] now equals i! */
  MPFR_MPZ_SIZEINBASE2 (prec_i_have, S[0]);
  diff = (mpfr_exp_t) prec_i_have - 2 * (mpfr_exp_t) precy;
  expo = diff;
  if (diff >= 0)
    mpz_fdiv_q_2exp (S[0], S[0], diff);
  else
    mpz_mul_2exp (S[0], S[0], -diff);

  MPFR_MPZ_SIZEINBASE2 (prec_i_have, Q[0]);
  diff = (mpfr_exp_t) prec_i_have - (mpfr_prec_t) precy;
  expo -= diff;
  if (diff > 0)
    mpz_fdiv_q_2exp (Q[0], Q[0], diff);
  else
    mpz_mul_2exp (Q[0], Q[0], -diff);

  mpz_tdiv_q (S[0], S[0], Q[0]);
  mpfr_set_z (y, S[0], MPFR_RNDD);
  MPFR_SET_EXP (y, MPFR_GET_EXP (y) + expo - r * (i - 1) );
}

#define shift (GMP_NUMB_BITS/2)

int
mpfr_exp_3 (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_t t, x_copy, tmp;
  mpz_t uk;
  mpfr_exp_t ttt, shift_x;
  unsigned long twopoweri;
  mpz_t *P;
  mpfr_prec_t *mult;
  int i, k, loop;
  int prec_x;
  mpfr_prec_t realprec, Prec;
  int iter;
  int inexact = 0;
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_ZIV_DECL (ziv_loop);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec(x), mpfr_log_prec, x, rnd_mode),
     ("y[%Pu]=%.*Rg inexact=%d", mpfr_get_prec(y), mpfr_log_prec, y,
      inexact));

  MPFR_SAVE_EXPO_MARK (expo);

  /* decompose x */
  /* we first write x = 1.xxxxxxxxxxxxx
                        ----- k bits -- */
  prec_x = MPFR_INT_CEIL_LOG2 (MPFR_PREC (x)) - MPFR_LOG2_GMP_NUMB_BITS;
  if (prec_x < 0)
    prec_x = 0;

  ttt = MPFR_GET_EXP (x);
  mpfr_init2 (x_copy, MPFR_PREC(x));
  mpfr_set (x_copy, x, MPFR_RNDD);

  /* we shift to get a number less than 1 */
  if (ttt > 0)
    {
      shift_x = ttt;
      mpfr_div_2ui (x_copy, x, ttt, MPFR_RNDN);
      ttt = MPFR_GET_EXP (x_copy);
    }
  else
    shift_x = 0;
  MPFR_ASSERTD (ttt <= 0);

  /* Init prec and vars */
  realprec = MPFR_PREC (y) + MPFR_INT_CEIL_LOG2 (prec_x + MPFR_PREC (y));
  Prec = realprec + shift + 2 + shift_x;
  mpfr_init2 (t, Prec);
  mpfr_init2 (tmp, Prec);
  mpz_init (uk);

  /* Main loop */
  MPFR_ZIV_INIT (ziv_loop, realprec);
  for (;;)
    {
      int scaled = 0;
      MPFR_BLOCK_DECL (flags);

      k = MPFR_INT_CEIL_LOG2 (Prec) - MPFR_LOG2_GMP_NUMB_BITS;

      /* now we have to extract */
      twopoweri = GMP_NUMB_BITS;

      /* Allocate tables */
      P    = (mpz_t*) (*__gmp_allocate_func) (3*(k+2)*sizeof(mpz_t));
      for (i = 0; i < 3*(k+2); i++)
        mpz_init (P[i]);
      mult = (mpfr_prec_t*) (*__gmp_allocate_func) (2*(k+2)*sizeof(mpfr_prec_t));

      /* Particular case for i==0 */
      mpfr_extract (uk, x_copy, 0);
      MPFR_ASSERTD (mpz_cmp_ui (uk, 0) != 0);
      mpfr_exp_rational (tmp, uk, shift + twopoweri - ttt, k + 1, P, mult);
      for (loop = 0; loop < shift; loop++)
        mpfr_sqr (tmp, tmp, MPFR_RNDD);
      twopoweri *= 2;

      /* General case */
      iter = (k <= prec_x) ? k : prec_x;
      for (i = 1; i <= iter; i++)
        {
          mpfr_extract (uk, x_copy, i);
          if (MPFR_LIKELY (mpz_cmp_ui (uk, 0) != 0))
            {
              mpfr_exp_rational (t, uk, twopoweri - ttt, k  - i + 1, P, mult);
              mpfr_mul (tmp, tmp, t, MPFR_RNDD);
            }
          MPFR_ASSERTN (twopoweri <= LONG_MAX/2);
          twopoweri *=2;
        }

      /* Clear tables */
      for (i = 0; i < 3*(k+2); i++)
        mpz_clear (P[i]);
      (*__gmp_free_func) (P, 3*(k+2)*sizeof(mpz_t));
      (*__gmp_free_func) (mult, 2*(k+2)*sizeof(mpfr_prec_t));

      if (shift_x > 0)
        {
          MPFR_BLOCK (flags, {
              for (loop = 0; loop < shift_x - 1; loop++)
                mpfr_sqr (tmp, tmp, MPFR_RNDD);
              mpfr_sqr (t, tmp, MPFR_RNDD);
            } );

          if (MPFR_UNLIKELY (MPFR_OVERFLOW (flags)))
            {
              /* tmp <= exact result, so that it is a real overflow. */
              inexact = mpfr_overflow (y, rnd_mode, 1);
              MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, MPFR_FLAGS_OVERFLOW);
              break;
            }

          if (MPFR_UNLIKELY (MPFR_UNDERFLOW (flags)))
            {
              /* This may be a spurious underflow. So, let's scale
                 the result. */
              mpfr_mul_2ui (tmp, tmp, 1, MPFR_RNDD);  /* no overflow, exact */
              mpfr_sqr (t, tmp, MPFR_RNDD);
              if (MPFR_IS_ZERO (t))
                {
                  /* approximate result < 2^(emin - 3), thus
                     exact result < 2^(emin - 2). */
                  inexact = mpfr_underflow (y, (rnd_mode == MPFR_RNDN) ?
                                            MPFR_RNDZ : rnd_mode, 1);
                  MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, MPFR_FLAGS_UNDERFLOW);
                  break;
                }
              scaled = 1;
            }
        }

      if (mpfr_can_round (shift_x > 0 ? t : tmp, realprec, MPFR_RNDN, MPFR_RNDZ,
                          MPFR_PREC(y) + (rnd_mode == MPFR_RNDN)))
        {
          inexact = mpfr_set (y, shift_x > 0 ? t : tmp, rnd_mode);
          if (MPFR_UNLIKELY (scaled && MPFR_IS_PURE_FP (y)))
            {
              int inex2;
              mpfr_exp_t ey;

              /* The result has been scaled and needs to be corrected. */
              ey = MPFR_GET_EXP (y);
              inex2 = mpfr_mul_2si (y, y, -2, rnd_mode);
              if (inex2)  /* underflow */
                {
                  if (rnd_mode == MPFR_RNDN && inexact < 0 &&
                      MPFR_IS_ZERO (y) && ey == __gmpfr_emin + 1)
                    {
                      /* Double rounding case: in MPFR_RNDN, the scaled
                         result has been rounded downward to 2^emin.
                         As the exact result is > 2^(emin - 2), correct
                         rounding must be done upward. */
                      /* TODO: make sure in coverage tests that this line
                         is reached. */
                      inexact = mpfr_underflow (y, MPFR_RNDU, 1);
                    }
                  else
                    {
                      /* No double rounding. */
                      inexact = inex2;
                    }
                  MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, MPFR_FLAGS_UNDERFLOW);
                }
            }
          break;
        }

      MPFR_ZIV_NEXT (ziv_loop, realprec);
      Prec = realprec + shift + 2 + shift_x;
      mpfr_set_prec (t, Prec);
      mpfr_set_prec (tmp, Prec);
    }
  MPFR_ZIV_FREE (ziv_loop);

  mpz_clear (uk);
  mpfr_clear (tmp);
  mpfr_clear (t);
  mpfr_clear (x_copy);
  MPFR_SAVE_EXPO_FREE (expo);
  return inexact;
}
