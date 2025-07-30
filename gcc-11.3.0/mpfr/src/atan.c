/* mpfr_atan -- arc-tangent of a floating-point number

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

/* If x = p/2^r, put in y an approximation of atan(x)/x using 2^m terms
   for the series expansion, with an error of at most 1 ulp.
   Assumes |x| < 1.

   If X=x^2, we want 1 - X/3 + X^2/5 - ... + (-1)^k*X^k/(2k+1) + ...

   Assume p is non-zero.

   When we sum terms up to x^k/(2k+1), the denominator Q[0] is
   3*5*7*...*(2k+1) ~ (2k/e)^k.
*/
static void
mpfr_atan_aux (mpfr_ptr y, mpz_ptr p, long r, int m, mpz_t *tab)
{
  mpz_t *S, *Q, *ptoj;
  unsigned long n, i, k, j, l;
  mpfr_exp_t diff, expo;
  int im, done;
  mpfr_prec_t mult, *accu, *log2_nb_terms;
  mpfr_prec_t precy = MPFR_PREC(y);

  MPFR_ASSERTD(mpz_cmp_ui (p, 0) != 0);

  accu = (mpfr_prec_t*) (*__gmp_allocate_func) ((2 * m + 2) * sizeof (mpfr_prec_t));
  log2_nb_terms = accu + m + 1;

  /* Set Tables */
  S    = tab;           /* S */
  ptoj = S + 1*(m+1);   /* p^2^j Precomputed table */
  Q    = S + 2*(m+1);   /* Product of Odd integer  table  */

  /* From p to p^2, and r to 2r */
  mpz_mul (p, p, p);
  MPFR_ASSERTD (2 * r > r);
  r = 2 * r;

  /* Normalize p */
  n = mpz_scan1 (p, 0);
  mpz_tdiv_q_2exp (p, p, n); /* exact */
  MPFR_ASSERTD (r > n);
  r -= n;
  /* since |p/2^r| < 1, and p is a non-zero integer, necessarily r > 0 */

  MPFR_ASSERTD (mpz_sgn (p) > 0);
  MPFR_ASSERTD (m > 0);

  /* check if p=1 (special case) */
  l = 0;
  /*
    We compute by binary splitting, with X = x^2 = p/2^r:
    P(a,b) = p if a+1=b, P(a,c)*P(c,b) otherwise
    Q(a,b) = (2a+1)*2^r if a+1=b [except Q(0,1)=1], Q(a,c)*Q(c,b) otherwise
    S(a,b) = p*(2a+1) if a+1=b, Q(c,b)*S(a,c)+Q(a,c)*P(a,c)*S(c,b) otherwise
    Then atan(x)/x ~ S(0,i)/Q(0,i) for i so that (p/2^r)^i/i is small enough.
    The factor 2^(r*(b-a)) in Q(a,b) is implicit, thus we have to take it
    into account when we compute with Q.
  */
  accu[0] = 0; /* accu[k] = Mult[0] + ... + Mult[k], where Mult[j] is the
                  number of bits of the corresponding term S[j]/Q[j] */
  if (mpz_cmp_ui (p, 1) != 0)
    {
      /* p <> 1: precompute ptoj table */
      mpz_set (ptoj[0], p);
      for (im = 1 ; im <= m ; im ++)
        mpz_mul (ptoj[im], ptoj[im - 1], ptoj[im - 1]);
      /* main loop */
      n = 1UL << m;
      /* the ith term being X^i/(2i+1) with X=p/2^r, we can stop when
         p^i/2^(r*i) < 2^(-precy), i.e. r*i > precy + log2(p^i) */
      for (i = k = done = 0; (i < n) && (done == 0); i += 2, k ++)
        {
          /* initialize both S[k],Q[k] and S[k+1],Q[k+1] */
          mpz_set_ui (Q[k+1], 2 * i + 3); /* Q(i+1,i+2) */
          mpz_mul_ui (S[k+1], p, 2 * i + 1); /* S(i+1,i+2) */
          mpz_mul_2exp (S[k], Q[k+1], r);
          mpz_sub (S[k], S[k], S[k+1]); /* S(i,i+2) */
          mpz_mul_ui (Q[k], Q[k+1], 2 * i + 1); /* Q(i,i+2) */
          log2_nb_terms[k] = 1; /* S[k]/Q[k] corresponds to 2 terms */
          for (j = (i + 2) >> 1, l = 1; (j & 1) == 0; l ++, j >>= 1, k --)
            {
              /* invariant: S[k-1]/Q[k-1] and S[k]/Q[k] correspond
                 to 2^l terms each. We combine them into S[k-1]/Q[k-1] */
              MPFR_ASSERTD (k > 0);
              mpz_mul (S[k], S[k], Q[k-1]);
              mpz_mul (S[k], S[k], ptoj[l]);
              mpz_mul (S[k-1], S[k-1], Q[k]);
              mpz_mul_2exp (S[k-1], S[k-1], r << l);
              mpz_add (S[k-1], S[k-1], S[k]);
              mpz_mul (Q[k-1], Q[k-1], Q[k]);
              log2_nb_terms[k-1] = l + 1;
              /* now S[k-1]/Q[k-1] corresponds to 2^(l+1) terms */
              MPFR_MPZ_SIZEINBASE2(mult, ptoj[l+1]);
              /* FIXME: precompute bits(ptoj[l+1]) outside the loop? */
              mult = (r << (l + 1)) - mult - 1;
              accu[k-1] = (k == 1) ? mult : accu[k-2] + mult;
              if (accu[k-1] > precy)
                done = 1;
            }
        }
    }
  else /* special case p=1: the ith term being X^i/(2i+1) with X=1/2^r,
          we can stop when r*i > precy i.e. i > precy/r */
    {
      n = 1UL << m;
      for (i = k = 0; (i < n) && (i <= precy / r); i += 2, k ++)
        {
          mpz_set_ui (Q[k + 1], 2 * i + 3);
          mpz_mul_2exp (S[k], Q[k+1], r);
          mpz_sub_ui (S[k], S[k], 1 + 2 * i);
          mpz_mul_ui (Q[k], Q[k + 1], 1 + 2 * i);
          log2_nb_terms[k] = 1; /* S[k]/Q[k] corresponds to 2 terms */
          for (j = (i + 2) >> 1, l = 1; (j & 1) == 0; l++, j >>= 1, k --)
            {
              MPFR_ASSERTD (k > 0);
              mpz_mul (S[k], S[k], Q[k-1]);
              mpz_mul (S[k-1], S[k-1], Q[k]);
              mpz_mul_2exp (S[k-1], S[k-1], r << l);
              mpz_add (S[k-1], S[k-1], S[k]);
              mpz_mul (Q[k-1], Q[k-1], Q[k]);
              log2_nb_terms[k-1] = l + 1;
            }
        }
    }

  /* we need to combine S[0]/Q[0]...S[k-1]/Q[k-1] */
  l = 0; /* number of terms accumulated in S[k]/Q[k] */
  while (k > 1)
    {
      k --;
      /* combine S[k-1]/Q[k-1] and S[k]/Q[k] */
      j = log2_nb_terms[k-1];
      mpz_mul (S[k], S[k], Q[k-1]);
      if (mpz_cmp_ui (p, 1) != 0)
        mpz_mul (S[k], S[k], ptoj[j]);
      mpz_mul (S[k-1], S[k-1], Q[k]);
      l += 1 << log2_nb_terms[k];
      mpz_mul_2exp (S[k-1], S[k-1], r * l);
      mpz_add (S[k-1], S[k-1], S[k]);
      mpz_mul (Q[k-1], Q[k-1], Q[k]);
    }
  (*__gmp_free_func) (accu, (2 * m + 2) * sizeof (mpfr_prec_t));

  MPFR_MPZ_SIZEINBASE2 (diff, S[0]);
  diff -= 2 * precy;
  expo = diff;
  if (diff >= 0)
    mpz_tdiv_q_2exp (S[0], S[0], diff);
  else
    mpz_mul_2exp (S[0], S[0], -diff);

  MPFR_MPZ_SIZEINBASE2 (diff, Q[0]);
  diff -= precy;
  expo -= diff;
  if (diff >= 0)
    mpz_tdiv_q_2exp (Q[0], Q[0], diff);
  else
    mpz_mul_2exp (Q[0], Q[0], -diff);

  mpz_tdiv_q (S[0], S[0], Q[0]);
  mpfr_set_z (y, S[0], MPFR_RNDD);
  MPFR_SET_EXP (y, MPFR_EXP(y) + expo - r * (i - 1));
}

int
mpfr_atan (mpfr_ptr atan, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  mpfr_t xp, arctgt, sk, tmp, tmp2;
  mpz_t  ukz;
  mpz_t *tabz;
  mpfr_exp_t exptol;
  mpfr_prec_t prec, realprec, est_lost, lost;
  unsigned long twopoweri, log2p, red;
  int comparaison, inexact;
  int i, n0, oldn0;
  MPFR_GROUP_DECL (group);
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_ZIV_DECL (loop);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg rnd=%d", mpfr_get_prec (x), mpfr_log_prec, x, rnd_mode),
     ("atan[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (atan), mpfr_log_prec, atan, inexact));

  /* Singular cases */
  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (atan);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (x))
        {
          MPFR_SAVE_EXPO_MARK (expo);
          if (MPFR_IS_POS (x))  /* arctan(+inf) = Pi/2 */
            inexact = mpfr_const_pi (atan, rnd_mode);
          else /* arctan(-inf) = -Pi/2 */
            {
              inexact = -mpfr_const_pi (atan,
                                        MPFR_INVERT_RND (rnd_mode));
              MPFR_CHANGE_SIGN (atan);
            }
          mpfr_div_2ui (atan, atan, 1, rnd_mode);  /* exact (no exceptions) */
          MPFR_SAVE_EXPO_FREE (expo);
          return mpfr_check_range (atan, inexact, rnd_mode);
        }
      else /* x is necessarily 0 */
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (x));
          MPFR_SET_ZERO (atan);
          MPFR_SET_SAME_SIGN (atan, x);
          MPFR_RET (0);
        }
    }

  /* atan(x) = x - x^3/3 + x^5/5...
     so the error is < 2^(3*EXP(x)-1)
     so `EXP(x)-(3*EXP(x)-1)` = -2*EXP(x)+1 */
  MPFR_FAST_COMPUTE_IF_SMALL_INPUT (atan, x, -2 * MPFR_GET_EXP (x), 1, 0,
                                    rnd_mode, {});

  /* Set x_p=|x| */
  MPFR_TMP_INIT_ABS (xp, x);

  MPFR_SAVE_EXPO_MARK (expo);

  /* Other simple case arctan(-+1)=-+pi/4 */
  comparaison = mpfr_cmp_ui (xp, 1);
  if (MPFR_UNLIKELY (comparaison == 0))
    {
      int neg = MPFR_IS_NEG (x);
      inexact = mpfr_const_pi (atan, MPFR_IS_POS (x) ? rnd_mode
                               : MPFR_INVERT_RND (rnd_mode));
      if (neg)
        {
          inexact = -inexact;
          MPFR_CHANGE_SIGN (atan);
        }
      mpfr_div_2ui (atan, atan, 2, rnd_mode);  /* exact (no exceptions) */
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_check_range (atan, inexact, rnd_mode);
    }

  realprec = MPFR_PREC (atan) + MPFR_INT_CEIL_LOG2 (MPFR_PREC (atan)) + 4;
  prec = realprec + GMP_NUMB_BITS;

  /* Initialisation */
  mpz_init (ukz);
  MPFR_GROUP_INIT_4 (group, prec, sk, tmp, tmp2, arctgt);
  oldn0 = 0;
  tabz = (mpz_t *) 0;

  MPFR_ZIV_INIT (loop, prec);
  for (;;)
    {
      /* First, if |x| < 1, we need to have more prec to be able to round (sup)
         n0 = ceil(log(prec_requested + 2 + 1+ln(2.4)/ln(2))/log(2)) */
      mpfr_prec_t sup;
      sup = MPFR_GET_EXP (xp) < 0 ? 2 - MPFR_GET_EXP (xp) : 1; /* sup >= 1 */

      n0 = MPFR_INT_CEIL_LOG2 ((realprec + sup) + 3);
      /* since realprec >= 4, n0 >= ceil(log2(8)) >= 3, thus 3*n0 > 2 */
      prec = (realprec + sup) + 1 + MPFR_INT_CEIL_LOG2 (3*n0-2);

      /* the number of lost bits due to argument reduction is
         9 - 2 * EXP(sk), which we estimate by 9 + 2*ceil(log2(p))
         since we manage that sk < 1/p */
      if (MPFR_PREC (atan) > 100)
        {
          log2p = MPFR_INT_CEIL_LOG2(prec) / 2 - 3;
          est_lost = 9 + 2 * log2p;
          prec += est_lost;
        }
      else
        log2p = est_lost = 0; /* don't reduce the argument */

      /* Initialisation */
      MPFR_GROUP_REPREC_4 (group, prec, sk, tmp, tmp2, arctgt);
      if (MPFR_LIKELY (oldn0 == 0))
        {
          oldn0 = 3 * (n0 + 1);
          tabz = (mpz_t *) (*__gmp_allocate_func) (oldn0 * sizeof (mpz_t));
          for (i = 0; i < oldn0; i++)
            mpz_init (tabz[i]);
        }
      else if (MPFR_UNLIKELY (oldn0 < 3 * (n0 + 1)))
        {
          tabz = (mpz_t *) (*__gmp_reallocate_func)
            (tabz, oldn0 * sizeof (mpz_t), 3 * (n0 + 1)*sizeof (mpz_t));
          for (i = oldn0; i < 3 * (n0 + 1); i++)
            mpz_init (tabz[i]);
          oldn0 = 3 * (n0 + 1);
        }

      /* The mpfr_ui_div below mustn't underflow. This is guaranteed by
         MPFR_SAVE_EXPO_MARK, but let's check that for maintainability. */
      MPFR_ASSERTD (__gmpfr_emax <= 1 - __gmpfr_emin);

      if (comparaison > 0) /* use atan(xp) = Pi/2 - atan(1/xp) */
        mpfr_ui_div (sk, 1, xp, MPFR_RNDN);
      else
        mpfr_set (sk, xp, MPFR_RNDN);

      /* now 0 < sk <= 1 */

      /* Argument reduction: atan(x) = 2 atan((sqrt(1+x^2)-1)/x).
         We want |sk| < k/sqrt(p) where p is the target precision. */
      lost = 0;
      for (red = 0; MPFR_GET_EXP(sk) > - (mpfr_exp_t) log2p; red ++)
        {
          lost = 9 - 2 * MPFR_EXP(sk);
          mpfr_mul (tmp, sk, sk, MPFR_RNDN);
          mpfr_add_ui (tmp, tmp, 1, MPFR_RNDN);
          mpfr_sqrt (tmp, tmp, MPFR_RNDN);
          mpfr_sub_ui (tmp, tmp, 1, MPFR_RNDN);
          if (red == 0 && comparaison > 0)
            /* use xp = 1/sk */
            mpfr_mul (sk, tmp, xp, MPFR_RNDN);
          else
            mpfr_div (sk, tmp, sk, MPFR_RNDN);
        }

      /* we started from x0 = 1/|x| if |x| > 1, and |x| otherwise, thus
         we had x0 = min(|x|, 1/|x|) <= 1, and applied 'red' times the
         argument reduction x -> (sqrt(1+x^2)-1)/x, which keeps 0 < x < 1,
         thus 0 < sk <= 1, and sk=1 can occur only if red=0 */

      /* If sk=1, then if |x| < 1, we have 1 - 2^(-prec-1) <= |x| < 1,
         or if |x| > 1, we have 1 - 2^(-prec-1) <= 1/|x| < 1, thus in all
         cases ||x| - 1| <= 2^(-prec), from which it follows
         |atan|x| - Pi/4| <= 2^(-prec), given the Taylor expansion
         atan(1+x) = Pi/4 + x/2 - x^2/4 + ...
         Since Pi/4 = 0.785..., the error is at most one ulp.
      */
      if (MPFR_UNLIKELY(mpfr_cmp_ui (sk, 1) == 0))
        {
          mpfr_const_pi (arctgt, MPFR_RNDN); /* 1/2 ulp extra error */
          mpfr_div_2ui (arctgt, arctgt, 2, MPFR_RNDN); /* exact */
          realprec = prec - 2;
          goto can_round;
        }

      /* Assignation  */
      MPFR_SET_ZERO (arctgt);
      twopoweri = 1 << 0;
      MPFR_ASSERTD (n0 >= 4);
      for (i = 0 ; i < n0; i++)
        {
          if (MPFR_UNLIKELY (MPFR_IS_ZERO (sk)))
            break;
          /* Calculation of trunc(tmp) --> mpz */
          mpfr_mul_2ui (tmp, sk, twopoweri, MPFR_RNDN);
          mpfr_trunc (tmp, tmp);
          if (!MPFR_IS_ZERO (tmp))
            {
              /* tmp = ukz*2^exptol */
              exptol = mpfr_get_z_2exp (ukz, tmp);
              /* since the s_k are decreasing (see algorithms.tex),
                 and s_0 = min(|x|, 1/|x|) < 1, we have sk < 1,
                 thus exptol < 0 */
              MPFR_ASSERTD (exptol < 0);
              mpz_tdiv_q_2exp (ukz, ukz, (unsigned long int) (-exptol));
              /* since tmp is a non-zero integer, and tmp = ukzold*2^exptol,
                 we now have ukz = tmp, thus ukz is non-zero */
              /* Calculation of arctan(Ak) */
              mpfr_set_z (tmp, ukz, MPFR_RNDN);
              mpfr_div_2ui (tmp, tmp, twopoweri, MPFR_RNDN);
              mpfr_atan_aux (tmp2, ukz, twopoweri, n0 - i, tabz);
              mpfr_mul (tmp2, tmp2, tmp, MPFR_RNDN);
              /* Addition */
              mpfr_add (arctgt, arctgt, tmp2, MPFR_RNDN);
              /* Next iteration */
              mpfr_sub (tmp2, sk, tmp, MPFR_RNDN);
              mpfr_mul (sk, sk, tmp, MPFR_RNDN);
              mpfr_add_ui (sk, sk, 1, MPFR_RNDN);
              mpfr_div (sk, tmp2, sk, MPFR_RNDN);
            }
          twopoweri <<= 1;
        }
      /* Add last step (Arctan(sk) ~= sk */
      mpfr_add (arctgt, arctgt, sk, MPFR_RNDN);

      /* argument reduction */
      mpfr_mul_2exp (arctgt, arctgt, red, MPFR_RNDN);

      if (comparaison > 0)
        { /* atan(x) = Pi/2-atan(1/x) for x > 0 */
          mpfr_const_pi (tmp, MPFR_RNDN);
          mpfr_div_2ui (tmp, tmp, 1, MPFR_RNDN);
          mpfr_sub (arctgt, tmp, arctgt, MPFR_RNDN);
        }
      MPFR_SET_POS (arctgt);

    can_round:
      if (MPFR_LIKELY (MPFR_CAN_ROUND (arctgt, realprec + est_lost - lost,
                                       MPFR_PREC (atan), rnd_mode)))
        break;
      MPFR_ZIV_NEXT (loop, realprec);
    }
  MPFR_ZIV_FREE (loop);

  inexact = mpfr_set4 (atan, arctgt, rnd_mode, MPFR_SIGN (x));

  for (i = 0 ; i < oldn0 ; i++)
    mpz_clear (tabz[i]);
  mpz_clear (ukz);
  (*__gmp_free_func) (tabz, oldn0 * sizeof (mpz_t));
  MPFR_GROUP_CLEAR (group);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (atan, inexact, rnd_mode);
}
