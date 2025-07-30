/* mpfr_jn_asympt, mpfr_yn_asympt -- shared code for mpfr_jn and mpfr_yn

Copyright 2007-2017 Free Software Foundation, Inc.
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

#ifdef MPFR_JN
# define FUNCTION mpfr_jn_asympt
#else
# ifdef MPFR_YN
#  define FUNCTION mpfr_yn_asympt
# else
#  error "neither MPFR_JN nor MPFR_YN is defined"
# endif
#endif

/* Implements asymptotic expansion for jn or yn (formulae 9.2.5 and 9.2.6
   from Abramowitz & Stegun).
   Assumes |z| > p log(2)/2, where p is the target precision
   (z can be negative only for jn).
   Return 0 if the expansion does not converge enough (the value 0 as inexact
   flag should not happen for normal input).
*/
static int
FUNCTION (mpfr_ptr res, long n, mpfr_srcptr z, mpfr_rnd_t r)
{
  mpfr_t s, c, P, Q, t, iz, err_t, err_s, err_u;
  mpfr_prec_t w;
  long k;
  int inex, stop, diverge = 0;
  mpfr_exp_t err2, err;
  MPFR_ZIV_DECL (loop);

  mpfr_init (c);

  w = MPFR_PREC(res) + MPFR_INT_CEIL_LOG2(MPFR_PREC(res)) + 4;

  MPFR_ZIV_INIT (loop, w);
  for (;;)
    {
      mpfr_set_prec (c, w);
      mpfr_init2 (s, w);
      mpfr_init2 (P, w);
      mpfr_init2 (Q, w);
      mpfr_init2 (t, w);
      mpfr_init2 (iz, w);
      mpfr_init2 (err_t, 31);
      mpfr_init2 (err_s, 31);
      mpfr_init2 (err_u, 31);

      /* Approximate sin(z) and cos(z). In the following, err <= k means that
         the approximate value y and the true value x are related by
         y = x * (1 + u)^k with |u| <= 2^(-w), following Higham's method. */
      mpfr_sin_cos (s, c, z, MPFR_RNDN);
      if (MPFR_IS_NEG(z))
        mpfr_neg (s, s, MPFR_RNDN); /* compute jn/yn(|z|), fix sign later */
      /* The absolute error on s/c is bounded by 1/2 ulp(1/2) <= 2^(-w-1). */
      mpfr_add (t, s, c, MPFR_RNDN);
      mpfr_sub (c, s, c, MPFR_RNDN);
      mpfr_swap (s, t);
      /* now s approximates sin(z)+cos(z), and c approximates sin(z)-cos(z),
         with total absolute error bounded by 2^(1-w). */

      /* precompute 1/(8|z|) */
      mpfr_si_div (iz, MPFR_IS_POS(z) ? 1 : -1, z, MPFR_RNDN);   /* err <= 1 */
      mpfr_div_2ui (iz, iz, 3, MPFR_RNDN);

      /* compute P and Q */
      mpfr_set_ui (P, 1, MPFR_RNDN);
      mpfr_set_ui (Q, 0, MPFR_RNDN);
      mpfr_set_ui (t, 1, MPFR_RNDN); /* current term */
      mpfr_set_ui (err_t, 0, MPFR_RNDN); /* error on t */
      mpfr_set_ui (err_s, 0, MPFR_RNDN); /* error on P and Q (sum of errors) */
      for (k = 1, stop = 0; stop < 4; k++)
        {
          /* compute next term: t(k)/t(k-1) = (2n+2k-1)(2n-2k+1)/(8kz) */
          mpfr_mul_si (t, t, 2 * (n + k) - 1, MPFR_RNDN); /* err <= err_k + 1 */
          mpfr_mul_si (t, t, 2 * (n - k) + 1, MPFR_RNDN); /* err <= err_k + 2 */
          mpfr_div_ui (t, t, k, MPFR_RNDN);               /* err <= err_k + 3 */
          mpfr_mul (t, t, iz, MPFR_RNDN);                 /* err <= err_k + 5 */
          /* the relative error on t is bounded by (1+u)^(5k)-1, which is
             bounded by 6ku for 6ku <= 0.02: first |5 log(1+u)| <= |5.5u|
             for |u| <= 0.15, then |exp(5.5u)-1| <= 6u for |u| <= 0.02. */
          mpfr_mul_ui (err_t, t, 6 * k, MPFR_IS_POS(t) ? MPFR_RNDU : MPFR_RNDD);
          mpfr_abs (err_t, err_t, MPFR_RNDN); /* exact */
          /* the absolute error on t is bounded by err_t * 2^(-w) */
          mpfr_abs (err_u, t, MPFR_RNDU);
          mpfr_mul_2ui (err_u, err_u, w, MPFR_RNDU); /* t * 2^w */
          mpfr_add (err_u, err_u, err_t, MPFR_RNDU); /* max|t| * 2^w */
          if (stop >= 2)
            {
              /* take into account the neglected terms: t * 2^w */
              mpfr_div_2ui (err_s, err_s, w, MPFR_RNDU);
              if (MPFR_IS_POS(t))
                mpfr_add (err_s, err_s, t, MPFR_RNDU);
              else
                mpfr_sub (err_s, err_s, t, MPFR_RNDU);
              mpfr_mul_2ui (err_s, err_s, w, MPFR_RNDU);
              stop ++;
            }
          /* if k is odd, add to Q, otherwise to P */
          else if (k & 1)
            {
              /* if k = 1 mod 4, add, otherwise subtract */
              if ((k & 2) == 0)
                mpfr_add (Q, Q, t, MPFR_RNDN);
              else
                mpfr_sub (Q, Q, t, MPFR_RNDN);
              /* check if the next term is smaller than ulp(Q): if EXP(err_u)
                 <= EXP(Q), since the current term is bounded by
                 err_u * 2^(-w), it is bounded by ulp(Q) */
              if (MPFR_EXP(err_u) <= MPFR_EXP(Q))
                stop ++;
              else
                stop = 0;
            }
          else
            {
              /* if k = 0 mod 4, add, otherwise subtract */
              if ((k & 2) == 0)
                mpfr_add (P, P, t, MPFR_RNDN);
              else
                mpfr_sub (P, P, t, MPFR_RNDN);
              /* check if the next term is smaller than ulp(P) */
              if (MPFR_EXP(err_u) <= MPFR_EXP(P))
                stop ++;
              else
                stop = 0;
            }
          mpfr_add (err_s, err_s, err_t, MPFR_RNDU);
          /* the sum of the rounding errors on P and Q is bounded by
             err_s * 2^(-w) */

          /* stop when start to diverge */
          if (stop < 2 &&
              ((MPFR_IS_POS(z) && mpfr_cmp_ui (z, (k + 1) / 2) < 0) ||
               (MPFR_IS_NEG(z) && mpfr_cmp_si (z, - ((k + 1) / 2)) > 0)))
            {
              /* if we have to stop the series because it diverges, then
                 increasing the precision will most probably fail, since
                 we will stop to the same point, and thus compute a very
                 similar approximation */
              diverge = 1;
              stop = 2; /* force stop */
            }
        }
      /* the sum of the total errors on P and Q is bounded by err_s * 2^(-w) */

      /* Now combine: the sum of the rounding errors on P and Q is bounded by
         err_s * 2^(-w), and the absolute error on s/c is bounded by 2^(1-w) */
      if ((n & 1) == 0) /* n even: P * (sin + cos) + Q (cos - sin) for jn
                                   Q * (sin + cos) + P (sin - cos) for yn */
        {
#ifdef MPFR_JN
          mpfr_mul (c, c, Q, MPFR_RNDN); /* Q * (sin - cos) */
          mpfr_mul (s, s, P, MPFR_RNDN); /* P * (sin + cos) */
#else
          mpfr_mul (c, c, P, MPFR_RNDN); /* P * (sin - cos) */
          mpfr_mul (s, s, Q, MPFR_RNDN); /* Q * (sin + cos) */
#endif
          err = MPFR_EXP(c);
          if (MPFR_EXP(s) > err)
            err = MPFR_EXP(s);
#ifdef MPFR_JN
          mpfr_sub (s, s, c, MPFR_RNDN);
#else
          mpfr_add (s, s, c, MPFR_RNDN);
#endif
        }
      else /* n odd: P * (sin - cos) + Q (cos + sin) for jn,
                     Q * (sin - cos) - P (cos + sin) for yn */
        {
#ifdef MPFR_JN
          mpfr_mul (c, c, P, MPFR_RNDN); /* P * (sin - cos) */
          mpfr_mul (s, s, Q, MPFR_RNDN); /* Q * (sin + cos) */
#else
          mpfr_mul (c, c, Q, MPFR_RNDN); /* Q * (sin - cos) */
          mpfr_mul (s, s, P, MPFR_RNDN); /* P * (sin + cos) */
#endif
          err = MPFR_EXP(c);
          if (MPFR_EXP(s) > err)
            err = MPFR_EXP(s);
#ifdef MPFR_JN
          mpfr_add (s, s, c, MPFR_RNDN);
#else
          mpfr_sub (s, c, s, MPFR_RNDN);
#endif
        }
      if ((n & 2) != 0)
        mpfr_neg (s, s, MPFR_RNDN);
      if (MPFR_EXP(s) > err)
        err = MPFR_EXP(s);
      /* the absolute error on s is bounded by P*err(s/c) + Q*err(s/c)
         + err(P)*(s/c) + err(Q)*(s/c) + 3 * 2^(err - w - 1)
         <= (|P|+|Q|) * 2^(1-w) + err_s * 2^(1-w) + 2^err * 2^(1-w),
         since |c|, |old_s| <= 2. */
      err2 = (MPFR_EXP(P) >= MPFR_EXP(Q)) ? MPFR_EXP(P) + 2 : MPFR_EXP(Q) + 2;
      /* (|P| + |Q|) * 2^(1 - w) <= 2^(err2 - w) */
      err = MPFR_EXP(err_s) >= err ? MPFR_EXP(err_s) + 2 : err + 2;
      /* err_s * 2^(1-w) + 2^old_err * 2^(1-w) <= 2^err * 2^(-w) */
      err2 = (err >= err2) ? err + 1 : err2 + 1;
      /* now the absolute error on s is bounded by 2^(err2 - w) */

      /* multiply by sqrt(1/(Pi*z)) */
      mpfr_const_pi (c, MPFR_RNDN);     /* Pi, err <= 1 */
      mpfr_mul (c, c, z, MPFR_RNDN);    /* err <= 2 */
      mpfr_si_div (c, MPFR_IS_POS(z) ? 1 : -1, c, MPFR_RNDN); /* err <= 3 */
      mpfr_sqrt (c, c, MPFR_RNDN);      /* err<=5/2, thus the absolute error is
                                          bounded by 3*u*|c| for |u| <= 0.25 */
      mpfr_mul (err_t, c, s, MPFR_SIGN(c)==MPFR_SIGN(s) ? MPFR_RNDU : MPFR_RNDD);
      mpfr_abs (err_t, err_t, MPFR_RNDU);
      mpfr_mul_ui (err_t, err_t, 3, MPFR_RNDU);
      /* 3*2^(-w)*|old_c|*|s| [see below] is bounded by err_t * 2^(-w) */
      err2 += MPFR_EXP(c);
      /* |old_c| * 2^(err2 - w) [see below] is bounded by 2^(err2-w) */
      mpfr_mul (c, c, s, MPFR_RNDN);    /* the absolute error on c is bounded by
                                          1/2 ulp(c) + 3*2^(-w)*|old_c|*|s|
                                          + |old_c| * 2^(err2 - w) */
      /* compute err_t * 2^(-w) + 1/2 ulp(c) = (err_t + 2^EXP(c)) * 2^(-w) */
      err = (MPFR_EXP(err_t) > MPFR_EXP(c)) ? MPFR_EXP(err_t) + 1 : MPFR_EXP(c) + 1;
      /* err_t * 2^(-w) + 1/2 ulp(c) <= 2^(err - w) */
      /* now err_t * 2^(-w) bounds 1/2 ulp(c) + 3*2^(-w)*|old_c|*|s| */
      err = (err >= err2) ? err + 1 : err2 + 1;
      /* the absolute error on c is bounded by 2^(err - w) */

      mpfr_clear (s);
      mpfr_clear (P);
      mpfr_clear (Q);
      mpfr_clear (t);
      mpfr_clear (iz);
      mpfr_clear (err_t);
      mpfr_clear (err_s);
      mpfr_clear (err_u);

      err -= MPFR_EXP(c);
      if (MPFR_LIKELY (MPFR_CAN_ROUND (c, w - err, MPFR_PREC(res), r)))
        break;
      if (diverge != 0)
        {
          MPFR_ZIV_FREE (loop);
          mpfr_clear (c);
          return 0; /* means that the asymptotic expansion failed */
        }
      MPFR_ZIV_NEXT (loop, w);
    }
  MPFR_ZIV_FREE (loop);

  inex = (MPFR_IS_POS(z) || ((n & 1) == 0)) ? mpfr_set (res, c, r)
    : mpfr_neg (res, c, r);
  mpfr_clear (c);

  return inex;
}
