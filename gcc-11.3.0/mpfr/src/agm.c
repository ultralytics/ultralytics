/* mpfr_agm -- arithmetic-geometric mean of two floating-point numbers

Copyright 1999-2017 Free Software Foundation, Inc.
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

/* agm(x,y) is between x and y, so we don't need to save exponent range */
int
mpfr_agm (mpfr_ptr r, mpfr_srcptr op2, mpfr_srcptr op1, mpfr_rnd_t rnd_mode)
{
  int compare, inexact;
  mp_size_t s;
  mpfr_prec_t p, q;
  mp_limb_t *up, *vp, *ufp, *vfp;
  mpfr_t u, v, uf, vf, sc1, sc2;
  mpfr_exp_t scaleop = 0, scaleit;
  unsigned long n; /* number of iterations */
  MPFR_ZIV_DECL (loop);
  MPFR_TMP_DECL(marker);
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_LOG_FUNC
    (("op2[%Pu]=%.*Rg op1[%Pu]=%.*Rg rnd=%d",
      mpfr_get_prec (op2), mpfr_log_prec, op2,
      mpfr_get_prec (op1), mpfr_log_prec, op1, rnd_mode),
     ("r[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (r), mpfr_log_prec, r, inexact));

  /* Deal with special values */
  if (MPFR_ARE_SINGULAR (op1, op2))
    {
      /* If a or b is NaN, the result is NaN */
      if (MPFR_IS_NAN(op1) || MPFR_IS_NAN(op2))
        {
          MPFR_SET_NAN(r);
          MPFR_RET_NAN;
        }
      /* now one of a or b is Inf or 0 */
      /* If a and b is +Inf, the result is +Inf.
         Otherwise if a or b is -Inf or 0, the result is NaN */
      else if (MPFR_IS_INF(op1) || MPFR_IS_INF(op2))
        {
          if (MPFR_IS_STRICTPOS(op1) && MPFR_IS_STRICTPOS(op2))
            {
              MPFR_SET_INF(r);
              MPFR_SET_SAME_SIGN(r, op1);
              MPFR_RET(0); /* exact */
            }
          else
            {
              MPFR_SET_NAN(r);
              MPFR_RET_NAN;
            }
        }
      else /* a and b are neither NaN nor Inf, and one is zero */
        {  /* If a or b is 0, the result is +0 since a sqrt is positive */
          MPFR_ASSERTD (MPFR_IS_ZERO (op1) || MPFR_IS_ZERO (op2));
          MPFR_SET_POS (r);
          MPFR_SET_ZERO (r);
          MPFR_RET (0); /* exact */
        }
    }

  /* If a or b is negative (excluding -Infinity), the result is NaN */
  if (MPFR_UNLIKELY(MPFR_IS_NEG(op1) || MPFR_IS_NEG(op2)))
    {
      MPFR_SET_NAN(r);
      MPFR_RET_NAN;
    }

  /* Precision of the following calculus */
  q = MPFR_PREC(r);
  p = q + MPFR_INT_CEIL_LOG2(q) + 15;
  MPFR_ASSERTD (p >= 7); /* see algorithms.tex */
  s = MPFR_PREC2LIMBS (p);

  /* b (op2) and a (op1) are the 2 operands but we want b >= a */
  compare = mpfr_cmp (op1, op2);
  if (MPFR_UNLIKELY( compare == 0 ))
    return mpfr_set (r, op1, rnd_mode);
  else if (compare > 0)
    {
      mpfr_srcptr t = op1;
      op1 = op2;
      op2 = t;
    }

  /* Now b (=op2) > a (=op1) */

  MPFR_SAVE_EXPO_MARK (expo);

  MPFR_TMP_MARK(marker);

  /* Main loop */
  MPFR_ZIV_INIT (loop, p);
  for (;;)
    {
      mpfr_prec_t eq;
      unsigned long err = 0;  /* must be set to 0 at each Ziv iteration */
      MPFR_BLOCK_DECL (flags);

      /* Init temporary vars */
      MPFR_TMP_INIT (up, u, p, s);
      MPFR_TMP_INIT (vp, v, p, s);
      MPFR_TMP_INIT (ufp, uf, p, s);
      MPFR_TMP_INIT (vfp, vf, p, s);

      /* Calculus of un and vn */
    retry:
      MPFR_BLOCK (flags,
                  mpfr_mul (u, op1, op2, MPFR_RNDN);
                  /* mpfr_mul(...): faster since PREC(op) < PREC(u) */
                  mpfr_add (v, op1, op2, MPFR_RNDN);
                  /* mpfr_add with !=prec is still good */);
      if (MPFR_UNLIKELY (MPFR_OVERFLOW (flags) || MPFR_UNDERFLOW (flags)))
        {
          mpfr_exp_t e1 , e2;

          MPFR_ASSERTN (scaleop == 0);
          e1 = MPFR_GET_EXP (op1);
          e2 = MPFR_GET_EXP (op2);

          /* Let's determine scaleop to avoid an overflow/underflow. */
          if (MPFR_OVERFLOW (flags))
            {
              /* Let's recall that emin <= e1 <= e2 <= emax.
                 There has been an overflow. Thus e2 >= emax/2.
                 If the mpfr_mul overflowed, then e1 + e2 > emax.
                 If the mpfr_add overflowed, then e2 = emax.
                 We want: (e1 + scale) + (e2 + scale) <= emax,
                 i.e. scale <= (emax - e1 - e2) / 2. Let's take
                 scale = min(floor((emax - e1 - e2) / 2), -1).
                 This is OK, as:
                 1. emin <= scale <= -1.
                 2. e1 + scale >= emin. Indeed:
                    * If e1 + e2 > emax, then
                      e1 + scale >= e1 + (emax - e1 - e2) / 2 - 1
                                 >= (emax + e1 - emax) / 2 - 1
                                 >= e1 / 2 - 1 >= emin.
                    * Otherwise, mpfr_mul didn't overflow, therefore
                      mpfr_add overflowed and e2 = emax, so that
                      e1 > emin (see restriction below).
                      e1 + scale > emin - 1, thus e1 + scale >= emin.
                 3. e2 + scale <= emax, since scale < 0. */
              if (e1 + e2 > MPFR_EXT_EMAX)
                {
                  scaleop = - (((e1 + e2) - MPFR_EXT_EMAX + 1) / 2);
                  MPFR_ASSERTN (scaleop < 0);
                }
              else
                {
                  /* The addition necessarily overflowed. */
                  MPFR_ASSERTN (e2 == MPFR_EXT_EMAX);
                  /* The case where e1 = emin and e2 = emax is not supported
                     here. This would mean that the precision of e2 would be
                     huge (and possibly not supported in practice anyway). */
                  MPFR_ASSERTN (e1 > MPFR_EXT_EMIN);
                  scaleop = -1;
                }

            }
          else  /* underflow only (in the multiplication) */
            {
              /* We have e1 + e2 <= emin (so, e1 <= e2 <= 0).
                 We want: (e1 + scale) + (e2 + scale) >= emin + 1,
                 i.e. scale >= (emin + 1 - e1 - e2) / 2. let's take
                 scale = ceil((emin + 1 - e1 - e2) / 2). This is OK, as:
                 1. 1 <= scale <= emax.
                 2. e1 + scale >= emin + 1 >= emin.
                 3. e2 + scale <= scale <= emax. */
              MPFR_ASSERTN (e1 <= e2 && e2 <= 0);
              scaleop = (MPFR_EXT_EMIN + 2 - e1 - e2) / 2;
              MPFR_ASSERTN (scaleop > 0);
            }

          MPFR_ALIAS (sc1, op1, MPFR_SIGN (op1), e1 + scaleop);
          MPFR_ALIAS (sc2, op2, MPFR_SIGN (op2), e2 + scaleop);
          op1 = sc1;
          op2 = sc2;
          MPFR_LOG_MSG (("Exception in pre-iteration, scale = %"
                         MPFR_EXP_FSPEC "d\n", scaleop));
          goto retry;
        }

      mpfr_clear_flags ();
      mpfr_sqrt (u, u, MPFR_RNDN);
      mpfr_div_2ui (v, v, 1, MPFR_RNDN);

      scaleit = 0;
      n = 1;
      while (mpfr_cmp2 (u, v, &eq) != 0 && eq <= p - 2)
        {
          MPFR_BLOCK_DECL (flags2);

          MPFR_LOG_MSG (("Iteration n = %lu\n", n));

        retry2:
          mpfr_add (vf, u, v, MPFR_RNDN);  /* No overflow? */
          mpfr_div_2ui (vf, vf, 1, MPFR_RNDN);
          /* See proof in algorithms.tex */
          if (4*eq > p)
            {
              mpfr_t w;
              MPFR_BLOCK_DECL (flags3);

              MPFR_LOG_MSG (("4*eq > p\n", 0));

              /* vf = V(k) */
              mpfr_init2 (w, (p + 1) / 2);
              MPFR_BLOCK
                (flags3,
                 mpfr_sub (w, v, u, MPFR_RNDN);       /* e = V(k-1)-U(k-1) */
                 mpfr_sqr (w, w, MPFR_RNDN);          /* e = e^2 */
                 mpfr_div_2ui (w, w, 4, MPFR_RNDN);   /* e*= (1/2)^2*1/4  */
                 mpfr_div (w, w, vf, MPFR_RNDN);      /* 1/4*e^2/V(k) */
                 );
              if (MPFR_LIKELY (! MPFR_UNDERFLOW (flags3)))
                {
                  mpfr_sub (v, vf, w, MPFR_RNDN);
                  err = MPFR_GET_EXP (vf) - MPFR_GET_EXP (v); /* 0 or 1 */
                  mpfr_clear (w);
                  break;
                }
              /* There has been an underflow because of the cancellation
                 between V(k-1) and U(k-1). Let's use the conventional
                 method. */
              MPFR_LOG_MSG (("4*eq > p -> underflow\n", 0));
              mpfr_clear (w);
              mpfr_clear_underflow ();
            }
          /* U(k) increases, so that U.V can overflow (but not underflow). */
          MPFR_BLOCK (flags2, mpfr_mul (uf, u, v, MPFR_RNDN););
          if (MPFR_UNLIKELY (MPFR_OVERFLOW (flags2)))
            {
              mpfr_exp_t scale2;

              scale2 = - (((MPFR_GET_EXP (u) + MPFR_GET_EXP (v))
                           - MPFR_EXT_EMAX + 1) / 2);
              MPFR_EXP (u) += scale2;
              MPFR_EXP (v) += scale2;
              scaleit += scale2;
              MPFR_LOG_MSG (("Overflow in iteration n = %lu, scaleit = %"
                             MPFR_EXP_FSPEC "d (%" MPFR_EXP_FSPEC "d)\n",
                             n, scaleit, scale2));
              mpfr_clear_overflow ();
              goto retry2;
            }
          mpfr_sqrt (u, uf, MPFR_RNDN);
          mpfr_swap (v, vf);
          n ++;
        }

      MPFR_LOG_MSG (("End of iterations (n = %lu)\n", n));

      /* the error on v is bounded by (18n+51) ulps, or twice if there
         was an exponent loss in the final subtraction */
      err += MPFR_INT_CEIL_LOG2(18 * n + 51); /* 18n+51 should not overflow
                                                 since n is about log(p) */
      /* we should have n+2 <= 2^(p/4) [see algorithms.tex] */
      if (MPFR_LIKELY (MPFR_INT_CEIL_LOG2(n + 2) <= p / 4 &&
                       MPFR_CAN_ROUND (v, p - err, q, rnd_mode)))
        break; /* Stop the loop */

      /* Next iteration */
      MPFR_ZIV_NEXT (loop, p);
      s = MPFR_PREC2LIMBS (p);
    }
  MPFR_ZIV_FREE (loop);

  if (MPFR_UNLIKELY ((__gmpfr_flags & (MPFR_FLAGS_ALL ^ MPFR_FLAGS_INEXACT))
                     != 0))
    {
      MPFR_ASSERTN (! mpfr_overflow_p ());   /* since mpfr_clear_flags */
      MPFR_ASSERTN (! mpfr_underflow_p ());  /* since mpfr_clear_flags */
      MPFR_ASSERTN (! mpfr_divby0_p ());     /* since mpfr_clear_flags */
      MPFR_ASSERTN (! mpfr_nanflag_p ());    /* since mpfr_clear_flags */
    }

  /* Setting of the result */
  inexact = mpfr_set (r, v, rnd_mode);
  MPFR_EXP (r) -= scaleop + scaleit;

  /* Let's clean */
  MPFR_TMP_FREE(marker);

  MPFR_SAVE_EXPO_FREE (expo);
  /* From the definition of the AGM, underflow and overflow
     are not possible. */
  return mpfr_check_range (r, inexact, rnd_mode);
  /* agm(u,v) can be exact for u, v rational only for u=v.
     Proof (due to Nicolas Brisebarre): it suffices to consider
     u=1 and v<1. Then 1/AGM(1,v) = 2F1(1/2,1/2,1;1-v^2),
     and a theorem due to G.V. Chudnovsky states that for x a
     non-zero algebraic number with |x|<1, then
     2F1(1/2,1/2,1;x) and 2F1(-1/2,1/2,1;x) are algebraically
     independent over Q. */
}
