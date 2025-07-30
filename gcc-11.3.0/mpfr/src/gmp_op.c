/* Implementations of operations between mpfr and mpz/mpq data

Copyright 2001, 2003-2017 Free Software Foundation, Inc.
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

/* Init and set a mpfr_t with enough precision to store a mpz.
   This function should be called in the extended exponent range. */
static void
init_set_z (mpfr_ptr t, mpz_srcptr z)
{
  mpfr_prec_t p;
  int i;

  if (mpz_size (z) <= 1)
    p = GMP_NUMB_BITS;
  else
    MPFR_MPZ_SIZEINBASE2 (p, z);
  mpfr_init2 (t, p);
  i = mpfr_set_z (t, z, MPFR_RNDN);
  /* Possible assertion failure in case of overflow. Such cases,
     which imply that z is huge (if the function is called in
     the extended exponent range), are currently not supported,
     just like precisions around MPFR_PREC_MAX. */
  MPFR_ASSERTN (i == 0);  (void) i; /* use i to avoid a warning */
}

/* Init, set a mpfr_t with enough precision to store a mpz_t without round,
   call the function, and clear the allocated mpfr_t  */
static int
foo (mpfr_ptr x, mpfr_srcptr y, mpz_srcptr z, mpfr_rnd_t r,
     int (*f)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t))
{
  mpfr_t t;
  int i;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_SAVE_EXPO_MARK (expo);
  init_set_z (t, z);  /* There should be no exceptions. */
  i = (*f) (x, y, t, r);
  MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);
  mpfr_clear (t);
  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (x, i, r);
}

static int
foo2 (mpfr_ptr x, mpz_srcptr y, mpfr_srcptr z, mpfr_rnd_t r,
     int (*f)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t))
{
  mpfr_t t;
  int i;
  MPFR_SAVE_EXPO_DECL (expo);

  MPFR_SAVE_EXPO_MARK (expo);
  init_set_z (t, y);  /* There should be no exceptions. */
  i = (*f) (x, t, z, r);
  MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);
  mpfr_clear (t);
  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (x, i, r);
}

int
mpfr_mul_z (mpfr_ptr y, mpfr_srcptr x, mpz_srcptr z, mpfr_rnd_t r)
{
  return foo (y, x, z, r, mpfr_mul);
}

int
mpfr_div_z (mpfr_ptr y, mpfr_srcptr x, mpz_srcptr z, mpfr_rnd_t r)
{
  return foo (y, x, z, r, mpfr_div);
}

int
mpfr_add_z (mpfr_ptr y, mpfr_srcptr x, mpz_srcptr z, mpfr_rnd_t r)
{
  /* Mpz 0 is unsigned */
  if (MPFR_UNLIKELY (mpz_sgn (z) == 0))
    return mpfr_set (y, x, r);
  else
    return foo (y, x, z, r, mpfr_add);
}

int
mpfr_sub_z (mpfr_ptr y, mpfr_srcptr x, mpz_srcptr z, mpfr_rnd_t r)
{
  /* Mpz 0 is unsigned */
  if (MPFR_UNLIKELY (mpz_sgn (z) == 0))
    return mpfr_set (y, x, r);
  else
    return foo (y, x, z, r, mpfr_sub);
}

int
mpfr_z_sub (mpfr_ptr y, mpz_srcptr x, mpfr_srcptr z, mpfr_rnd_t r)
{
  /* Mpz 0 is unsigned */
  if (MPFR_UNLIKELY (mpz_sgn (x) == 0))
    return mpfr_neg (y, z, r);
  else
    return foo2 (y, x, z, r, mpfr_sub);
}

int
mpfr_cmp_z (mpfr_srcptr x, mpz_srcptr z)
{
  mpfr_t t;
  int res;
  mpfr_prec_t p;
  unsigned int flags;

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    return mpfr_cmp_si (x, mpz_sgn (z));

  if (mpz_size (z) <= 1)
    p = GMP_NUMB_BITS;
  else
    MPFR_MPZ_SIZEINBASE2 (p, z);
  mpfr_init2 (t, p);
  flags = __gmpfr_flags;
  if (mpfr_set_z (t, z, MPFR_RNDN))
    {
      /* overflow (t is an infinity) or underflow */
      mpfr_div_2ui (t, t, 2, MPFR_RNDZ);  /* if underflow, set t to zero */
      __gmpfr_flags = flags;  /* restore the flags */
      /* The real value of t (= z), which falls outside the exponent range,
         has been replaced by an equivalent value for the comparison: zero
         or an infinity. */
    }
  res = mpfr_cmp (x, t);
  mpfr_clear (t);
  return res;
}

/* Compute y = RND(x*n/d), where n and d are mpz integers.
   An integer 0 is assumed to have a positive sign.
   This function is used by mpfr_mul_q and mpfr_div_q.
   Note: the status of the rational 0/(-1) is not clear (if there is
   a signed infinity, there should be a signed zero). But infinities
   are not currently supported/documented in GMP, and if the rational
   is canonicalized as it should be, the case 0/(-1) cannot occur. */
static int
mpfr_muldiv_z (mpfr_ptr y, mpfr_srcptr x, mpz_srcptr n, mpz_srcptr d,
               mpfr_rnd_t rnd_mode)
{
  if (MPFR_UNLIKELY (mpz_sgn (n) == 0))
    {
      if (MPFR_UNLIKELY (mpz_sgn (d) == 0))
        MPFR_SET_NAN (y);
      else
        {
          mpfr_mul_ui (y, x, 0, MPFR_RNDN);  /* exact: +0, -0 or NaN */
          if (MPFR_UNLIKELY (mpz_sgn (d) < 0))
            MPFR_CHANGE_SIGN (y);
        }
      return 0;
    }
  else if (MPFR_UNLIKELY (mpz_sgn (d) == 0))
    {
      mpfr_div_ui (y, x, 0, MPFR_RNDN);  /* exact: +Inf, -Inf or NaN */
      if (MPFR_UNLIKELY (mpz_sgn (n) < 0))
        MPFR_CHANGE_SIGN (y);
      return 0;
    }
  else
    {
      mpfr_prec_t p;
      mpfr_t tmp;
      int inexact;
      MPFR_SAVE_EXPO_DECL (expo);

      MPFR_SAVE_EXPO_MARK (expo);

      /* With the current MPFR code, using mpfr_mul_z and mpfr_div_z
         for the general case should be faster than doing everything
         in mpn, mpz and/or mpq. MPFR_SAVE_EXPO_MARK could be avoided
         here, but it would be more difficult to handle corner cases. */
      MPFR_MPZ_SIZEINBASE2 (p, n);
      mpfr_init2 (tmp, MPFR_PREC (x) + p);
      inexact = mpfr_mul_z (tmp, x, n, MPFR_RNDN);
      /* Since |n| >= 1, an underflow is not possible. And the precision of
         tmp has been chosen so that inexact != 0 iff there's an overflow. */
      if (MPFR_UNLIKELY (inexact != 0))
        {
          mpfr_t x0;
          mpfr_exp_t ex;
          MPFR_BLOCK_DECL (flags);

          /* intermediate overflow case */
          MPFR_ASSERTD (mpfr_inf_p (tmp));
          ex = MPFR_GET_EXP (x);  /* x is a pure FP number */
          MPFR_ALIAS (x0, x, MPFR_SIGN(x), 0);  /* x0 = x / 2^ex */
          MPFR_BLOCK (flags,
                      inexact = mpfr_mul_z (tmp, x0, n, MPFR_RNDN);
                      MPFR_ASSERTD (inexact == 0);
                      inexact = mpfr_div_z (y, tmp, d, rnd_mode);
                      /* Just in case the division underflows
                         (highly unlikely, not supported)... */
                      MPFR_ASSERTN (!MPFR_BLOCK_EXCEP));
          MPFR_EXP (y) += ex;
          /* Detect highly unlikely, not supported corner cases... */
          MPFR_ASSERTN (MPFR_EXP (y) >= __gmpfr_emin && MPFR_IS_PURE_FP (y));
          /* The potential overflow will be detected by mpfr_check_range. */
        }
      else
        inexact = mpfr_div_z (y, tmp, d, rnd_mode);

      mpfr_clear (tmp);

      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_check_range (y, inexact, rnd_mode);
    }
}

int
mpfr_mul_q (mpfr_ptr y, mpfr_srcptr x, mpq_srcptr z, mpfr_rnd_t rnd_mode)
{
  return mpfr_muldiv_z (y, x, mpq_numref (z), mpq_denref (z), rnd_mode);
}

int
mpfr_div_q (mpfr_ptr y, mpfr_srcptr x, mpq_srcptr z, mpfr_rnd_t rnd_mode)
{
  return mpfr_muldiv_z (y, x, mpq_denref (z), mpq_numref (z), rnd_mode);
}

int
mpfr_add_q (mpfr_ptr y, mpfr_srcptr x, mpq_srcptr z, mpfr_rnd_t rnd_mode)
{
  mpfr_t      t,q;
  mpfr_prec_t p;
  mpfr_exp_t  err;
  int res;
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_ZIV_DECL (loop);

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (x))
        {
          if (MPFR_UNLIKELY (mpz_sgn (mpq_denref (z)) == 0 &&
                             MPFR_MULT_SIGN (mpz_sgn (mpq_numref (z)),
                                             MPFR_SIGN (x)) <= 0))
            {
              MPFR_SET_NAN (y);
              MPFR_RET_NAN;
            }
          MPFR_SET_INF (y);
          MPFR_SET_SAME_SIGN (y, x);
          MPFR_RET (0);
        }
      else
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (x));
          if (MPFR_UNLIKELY (mpq_sgn (z) == 0))
            return mpfr_set (y, x, rnd_mode); /* signed 0 - Unsigned 0 */
          else
            return mpfr_set_q (y, z, rnd_mode);
        }
    }

  MPFR_SAVE_EXPO_MARK (expo);

  p = MPFR_PREC (y) + 10;
  mpfr_init2 (t, p);
  mpfr_init2 (q, p);

  MPFR_ZIV_INIT (loop, p);
  for (;;)
    {
      MPFR_BLOCK_DECL (flags);

      res = mpfr_set_q (q, z, MPFR_RNDN);  /* Error <= 1/2 ulp(q) */
      /* If z if @INF@ (1/0), res = 0, so it quits immediately */
      if (MPFR_UNLIKELY (res == 0))
        /* Result is exact so we can add it directly! */
        {
          res = mpfr_add (y, x, q, rnd_mode);
          break;
        }
      MPFR_BLOCK (flags, mpfr_add (t, x, q, MPFR_RNDN));
      /* Error on t is <= 1/2 ulp(t), except in case of overflow/underflow,
         but such an exception is very unlikely as it would be possible
         only if q has a huge numerator or denominator. Not supported! */
      MPFR_ASSERTN (! (MPFR_OVERFLOW (flags) || MPFR_UNDERFLOW (flags)));
      /* Error / ulp(t)      <= 1/2 + 1/2 * 2^(EXP(q)-EXP(t))
         If EXP(q)-EXP(t)>0, <= 2^(EXP(q)-EXP(t)-1)*(1+2^-(EXP(q)-EXP(t)))
                             <= 2^(EXP(q)-EXP(t))
         If EXP(q)-EXP(t)<0, <= 2^0 */
      /* We can get 0, but we can't round since q is inexact */
      if (MPFR_LIKELY (!MPFR_IS_ZERO (t)))
        {
          err = (mpfr_exp_t) p - 1 - MAX (MPFR_GET_EXP(q)-MPFR_GET_EXP(t), 0);
          if (MPFR_LIKELY (MPFR_CAN_ROUND (t, err, MPFR_PREC (y), rnd_mode)))
            {
              res = mpfr_set (y, t, rnd_mode);
              break;
            }
        }
      MPFR_ZIV_NEXT (loop, p);
      mpfr_set_prec (t, p);
      mpfr_set_prec (q, p);
    }
  MPFR_ZIV_FREE (loop);
  mpfr_clear (t);
  mpfr_clear (q);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, res, rnd_mode);
}

int
mpfr_sub_q (mpfr_ptr y, mpfr_srcptr x, mpq_srcptr z,mpfr_rnd_t rnd_mode)
{
  mpfr_t t,q;
  mpfr_prec_t p;
  int res;
  mpfr_exp_t err;
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_ZIV_DECL (loop);

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    {
      if (MPFR_IS_NAN (x))
        {
          MPFR_SET_NAN (y);
          MPFR_RET_NAN;
        }
      else if (MPFR_IS_INF (x))
        {
          if (MPFR_UNLIKELY (mpz_sgn (mpq_denref (z)) == 0 &&
                             MPFR_MULT_SIGN (mpz_sgn (mpq_numref (z)),
                                             MPFR_SIGN (x)) >= 0))
            {
              MPFR_SET_NAN (y);
              MPFR_RET_NAN;
            }
          MPFR_SET_INF (y);
          MPFR_SET_SAME_SIGN (y, x);
          MPFR_RET (0);
        }
      else
        {
          MPFR_ASSERTD (MPFR_IS_ZERO (x));

          if (MPFR_UNLIKELY (mpq_sgn (z) == 0))
            return mpfr_set (y, x, rnd_mode); /* signed 0 - Unsigned 0 */
          else
            {
              res =  mpfr_set_q (y, z, MPFR_INVERT_RND (rnd_mode));
              MPFR_CHANGE_SIGN (y);
              return -res;
            }
        }
    }

  MPFR_SAVE_EXPO_MARK (expo);

  p = MPFR_PREC (y) + 10;
  mpfr_init2 (t, p);
  mpfr_init2 (q, p);

  MPFR_ZIV_INIT (loop, p);
  for(;;)
    {
      MPFR_BLOCK_DECL (flags);

      res = mpfr_set_q(q, z, MPFR_RNDN);  /* Error <= 1/2 ulp(q) */
      /* If z if @INF@ (1/0), res = 0, so it quits immediately */
      if (MPFR_UNLIKELY (res == 0))
        /* Result is exact so we can add it directly!*/
        {
          res = mpfr_sub (y, x, q, rnd_mode);
          break;
        }
      MPFR_BLOCK (flags, mpfr_sub (t, x, q, MPFR_RNDN));
      /* Error on t is <= 1/2 ulp(t), except in case of overflow/underflow,
         but such an exception is very unlikely as it would be possible
         only if q has a huge numerator or denominator. Not supported! */
      MPFR_ASSERTN (! (MPFR_OVERFLOW (flags) || MPFR_UNDERFLOW (flags)));
      /* Error / ulp(t)      <= 1/2 + 1/2 * 2^(EXP(q)-EXP(t))
         If EXP(q)-EXP(t)>0, <= 2^(EXP(q)-EXP(t)-1)*(1+2^-(EXP(q)-EXP(t)))
                             <= 2^(EXP(q)-EXP(t))
                             If EXP(q)-EXP(t)<0, <= 2^0 */
      /* We can get 0, but we can't round since q is inexact */
      if (MPFR_LIKELY (!MPFR_IS_ZERO (t)))
        {
          err = (mpfr_exp_t) p - 1 - MAX (MPFR_GET_EXP(q)-MPFR_GET_EXP(t), 0);
          res = MPFR_CAN_ROUND (t, err, MPFR_PREC (y), rnd_mode);
          if (MPFR_LIKELY (res != 0))  /* We can round! */
            {
              res = mpfr_set (y, t, rnd_mode);
              break;
            }
        }
      MPFR_ZIV_NEXT (loop, p);
      mpfr_set_prec (t, p);
      mpfr_set_prec (q, p);
    }
  MPFR_ZIV_FREE (loop);
  mpfr_clear (t);
  mpfr_clear (q);

  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (y, res, rnd_mode);
}

int
mpfr_cmp_q (mpfr_srcptr x, mpq_srcptr q)
{
  mpfr_t t;
  int res;
  mpfr_prec_t p;
  MPFR_SAVE_EXPO_DECL (expo);

  if (MPFR_UNLIKELY (mpq_denref (q) == 0))
    {
      /* q is an infinity or NaN */
      mpfr_init2 (t, 2);
      mpfr_set_q (t, q, MPFR_RNDN);
      res = mpfr_cmp (x, t);
      mpfr_clear (t);
      return res;
    }

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    return mpfr_cmp_si (x, mpq_sgn (q));

  MPFR_SAVE_EXPO_MARK (expo);

  /* x < a/b ? <=> x*b < a */
  MPFR_MPZ_SIZEINBASE2 (p, mpq_denref (q));
  mpfr_init2 (t, MPFR_PREC(x) + p);
  res = mpfr_mul_z (t, x, mpq_denref (q), MPFR_RNDN);
  MPFR_ASSERTD (res == 0);
  res = mpfr_cmp_z (t, mpq_numref (q));
  mpfr_clear (t);

  MPFR_SAVE_EXPO_FREE (expo);
  return res;
}

int
mpfr_cmp_f (mpfr_srcptr x, mpf_srcptr z)
{
  mpfr_t t;
  int res;
  MPFR_SAVE_EXPO_DECL (expo);

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x)))
    return mpfr_cmp_si (x, mpf_sgn (z));

  MPFR_SAVE_EXPO_MARK (expo);

  mpfr_init2 (t, MPFR_PREC_MIN + ABS(SIZ(z)) * GMP_NUMB_BITS );
  res = mpfr_set_f (t, z, MPFR_RNDN);
  MPFR_ASSERTD (res == 0);
  res = mpfr_cmp (x, t);
  mpfr_clear (t);

  MPFR_SAVE_EXPO_FREE (expo);
  return res;
}
