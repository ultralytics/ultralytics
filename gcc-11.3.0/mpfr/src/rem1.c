/* mpfr_rem1 -- internal function
   mpfr_fmod -- compute the floating-point remainder of x/y
   mpfr_remquo and mpfr_remainder -- argument reduction functions

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

# include "mpfr-impl.h"

/* we return as many bits as we can, keeping just one bit for the sign */
# define WANTED_BITS (sizeof(long) * CHAR_BIT - 1)

/*
  rem1 works as follows:
  The first rounding mode rnd_q indicate if we are actually computing
  a fmod (MPFR_RNDZ) or a remainder/remquo (MPFR_RNDN).

  Let q = x/y rounded to an integer in the direction rnd_q.
  Put x - q*y in rem, rounded according to rnd.
  If quo is not null, the value stored in *quo has the sign of q,
  and agrees with q with the 2^n low order bits.
  In other words, *quo = q (mod 2^n) and *quo q >= 0.
  If rem is zero, then it has the sign of x.
  The returned 'int' is the inexact flag giving the place of rem wrt x - q*y.

  If x or y is NaN: *quo is undefined, rem is NaN.
  If x is Inf, whatever y: *quo is undefined, rem is NaN.
  If y is Inf, x not NaN nor Inf: *quo is 0, rem is x.
  If y is 0, whatever x: *quo is undefined, rem is NaN.
  If x is 0, whatever y (not NaN nor 0): *quo is 0, rem is x.

  Otherwise if x and y are neither NaN, Inf nor 0, q is always defined,
  thus *quo is.
  Since |x - q*y| <= y/2, no overflow is possible.
  Only an underflow is possible when y is very small.
 */

static int
mpfr_rem1 (mpfr_ptr rem, long *quo, mpfr_rnd_t rnd_q,
           mpfr_srcptr x, mpfr_srcptr y, mpfr_rnd_t rnd)
{
  mpfr_exp_t ex, ey;
  int compare, inex, q_is_odd, sign, signx = MPFR_SIGN (x);
  mpz_t mx, my, r;
  int tiny = 0;

  MPFR_ASSERTD (rnd_q == MPFR_RNDN || rnd_q == MPFR_RNDZ);

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (x) || MPFR_IS_SINGULAR (y)))
    {
      if (MPFR_IS_NAN (x) || MPFR_IS_NAN (y) || MPFR_IS_INF (x)
          || MPFR_IS_ZERO (y))
        {
          /* for remquo, quo is undefined */
          MPFR_SET_NAN (rem);
          MPFR_RET_NAN;
        }
      else                      /* either y is Inf and x is 0 or non-special,
                                   or x is 0 and y is non-special,
                                   in both cases the quotient is zero. */
        {
          if (quo)
            *quo = 0;
          return mpfr_set (rem, x, rnd);
        }
    }

  /* now neither x nor y is NaN, Inf or zero */

  mpz_init (mx);
  mpz_init (my);
  mpz_init (r);

  ex = mpfr_get_z_2exp (mx, x);  /* x = mx*2^ex */
  ey = mpfr_get_z_2exp (my, y);  /* y = my*2^ey */

  /* to get rid of sign problems, we compute it separately:
     quo(-x,-y) = quo(x,y), rem(-x,-y) = -rem(x,y)
     quo(-x,y) = -quo(x,y), rem(-x,y)  = -rem(x,y)
     thus quo = sign(x/y)*quo(|x|,|y|), rem = sign(x)*rem(|x|,|y|) */
  sign = (signx == MPFR_SIGN (y)) ? 1 : -1;
  mpz_abs (mx, mx);
  mpz_abs (my, my);
  q_is_odd = 0;

  /* divide my by 2^k if possible to make operations mod my easier */
  {
    unsigned long k = mpz_scan1 (my, 0);
    ey += k;
    mpz_fdiv_q_2exp (my, my, k);
  }

  if (ex <= ey)
    {
      /* q = x/y = mx/(my*2^(ey-ex)) */

      /* First detect cases where q=0, to avoid creating a huge number
         my*2^(ey-ex): if sx = mpz_sizeinbase (mx, 2) and sy =
         mpz_sizeinbase (my, 2), we have x < 2^(ex + sx) and
         y >= 2^(ey + sy - 1), thus if ex + sx <= ey + sy - 1
         the quotient is 0 */
      if (ex + (mpfr_exp_t) mpz_sizeinbase (mx, 2) <
          ey + (mpfr_exp_t) mpz_sizeinbase (my, 2))
        {
          tiny = 1;
          mpz_set (r, mx);
          mpz_set_ui (mx, 0);
        }
      else
        {
          mpz_mul_2exp (my, my, ey - ex);   /* divide mx by my*2^(ey-ex) */

          /* since mx > 0 and my > 0, we can use mpz_tdiv_qr in all cases */
          mpz_tdiv_qr (mx, r, mx, my);
          /* 0 <= |r| <= |my|, r has the same sign as mx */
        }

      if (rnd_q == MPFR_RNDN)
        q_is_odd = mpz_tstbit (mx, 0);
      if (quo)                  /* mx is the quotient */
        {
          mpz_tdiv_r_2exp (mx, mx, WANTED_BITS);
          *quo = mpz_get_si (mx);
        }
    }
  else                          /* ex > ey */
    {
      if (quo) /* remquo case */
        /* for remquo, to get the low WANTED_BITS more bits of the quotient,
           we first compute R =  X mod Y*2^WANTED_BITS, where X and Y are
           defined below. Then the low WANTED_BITS of the quotient are
           floor(R/Y). */
        mpz_mul_2exp (my, my, WANTED_BITS);     /* 2^WANTED_BITS*Y */

      else if (rnd_q == MPFR_RNDN) /* remainder case */
        /* Let X = mx*2^(ex-ey) and Y = my. Then both X and Y are integers.
           Assume X = R mod Y, then x = X*2^ey = R*2^ey mod (Y*2^ey=y).
           To be able to perform the rounding, we need the least significant
           bit of the quotient, i.e., one more bit in the remainder,
           which is obtained by dividing by 2Y. */
        mpz_mul_2exp (my, my, 1);       /* 2Y */

      mpz_set_ui (r, 2);
      mpz_powm_ui (r, r, ex - ey, my);  /* 2^(ex-ey) mod my */
      mpz_mul (r, r, mx);
      mpz_mod (r, r, my);

      if (quo)                  /* now 0 <= r < 2^WANTED_BITS*Y */
        {
          mpz_fdiv_q_2exp (my, my, WANTED_BITS);   /* back to Y */
          mpz_tdiv_qr (mx, r, r, my);
          /* oldr = mx*my + newr */
          *quo = mpz_get_si (mx);
          q_is_odd = *quo & 1;
        }
      else if (rnd_q == MPFR_RNDN) /* now 0 <= r < 2Y in the remainder case */
        {
          mpz_fdiv_q_2exp (my, my, 1);     /* back to Y */
          /* least significant bit of q */
          q_is_odd = mpz_cmpabs (r, my) >= 0;
          if (q_is_odd)
            mpz_sub (r, r, my);
        }
      /* now 0 <= |r| < |my|, and if needed,
         q_is_odd is the least significant bit of q */
    }

  if (mpz_cmp_ui (r, 0) == 0)
    {
      inex = mpfr_set_ui (rem, 0, MPFR_RNDN);
      /* take into account sign of x */
      if (signx < 0)
        mpfr_neg (rem, rem, MPFR_RNDN);
    }
  else
    {
      if (rnd_q == MPFR_RNDN)
        {
          /* FIXME: the comparison 2*r < my could be done more efficiently
             at the mpn level */
          mpz_mul_2exp (r, r, 1);
          /* if tiny=1, we should compare r with my*2^(ey-ex) */
          if (tiny)
            {
              if (ex + (mpfr_exp_t) mpz_sizeinbase (r, 2) <
                  ey + (mpfr_exp_t) mpz_sizeinbase (my, 2))
                compare = 0; /* r*2^ex < my*2^ey */
              else
                {
                  mpz_mul_2exp (my, my, ey - ex);
                  compare = mpz_cmpabs (r, my);
                }
            }
          else
            compare = mpz_cmpabs (r, my);
          mpz_fdiv_q_2exp (r, r, 1);
          compare = ((compare > 0) ||
                     ((rnd_q == MPFR_RNDN) && (compare == 0) && q_is_odd));
          /* if compare != 0, we need to subtract my to r, and add 1 to quo */
          if (compare)
            {
              mpz_sub (r, r, my);
              if (quo && (rnd_q == MPFR_RNDN))
                *quo += 1;
            }
        }
      /* take into account sign of x */
      if (signx < 0)
        mpz_neg (r, r);
      inex = mpfr_set_z_2exp (rem, r, ex > ey ? ey : ex, rnd);
    }

  if (quo)
    *quo *= sign;

  mpz_clear (mx);
  mpz_clear (my);
  mpz_clear (r);

  return inex;
}

int
mpfr_remainder (mpfr_ptr rem, mpfr_srcptr x, mpfr_srcptr y, mpfr_rnd_t rnd)
{
  return mpfr_rem1 (rem, (long *) 0, MPFR_RNDN, x, y, rnd);
}

int
mpfr_remquo (mpfr_ptr rem, long *quo,
             mpfr_srcptr x, mpfr_srcptr y, mpfr_rnd_t rnd)
{
  return mpfr_rem1 (rem, quo, MPFR_RNDN, x, y, rnd);
}

int
mpfr_fmod (mpfr_ptr rem, mpfr_srcptr x, mpfr_srcptr y, mpfr_rnd_t rnd)
{
  return mpfr_rem1 (rem, (long *) 0, MPFR_RNDZ, x, y, rnd);
}
