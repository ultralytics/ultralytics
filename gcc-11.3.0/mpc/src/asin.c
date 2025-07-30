/* mpc_asin -- arcsine of a complex number.

Copyright (C) 2009, 2010, 2011 INRIA

This file is part of GNU MPC.

GNU MPC is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

GNU MPC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see http://www.gnu.org/licenses/ .
*/

#include "mpc-impl.h"

int
mpc_asin (mpc_ptr rop, mpc_srcptr op, mpc_rnd_t rnd)
{
  mpfr_prec_t p, p_re, p_im, incr_p = 0;
  mpfr_rnd_t rnd_re, rnd_im;
  mpc_t z1;
  int inex;

  /* special values */
  if (mpfr_nan_p (mpc_realref (op)) || mpfr_nan_p (mpc_imagref (op)))
    {
      if (mpfr_inf_p (mpc_realref (op)) || mpfr_inf_p (mpc_imagref (op)))
        {
          mpfr_set_nan (mpc_realref (rop));
          mpfr_set_inf (mpc_imagref (rop), mpfr_signbit (mpc_imagref (op)) ? -1 : +1);
        }
      else if (mpfr_zero_p (mpc_realref (op)))
        {
          mpfr_set (mpc_realref (rop), mpc_realref (op), GMP_RNDN);
          mpfr_set_nan (mpc_imagref (rop));
        }
      else
        {
          mpfr_set_nan (mpc_realref (rop));
          mpfr_set_nan (mpc_imagref (rop));
        }

      return 0;
    }

  if (mpfr_inf_p (mpc_realref (op)) || mpfr_inf_p (mpc_imagref (op)))
    {
      int inex_re;
      if (mpfr_inf_p (mpc_realref (op)))
        {
          int inf_im = mpfr_inf_p (mpc_imagref (op));

          inex_re = set_pi_over_2 (mpc_realref (rop),
             (mpfr_signbit (mpc_realref (op)) ? -1 : 1), MPC_RND_RE (rnd));
          mpfr_set_inf (mpc_imagref (rop), (mpfr_signbit (mpc_imagref (op)) ? -1 : 1));

          if (inf_im)
            mpfr_div_2ui (mpc_realref (rop), mpc_realref (rop), 1, GMP_RNDN);
        }
      else
        {
          mpfr_set_zero (mpc_realref (rop), (mpfr_signbit (mpc_realref (op)) ? -1 : 1));
          inex_re = 0;
          mpfr_set_inf (mpc_imagref (rop), (mpfr_signbit (mpc_imagref (op)) ? -1 : 1));
        }

      return MPC_INEX (inex_re, 0);
    }

  /* pure real argument */
  if (mpfr_zero_p (mpc_imagref (op)))
    {
      int inex_re;
      int inex_im;
      int s_im;
      s_im = mpfr_signbit (mpc_imagref (op));

      if (mpfr_cmp_ui (mpc_realref (op), 1) > 0)
        {
          if (s_im)
            inex_im = -mpfr_acosh (mpc_imagref (rop), mpc_realref (op),
                                   INV_RND (MPC_RND_IM (rnd)));
          else
            inex_im = mpfr_acosh (mpc_imagref (rop), mpc_realref (op),
                                  MPC_RND_IM (rnd));
          inex_re = set_pi_over_2 (mpc_realref (rop),
             (mpfr_signbit (mpc_realref (op)) ? -1 : 1), MPC_RND_RE (rnd));
          if (s_im)
            mpc_conj (rop, rop, MPC_RNDNN);
        }
      else if (mpfr_cmp_si (mpc_realref (op), -1) < 0)
        {
          mpfr_t minus_op_re;
          minus_op_re[0] = mpc_realref (op)[0];
          MPFR_CHANGE_SIGN (minus_op_re);

          if (s_im)
            inex_im = -mpfr_acosh (mpc_imagref (rop), minus_op_re,
                                   INV_RND (MPC_RND_IM (rnd)));
          else
            inex_im = mpfr_acosh (mpc_imagref (rop), minus_op_re,
                                  MPC_RND_IM (rnd));
          inex_re = set_pi_over_2 (mpc_realref (rop),
             (mpfr_signbit (mpc_realref (op)) ? -1 : 1), MPC_RND_RE (rnd));
          if (s_im)
            mpc_conj (rop, rop, MPC_RNDNN);
        }
      else
        {
          inex_im = mpfr_set_ui (mpc_imagref (rop), 0, MPC_RND_IM (rnd));
          if (s_im)
            mpfr_neg (mpc_imagref (rop), mpc_imagref (rop), GMP_RNDN);
          inex_re = mpfr_asin (mpc_realref (rop), mpc_realref (op), MPC_RND_RE (rnd));
        }

      return MPC_INEX (inex_re, inex_im);
    }

  /* pure imaginary argument */
  if (mpfr_zero_p (mpc_realref (op)))
    {
      int inex_im;
      int s;
      s = mpfr_signbit (mpc_realref (op));
      mpfr_set_ui (mpc_realref (rop), 0, GMP_RNDN);
      if (s)
        mpfr_neg (mpc_realref (rop), mpc_realref (rop), GMP_RNDN);
      inex_im = mpfr_asinh (mpc_imagref (rop), mpc_imagref (op), MPC_RND_IM (rnd));

      return MPC_INEX (0, inex_im);
    }

  /* regular complex: asin(z) = -i*log(i*z+sqrt(1-z^2)) */
  p_re = mpfr_get_prec (mpc_realref(rop));
  p_im = mpfr_get_prec (mpc_imagref(rop));
  rnd_re = MPC_RND_RE(rnd);
  rnd_im = MPC_RND_IM(rnd);
  p = p_re >= p_im ? p_re : p_im;
  mpc_init2 (z1, p);
  while (1)
  {
    mpfr_exp_t ex, ey, err;

    p += mpc_ceil_log2 (p) + 3 + incr_p; /* incr_p is zero initially */
    incr_p = p / 2;
    mpfr_set_prec (mpc_realref(z1), p);
    mpfr_set_prec (mpc_imagref(z1), p);

    /* z1 <- z^2 */
    mpc_sqr (z1, op, MPC_RNDNN);
    /* err(x) <= 1/2 ulp(x), err(y) <= 1/2 ulp(y) */
    /* z1 <- 1-z1 */
    ex = mpfr_get_exp (mpc_realref(z1));
    mpfr_ui_sub (mpc_realref(z1), 1, mpc_realref(z1), GMP_RNDN);
    mpfr_neg (mpc_imagref(z1), mpc_imagref(z1), GMP_RNDN);
    ex = ex - mpfr_get_exp (mpc_realref(z1));
    ex = (ex <= 0) ? 0 : ex;
    /* err(x) <= 2^ex * ulp(x) */
    ex = ex + mpfr_get_exp (mpc_realref(z1)) - p;
    /* err(x) <= 2^ex */
    ey = mpfr_get_exp (mpc_imagref(z1)) - p - 1;
    /* err(y) <= 2^ey */
    ex = (ex >= ey) ? ex : ey; /* err(x), err(y) <= 2^ex, i.e., the norm
                                  of the error is bounded by |h|<=2^(ex+1/2) */
    /* z1 <- sqrt(z1): if z1 = z + h, then sqrt(z1) = sqrt(z) + h/2/sqrt(t) */
    ey = mpfr_get_exp (mpc_realref(z1)) >= mpfr_get_exp (mpc_imagref(z1))
      ? mpfr_get_exp (mpc_realref(z1)) : mpfr_get_exp (mpc_imagref(z1));
    /* we have |z1| >= 2^(ey-1) thus 1/|z1| <= 2^(1-ey) */
    mpc_sqrt (z1, z1, MPC_RNDNN);
    ex = (2 * ex + 1) - 2 - (ey - 1); /* |h^2/4/|t| <= 2^ex */
    ex = (ex + 1) / 2; /* ceil(ex/2) */
    /* express ex in terms of ulp(z1) */
    ey = mpfr_get_exp (mpc_realref(z1)) <= mpfr_get_exp (mpc_imagref(z1))
      ? mpfr_get_exp (mpc_realref(z1)) : mpfr_get_exp (mpc_imagref(z1));
    ex = ex - ey + p;
    /* take into account the rounding error in the mpc_sqrt call */
    err = (ex <= 0) ? 1 : ex + 1;
    /* err(x) <= 2^err * ulp(x), err(y) <= 2^err * ulp(y) */
    /* z1 <- i*z + z1 */
    ex = mpfr_get_exp (mpc_realref(z1));
    ey = mpfr_get_exp (mpc_imagref(z1));
    mpfr_sub (mpc_realref(z1), mpc_realref(z1), mpc_imagref(op), GMP_RNDN);
    mpfr_add (mpc_imagref(z1), mpc_imagref(z1), mpc_realref(op), GMP_RNDN);
    if (mpfr_cmp_ui (mpc_realref(z1), 0) == 0 || mpfr_cmp_ui (mpc_imagref(z1), 0) == 0)
      continue;
    ex -= mpfr_get_exp (mpc_realref(z1)); /* cancellation in x */
    ey -= mpfr_get_exp (mpc_imagref(z1)); /* cancellation in y */
    ex = (ex >= ey) ? ex : ey; /* maximum cancellation */
    err += ex;
    err = (err <= 0) ? 1 : err + 1; /* rounding error in sub/add */
    /* z1 <- log(z1): if z1 = z + h, then log(z1) = log(z) + h/t with
       |t| >= min(|z1|,|z|) */
    ex = mpfr_get_exp (mpc_realref(z1));
    ey = mpfr_get_exp (mpc_imagref(z1));
    ex = (ex >= ey) ? ex : ey;
    err += ex - p; /* revert to absolute error <= 2^err */
    mpc_log (z1, z1, GMP_RNDN);
    err -= ex - 1; /* 1/|t| <= 1/|z| <= 2^(1-ex) */
    /* express err in terms of ulp(z1) */
    ey = mpfr_get_exp (mpc_realref(z1)) <= mpfr_get_exp (mpc_imagref(z1))
      ? mpfr_get_exp (mpc_realref(z1)) : mpfr_get_exp (mpc_imagref(z1));
    err = err - ey + p;
    /* take into account the rounding error in the mpc_log call */
    err = (err <= 0) ? 1 : err + 1;
    /* z1 <- -i*z1 */
    mpfr_swap (mpc_realref(z1), mpc_imagref(z1));
    mpfr_neg (mpc_imagref(z1), mpc_imagref(z1), GMP_RNDN);
    if (mpfr_can_round (mpc_realref(z1), p - err, GMP_RNDN, GMP_RNDZ,
                        p_re + (rnd_re == GMP_RNDN)) &&
        mpfr_can_round (mpc_imagref(z1), p - err, GMP_RNDN, GMP_RNDZ,
                        p_im + (rnd_im == GMP_RNDN)))
      break;
  }

  inex = mpc_set (rop, z1, rnd);
  mpc_clear (z1);

  return inex;
}
