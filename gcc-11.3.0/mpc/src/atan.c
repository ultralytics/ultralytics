/* mpc_atan -- arctangent of a complex number.

Copyright (C) 2009, 2010, 2011, 2012, 2013 INRIA

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

#include <stdio.h>
#include "mpc-impl.h"

/* set rop to
   -pi/2 if s < 0
   +pi/2 else
   rounded in the direction rnd
*/
int
set_pi_over_2 (mpfr_ptr rop, int s, mpfr_rnd_t rnd)
{
  int inex;

  inex = mpfr_const_pi (rop, s < 0 ? INV_RND (rnd) : rnd);
  mpfr_div_2ui (rop, rop, 1, GMP_RNDN);
  if (s < 0)
    {
      inex = -inex;
      mpfr_neg (rop, rop, GMP_RNDN);
    }

  return inex;
}

int
mpc_atan (mpc_ptr rop, mpc_srcptr op, mpc_rnd_t rnd)
{
  int s_re;
  int s_im;
  int inex_re;
  int inex_im;
  int inex;

  inex_re = 0;
  inex_im = 0;
  s_re = mpfr_signbit (mpc_realref (op));
  s_im = mpfr_signbit (mpc_imagref (op));

  /* special values */
  if (mpfr_nan_p (mpc_realref (op)) || mpfr_nan_p (mpc_imagref (op)))
    {
      if (mpfr_nan_p (mpc_realref (op)))
        {
          mpfr_set_nan (mpc_realref (rop));
          if (mpfr_zero_p (mpc_imagref (op)) || mpfr_inf_p (mpc_imagref (op)))
            {
              mpfr_set_ui (mpc_imagref (rop), 0, GMP_RNDN);
              if (s_im)
                mpc_conj (rop, rop, MPC_RNDNN);
            }
          else
            mpfr_set_nan (mpc_imagref (rop));
        }
      else
        {
          if (mpfr_inf_p (mpc_realref (op)))
            {
              inex_re = set_pi_over_2 (mpc_realref (rop), -s_re, MPC_RND_RE (rnd));
              mpfr_set_ui (mpc_imagref (rop), 0, GMP_RNDN);
            }
          else
            {
              mpfr_set_nan (mpc_realref (rop));
              mpfr_set_nan (mpc_imagref (rop));
            }
        }
      return MPC_INEX (inex_re, 0);
    }

  if (mpfr_inf_p (mpc_realref (op)) || mpfr_inf_p (mpc_imagref (op)))
    {
      inex_re = set_pi_over_2 (mpc_realref (rop), -s_re, MPC_RND_RE (rnd));

      mpfr_set_ui (mpc_imagref (rop), 0, GMP_RNDN);
      if (s_im)
        mpc_conj (rop, rop, GMP_RNDN);

      return MPC_INEX (inex_re, 0);
    }

  /* pure real argument */
  if (mpfr_zero_p (mpc_imagref (op)))
    {
      inex_re = mpfr_atan (mpc_realref (rop), mpc_realref (op), MPC_RND_RE (rnd));

      mpfr_set_ui (mpc_imagref (rop), 0, GMP_RNDN);
      if (s_im)
        mpc_conj (rop, rop, GMP_RNDN);

      return MPC_INEX (inex_re, 0);
    }

  /* pure imaginary argument */
  if (mpfr_zero_p (mpc_realref (op)))
    {
      int cmp_1;

      if (s_im)
        cmp_1 = -mpfr_cmp_si (mpc_imagref (op), -1);
      else
        cmp_1 = mpfr_cmp_ui (mpc_imagref (op), +1);

      if (cmp_1 < 0)
        {
          /* atan(+0+iy) = +0 +i*atanh(y), if |y| < 1
             atan(-0+iy) = -0 +i*atanh(y), if |y| < 1 */

          mpfr_set_ui (mpc_realref (rop), 0, GMP_RNDN);
          if (s_re)
            mpfr_neg (mpc_realref (rop), mpc_realref (rop), GMP_RNDN);

          inex_im = mpfr_atanh (mpc_imagref (rop), mpc_imagref (op), MPC_RND_IM (rnd));
        }
      else if (cmp_1 == 0)
        {
          /* atan(+/-0 +i) = +/-0 +i*inf
             atan(+/-0 -i) = +/-0 -i*inf */
          mpfr_set_zero (mpc_realref (rop), s_re ? -1 : +1);
          mpfr_set_inf  (mpc_imagref (rop), s_im ? -1 : +1);
        }
      else
        {
          /* atan(+0+iy) = +pi/2 +i*atanh(1/y), if |y| > 1
             atan(-0+iy) = -pi/2 +i*atanh(1/y), if |y| > 1 */
          mpfr_rnd_t rnd_im, rnd_away;
          mpfr_t y;
          mpfr_prec_t p, p_im;
          int ok;

          rnd_im = MPC_RND_IM (rnd);
          mpfr_init (y);
          p_im = mpfr_get_prec (mpc_imagref (rop));
          p = p_im;

          /* a = o(1/y)      with error(a) < 1 ulp(a)
             b = o(atanh(a)) with error(b) < (1+2^{1+Exp(a)-Exp(b)}) ulp(b)

             As |atanh (1/y)| > |1/y| we have Exp(a)-Exp(b) <=0 so, at most,
             2 bits of precision are lost.

             We round atanh(1/y) away from 0.
          */
          do
            {
              p += mpc_ceil_log2 (p) + 2;
              mpfr_set_prec (y, p);
              rnd_away = s_im == 0 ? GMP_RNDU : GMP_RNDD;
              inex_im = mpfr_ui_div (y, 1, mpc_imagref (op), rnd_away);
              /* FIXME: should we consider the case with unreasonably huge
                 precision prec(y)>3*exp_min, where atanh(1/Im(op)) could be
                 representable while 1/Im(op) underflows ?
                 This corresponds to |y| = 0.5*2^emin, in which case the
                 result may be wrong. */

              /* atanh cannot underflow: |atanh(x)| > |x| for |x| < 1 */
              inex_im |= mpfr_atanh (y, y, rnd_away);

              ok = inex_im == 0
                || mpfr_can_round (y, p - 2, rnd_away, GMP_RNDZ,
                                   p_im + (rnd_im == GMP_RNDN));
            } while (ok == 0);

          inex_re = set_pi_over_2 (mpc_realref (rop), -s_re, MPC_RND_RE (rnd));
          inex_im = mpfr_set (mpc_imagref (rop), y, rnd_im);
          mpfr_clear (y);
        }
      return MPC_INEX (inex_re, inex_im);
    }

  /* regular number argument */
  {
    mpfr_t a, b, x, y;
    mpfr_prec_t prec, p;
    mpfr_exp_t err, expo;
    int ok = 0;
    mpfr_t minus_op_re;
    mpfr_exp_t op_re_exp, op_im_exp;
    mpfr_rnd_t rnd1, rnd2;

    mpfr_inits2 (MPFR_PREC_MIN, a, b, x, y, (mpfr_ptr) 0);

    /* real part: Re(arctan(x+i*y)) = [arctan2(x,1-y) - arctan2(-x,1+y)]/2 */
    minus_op_re[0] = mpc_realref (op)[0];
    MPFR_CHANGE_SIGN (minus_op_re);
    op_re_exp = mpfr_get_exp (mpc_realref (op));
    op_im_exp = mpfr_get_exp (mpc_imagref (op));

    prec = mpfr_get_prec (mpc_realref (rop)); /* result precision */

    /* a = o(1-y)         error(a) < 1 ulp(a)
       b = o(atan2(x,a))  error(b) < [1+2^{3+Exp(x)-Exp(a)-Exp(b)}] ulp(b)
                                     = kb ulp(b)
       c = o(1+y)         error(c) < 1 ulp(c)
       d = o(atan2(-x,c)) error(d) < [1+2^{3+Exp(x)-Exp(c)-Exp(d)}] ulp(d)
                                     = kd ulp(d)
       e = o(b - d)       error(e) < [1 + kb*2^{Exp(b}-Exp(e)}
                                        + kd*2^{Exp(d)-Exp(e)}] ulp(e)
                          error(e) < [1 + 2^{4+Exp(x)-Exp(a)-Exp(e)}
                                        + 2^{4+Exp(x)-Exp(c)-Exp(e)}] ulp(e)
                          because |atan(u)| < |u|
                                   < [1 + 2^{5+Exp(x)-min(Exp(a),Exp(c))
                                             -Exp(e)}] ulp(e)
       f = e/2            exact
    */

    /* p: working precision */
    p = (op_im_exp > 0 || prec > SAFE_ABS (mpfr_prec_t, op_im_exp)) ? prec
      : (prec - op_im_exp);
    rnd1 = mpfr_sgn (mpc_realref (op)) > 0 ? GMP_RNDD : GMP_RNDU;
    rnd2 = mpfr_sgn (mpc_realref (op)) < 0 ? GMP_RNDU : GMP_RNDD;

    do
      {
        p += mpc_ceil_log2 (p) + 2;
        mpfr_set_prec (a, p);
        mpfr_set_prec (b, p);
        mpfr_set_prec (x, p);

        /* x = upper bound for atan (x/(1-y)). Since atan is increasing, we
           need an upper bound on x/(1-y), i.e., a lower bound on 1-y for
           x positive, and an upper bound on 1-y for x negative */
        mpfr_ui_sub (a, 1, mpc_imagref (op), rnd1);
        if (mpfr_sgn (a) == 0) /* y is near 1, thus 1+y is near 2, and
                                  expo will be 1 or 2 below */
          {
            MPC_ASSERT (mpfr_cmp_ui (mpc_imagref(op), 1) == 0);
               /* check for intermediate underflow */
            err = 2; /* ensures err will be expo below */
          }
        else
          err = mpfr_get_exp (a); /* err = Exp(a) with the notations above */
        mpfr_atan2 (x, mpc_realref (op), a, GMP_RNDU);

        /* b = lower bound for atan (-x/(1+y)): for x negative, we need a
           lower bound on -x/(1+y), i.e., an upper bound on 1+y */
        mpfr_add_ui (a, mpc_imagref(op), 1, rnd2);
        /* if a is exactly zero, i.e., Im(op) = -1, then the error on a is 0,
           and we can simply ignore the terms involving Exp(a) in the error */
        if (mpfr_sgn (a) == 0)
          {
            MPC_ASSERT (mpfr_cmp_si (mpc_imagref(op), -1) == 0);
               /* check for intermediate underflow */
            expo = err; /* will leave err unchanged below */
          }
        else
          expo = mpfr_get_exp (a); /* expo = Exp(c) with the notations above */
        mpfr_atan2 (b, minus_op_re, a, GMP_RNDD);

        err = err < expo ? err : expo; /* err = min(Exp(a),Exp(c)) */
        mpfr_sub (x, x, b, GMP_RNDU);

        err = 5 + op_re_exp - err - mpfr_get_exp (x);
        /* error is bounded by [1 + 2^err] ulp(e) */
        err = err < 0 ? 1 : err + 1;

        mpfr_div_2ui (x, x, 1, GMP_RNDU);

        /* Note: using RND2=RNDD guarantees that if x is exactly representable
           on prec + ... bits, mpfr_can_round will return 0 */
        ok = mpfr_can_round (x, p - err, GMP_RNDU, GMP_RNDD,
                             prec + (MPC_RND_RE (rnd) == GMP_RNDN));
      } while (ok == 0);

    /* Imaginary part
       Im(atan(x+I*y)) = 1/4 * [log(x^2+(1+y)^2) - log (x^2 +(1-y)^2)] */
    prec = mpfr_get_prec (mpc_imagref (rop)); /* result precision */

    /* a = o(1+y)    error(a) < 1 ulp(a)
       b = o(a^2)    error(b) < 5 ulp(b)
       c = o(x^2)    error(c) < 1 ulp(c)
       d = o(b+c)    error(d) < 7 ulp(d)
       e = o(log(d)) error(e) < [1 + 7*2^{2-Exp(e)}] ulp(e) = ke ulp(e)
       f = o(1-y)    error(f) < 1 ulp(f)
       g = o(f^2)    error(g) < 5 ulp(g)
       h = o(c+f)    error(h) < 7 ulp(h)
       i = o(log(h)) error(i) < [1 + 7*2^{2-Exp(i)}] ulp(i) = ki ulp(i)
       j = o(e-i)    error(j) < [1 + ke*2^{Exp(e)-Exp(j)}
                                   + ki*2^{Exp(i)-Exp(j)}] ulp(j)
                     error(j) < [1 + 2^{Exp(e)-Exp(j)} + 2^{Exp(i)-Exp(j)}
                                   + 7*2^{3-Exp(j)}] ulp(j)
                              < [1 + 2^{max(Exp(e),Exp(i))-Exp(j)+1}
                                   + 7*2^{3-Exp(j)}] ulp(j)
       k = j/4       exact
    */
    err = 2;
    p = prec; /* working precision */

    do
      {
        p += mpc_ceil_log2 (p) + err;
        mpfr_set_prec (a, p);
        mpfr_set_prec (b, p);
        mpfr_set_prec (y, p);

        /* a = upper bound for log(x^2 + (1+y)^2) */
        ROUND_AWAY (mpfr_add_ui (a, mpc_imagref (op), 1, MPFR_RNDA), a);
        mpfr_sqr (a, a, GMP_RNDU);
        mpfr_sqr (y, mpc_realref (op), GMP_RNDU);
        mpfr_add (a, a, y, GMP_RNDU);
        mpfr_log (a, a, GMP_RNDU);

        /* b = lower bound for log(x^2 + (1-y)^2) */
        mpfr_ui_sub (b, 1, mpc_imagref (op), GMP_RNDZ); /* round to zero */
        mpfr_sqr (b, b, GMP_RNDZ);
        /* we could write mpfr_sqr (y, mpc_realref (op), GMP_RNDZ) but it is
           more efficient to reuse the value of y (x^2) above and subtract
           one ulp */
        mpfr_nextbelow (y);
        mpfr_add (b, b, y, GMP_RNDZ);
        mpfr_log (b, b, GMP_RNDZ);

        mpfr_sub (y, a, b, GMP_RNDU);

        if (mpfr_zero_p (y))
           /* FIXME: happens when x and y have very different magnitudes;
              could be handled more efficiently                           */
          ok = 0;
        else
          {
            expo = MPC_MAX (mpfr_get_exp (a), mpfr_get_exp (b));
            expo = expo - mpfr_get_exp (y) + 1;
            err = 3 - mpfr_get_exp (y);
            /* error(j) <= [1 + 2^expo + 7*2^err] ulp(j) */
            if (expo <= err) /* error(j) <= [1 + 2^{err+1}] ulp(j) */
              err = (err < 0) ? 1 : err + 2;
            else
              err = (expo < 0) ? 1 : expo + 2;

            mpfr_div_2ui (y, y, 2, GMP_RNDN);
            MPC_ASSERT (!mpfr_zero_p (y));
               /* FIXME: underflow. Since the main term of the Taylor series
                  in y=0 is 1/(x^2+1) * y, this means that y is very small
                  and/or x very large; but then the mpfr_zero_p (y) above
                  should be true. This needs a proof, or better yet,
                  special code.                                              */

            ok = mpfr_can_round (y, p - err, GMP_RNDU, GMP_RNDD,
                                 prec + (MPC_RND_IM (rnd) == GMP_RNDN));
          }
      } while (ok == 0);

    inex = mpc_set_fr_fr (rop, x, y, rnd);

    mpfr_clears (a, b, x, y, (mpfr_ptr) 0);
    return inex;
  }
}
