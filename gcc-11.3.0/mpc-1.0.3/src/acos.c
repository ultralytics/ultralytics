/* mpc_acos -- arccosine of a complex number.

Copyright (C) 2009, 2010, 2011, 2012 INRIA

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

#include <stdio.h>    /* for MPC_ASSERT */
#include "mpc-impl.h"

int
mpc_acos (mpc_ptr rop, mpc_srcptr op, mpc_rnd_t rnd)
{
  int inex_re, inex_im, inex;
  mpfr_prec_t p_re, p_im, p;
  mpc_t z1;
  mpfr_t pi_over_2;
  mpfr_exp_t e1, e2;
  mpfr_rnd_t rnd_im;
  mpc_rnd_t rnd1;

  inex_re = 0;
  inex_im = 0;

  /* special values */
  if (mpfr_nan_p (mpc_realref (op)) || mpfr_nan_p (mpc_imagref (op)))
    {
      if (mpfr_inf_p (mpc_realref (op)) || mpfr_inf_p (mpc_imagref (op)))
        {
          mpfr_set_inf (mpc_imagref (rop), mpfr_signbit (mpc_imagref (op)) ? +1 : -1);
          mpfr_set_nan (mpc_realref (rop));
        }
      else if (mpfr_zero_p (mpc_realref (op)))
        {
          inex_re = set_pi_over_2 (mpc_realref (rop), +1, MPC_RND_RE (rnd));
          mpfr_set_nan (mpc_imagref (rop));
        }
      else
        {
          mpfr_set_nan (mpc_realref (rop));
          mpfr_set_nan (mpc_imagref (rop));
        }

      return MPC_INEX (inex_re, 0);
    }

  if (mpfr_inf_p (mpc_realref (op)) || mpfr_inf_p (mpc_imagref (op)))
    {
      if (mpfr_inf_p (mpc_realref (op)))
        {
          if (mpfr_inf_p (mpc_imagref (op)))
            {
              if (mpfr_sgn (mpc_realref (op)) > 0)
                {
                  inex_re =
                    set_pi_over_2 (mpc_realref (rop), +1, MPC_RND_RE (rnd));
                  mpfr_div_2ui (mpc_realref (rop), mpc_realref (rop), 1, GMP_RNDN);
                }
              else
                {

                  /* the real part of the result is 3*pi/4
                     a = o(pi)  error(a) < 1 ulp(a)
                     b = o(3*a) error(b) < 2 ulp(b)
                     c = b/4    exact
                     thus 1 bit is lost */
                  mpfr_t x;
                  mpfr_prec_t prec;
                  int ok;
                  mpfr_init (x);
                  prec = mpfr_get_prec (mpc_realref (rop));
                  p = prec;

                  do
                    {
                      p += mpc_ceil_log2 (p);
                      mpfr_set_prec (x, p);
                      mpfr_const_pi (x, GMP_RNDD);
                      mpfr_mul_ui (x, x, 3, GMP_RNDD);
                      ok =
                        mpfr_can_round (x, p - 1, GMP_RNDD, MPC_RND_RE (rnd),
                                        prec+(MPC_RND_RE (rnd) == GMP_RNDN));

                    } while (ok == 0);
                  inex_re =
                    mpfr_div_2ui (mpc_realref (rop), x, 2, MPC_RND_RE (rnd));
                  mpfr_clear (x);
                }
            }
          else
            {
              if (mpfr_sgn (mpc_realref (op)) > 0)
                mpfr_set_ui (mpc_realref (rop), 0, GMP_RNDN);
              else
                inex_re = mpfr_const_pi (mpc_realref (rop), MPC_RND_RE (rnd));
            }
        }
      else
        inex_re = set_pi_over_2 (mpc_realref (rop), +1, MPC_RND_RE (rnd));

      mpfr_set_inf (mpc_imagref (rop), mpfr_signbit (mpc_imagref (op)) ? +1 : -1);

      return MPC_INEX (inex_re, 0);
    }

  /* pure real argument */
  if (mpfr_zero_p (mpc_imagref (op)))
    {
      int s_im;
      s_im = mpfr_signbit (mpc_imagref (op));

      if (mpfr_cmp_ui (mpc_realref (op), 1) > 0)
        {
          if (s_im)
            inex_im = mpfr_acosh (mpc_imagref (rop), mpc_realref (op),
                                  MPC_RND_IM (rnd));
          else
            inex_im = -mpfr_acosh (mpc_imagref (rop), mpc_realref (op),
                                   INV_RND (MPC_RND_IM (rnd)));

          mpfr_set_ui (mpc_realref (rop), 0, GMP_RNDN);
        }
      else if (mpfr_cmp_si (mpc_realref (op), -1) < 0)
        {
          mpfr_t minus_op_re;
          minus_op_re[0] = mpc_realref (op)[0];
          MPFR_CHANGE_SIGN (minus_op_re);

          if (s_im)
            inex_im = mpfr_acosh (mpc_imagref (rop), minus_op_re,
                                  MPC_RND_IM (rnd));
          else
            inex_im = -mpfr_acosh (mpc_imagref (rop), minus_op_re,
                                   INV_RND (MPC_RND_IM (rnd)));
          inex_re = mpfr_const_pi (mpc_realref (rop), MPC_RND_RE (rnd));
        }
      else
        {
          inex_re = mpfr_acos (mpc_realref (rop), mpc_realref (op), MPC_RND_RE (rnd));
          mpfr_set_ui (mpc_imagref (rop), 0, MPC_RND_IM (rnd));
        }

      if (!s_im)
        mpc_conj (rop, rop, MPC_RNDNN);

      return MPC_INEX (inex_re, inex_im);
    }

  /* pure imaginary argument */
  if (mpfr_zero_p (mpc_realref (op)))
    {
      inex_re = set_pi_over_2 (mpc_realref (rop), +1, MPC_RND_RE (rnd));
      inex_im = -mpfr_asinh (mpc_imagref (rop), mpc_imagref (op),
                             INV_RND (MPC_RND_IM (rnd)));
      mpc_conj (rop,rop, MPC_RNDNN);

      return MPC_INEX (inex_re, inex_im);
    }

  /* regular complex argument: acos(z) = Pi/2 - asin(z) */
  p_re = mpfr_get_prec (mpc_realref(rop));
  p_im = mpfr_get_prec (mpc_imagref(rop));
  p = p_re;
  mpc_init3 (z1, p, p_im); /* we round directly the imaginary part to p_im,
                              with rounding mode opposite to rnd_im */
  rnd_im = MPC_RND_IM(rnd);
  /* the imaginary part of asin(z) has the same sign as Im(z), thus if
     Im(z) > 0 and rnd_im = RNDZ, we want to round the Im(asin(z)) to -Inf
     so that -Im(asin(z)) is rounded to zero */
  if (rnd_im == GMP_RNDZ)
    rnd_im = mpfr_sgn (mpc_imagref(op)) > 0 ? GMP_RNDD : GMP_RNDU;
  else
    rnd_im = rnd_im == GMP_RNDU ? GMP_RNDD
      : rnd_im == GMP_RNDD ? GMP_RNDU
      : rnd_im; /* both RNDZ and RNDA map to themselves for -asin(z) */
  rnd1 = MPC_RND (GMP_RNDN, rnd_im);
  mpfr_init2 (pi_over_2, p);
  for (;;)
    {
      p += mpc_ceil_log2 (p) + 3;

      mpfr_set_prec (mpc_realref(z1), p);
      mpfr_set_prec (pi_over_2, p);

      set_pi_over_2 (pi_over_2, +1, GMP_RNDN);
      e1 = 1; /* Exp(pi_over_2) */
      inex = mpc_asin (z1, op, rnd1); /* asin(z) */
      MPC_ASSERT (mpfr_sgn (mpc_imagref(z1)) * mpfr_sgn (mpc_imagref(op)) > 0);
      inex_im = MPC_INEX_IM(inex); /* inex_im is in {-1, 0, 1} */
      e2 = mpfr_get_exp (mpc_realref(z1));
      mpfr_sub (mpc_realref(z1), pi_over_2, mpc_realref(z1), GMP_RNDN);
      if (!mpfr_zero_p (mpc_realref(z1)))
        {
          /* the error on x=Re(z1) is bounded by 1/2 ulp(x) + 2^(e1-p-1) +
             2^(e2-p-1) */
          e1 = e1 >= e2 ? e1 + 1 : e2 + 1;
          /* the error on x is bounded by 1/2 ulp(x) + 2^(e1-p-1) */
          e1 -= mpfr_get_exp (mpc_realref(z1));
          /* the error on x is bounded by 1/2 ulp(x) [1 + 2^e1] */
          e1 = e1 <= 0 ? 0 : e1;
          /* the error on x is bounded by 2^e1 * ulp(x) */
          mpfr_neg (mpc_imagref(z1), mpc_imagref(z1), GMP_RNDN); /* exact */
          inex_im = -inex_im;
          if (mpfr_can_round (mpc_realref(z1), p - e1, GMP_RNDN, GMP_RNDZ,
                              p_re + (MPC_RND_RE(rnd) == GMP_RNDN)))
            break;
        }
    }
  inex = mpc_set (rop, z1, rnd);
  inex_re = MPC_INEX_RE(inex);
  mpc_clear (z1);
  mpfr_clear (pi_over_2);

  return MPC_INEX(inex_re, inex_im);
}
