/* mpc_tan -- tangent of a complex number.

Copyright (C) 2008, 2009, 2010, 2011 INRIA

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
#include <limits.h>
#include "mpc-impl.h"

int
mpc_tan (mpc_ptr rop, mpc_srcptr op, mpc_rnd_t rnd)
{
  mpc_t x, y;
  mpfr_prec_t prec;
  mpfr_exp_t err;
  int ok = 0;
  int inex;

  /* special values */
  if (!mpc_fin_p (op))
    {
      if (mpfr_nan_p (mpc_realref (op)))
        {
          if (mpfr_inf_p (mpc_imagref (op)))
            /* tan(NaN -i*Inf) = +/-0 -i */
            /* tan(NaN +i*Inf) = +/-0 +i */
            {
              /* exact unless 1 is not in exponent range */
              inex = mpc_set_si_si (rop, 0,
                                    (MPFR_SIGN (mpc_imagref (op)) < 0) ? -1 : +1,
                                    rnd);
            }
          else
            /* tan(NaN +i*y) = NaN +i*NaN, when y is finite */
            /* tan(NaN +i*NaN) = NaN +i*NaN */
            {
              mpfr_set_nan (mpc_realref (rop));
              mpfr_set_nan (mpc_imagref (rop));
              inex = MPC_INEX (0, 0); /* always exact */
            }
        }
      else if (mpfr_nan_p (mpc_imagref (op)))
        {
          if (mpfr_cmp_ui (mpc_realref (op), 0) == 0)
            /* tan(-0 +i*NaN) = -0 +i*NaN */
            /* tan(+0 +i*NaN) = +0 +i*NaN */
            {
              mpc_set (rop, op, rnd);
              inex = MPC_INEX (0, 0); /* always exact */
            }
          else
            /* tan(x +i*NaN) = NaN +i*NaN, when x != 0 */
            {
              mpfr_set_nan (mpc_realref (rop));
              mpfr_set_nan (mpc_imagref (rop));
              inex = MPC_INEX (0, 0); /* always exact */
            }
        }
      else if (mpfr_inf_p (mpc_realref (op)))
        {
          if (mpfr_inf_p (mpc_imagref (op)))
            /* tan(-Inf -i*Inf) = -/+0 -i */
            /* tan(-Inf +i*Inf) = -/+0 +i */
            /* tan(+Inf -i*Inf) = +/-0 -i */
            /* tan(+Inf +i*Inf) = +/-0 +i */
            {
              const int sign_re = mpfr_signbit (mpc_realref (op));
              int inex_im;

              mpfr_set_ui (mpc_realref (rop), 0, MPC_RND_RE (rnd));
              mpfr_setsign (mpc_realref (rop), mpc_realref (rop), sign_re, GMP_RNDN);

              /* exact, unless 1 is not in exponent range */
              inex_im = mpfr_set_si (mpc_imagref (rop),
                                     mpfr_signbit (mpc_imagref (op)) ? -1 : +1,
                                     MPC_RND_IM (rnd));
              inex = MPC_INEX (0, inex_im);
            }
          else
            /* tan(-Inf +i*y) = tan(+Inf +i*y) = NaN +i*NaN, when y is
               finite */
            {
              mpfr_set_nan (mpc_realref (rop));
              mpfr_set_nan (mpc_imagref (rop));
              inex = MPC_INEX (0, 0); /* always exact */
            }
        }
      else
        /* tan(x -i*Inf) = +0*sin(x)*cos(x) -i, when x is finite */
        /* tan(x +i*Inf) = +0*sin(x)*cos(x) +i, when x is finite */
        {
          mpfr_t c;
          mpfr_t s;
          int inex_im;

          mpfr_init (c);
          mpfr_init (s);

          mpfr_sin_cos (s, c, mpc_realref (op), GMP_RNDN);
          mpfr_set_ui (mpc_realref (rop), 0, MPC_RND_RE (rnd));
          mpfr_setsign (mpc_realref (rop), mpc_realref (rop),
                        mpfr_signbit (c) != mpfr_signbit (s), GMP_RNDN);
          /* exact, unless 1 is not in exponent range */
          inex_im = mpfr_set_si (mpc_imagref (rop),
                                 (mpfr_signbit (mpc_imagref (op)) ? -1 : +1),
                                 MPC_RND_IM (rnd));
          inex = MPC_INEX (0, inex_im);

          mpfr_clear (s);
          mpfr_clear (c);
        }

      return inex;
    }

  if (mpfr_zero_p (mpc_realref (op)))
    /* tan(-0 -i*y) = -0 +i*tanh(y), when y is finite. */
    /* tan(+0 +i*y) = +0 +i*tanh(y), when y is finite. */
    {
      int inex_im;

      mpfr_set (mpc_realref (rop), mpc_realref (op), MPC_RND_RE (rnd));
      inex_im = mpfr_tanh (mpc_imagref (rop), mpc_imagref (op), MPC_RND_IM (rnd));

      return MPC_INEX (0, inex_im);
    }

  if (mpfr_zero_p (mpc_imagref (op)))
    /* tan(x -i*0) = tan(x) -i*0, when x is finite. */
    /* tan(x +i*0) = tan(x) +i*0, when x is finite. */
    {
      int inex_re;

      inex_re = mpfr_tan (mpc_realref (rop), mpc_realref (op), MPC_RND_RE (rnd));
      mpfr_set (mpc_imagref (rop), mpc_imagref (op), MPC_RND_IM (rnd));

      return MPC_INEX (inex_re, 0);
    }

  /* ordinary (non-zero) numbers */

  /* tan(op) = sin(op) / cos(op).

     We use the following algorithm with rounding away from 0 for all
     operations, and working precision w:

     (1) x = A(sin(op))
     (2) y = A(cos(op))
     (3) z = A(x/y)

     the error on Im(z) is at most 81 ulp,
     the error on Re(z) is at most
     7 ulp if k < 2,
     8 ulp if k = 2,
     else 5+k ulp, where
     k = Exp(Re(x))+Exp(Re(y))-2min{Exp(Re(y)), Exp(Im(y))}-Exp(Re(x/y))
     see proof in algorithms.tex.
  */

  prec = MPC_MAX_PREC(rop);

  mpc_init2 (x, 2);
  mpc_init2 (y, 2);

  err = 7;

  do
    {
      mpfr_exp_t k, exr, eyr, eyi, ezr;

      ok = 0;

      /* FIXME: prevent addition overflow */
      prec += mpc_ceil_log2 (prec) + err;
      mpc_set_prec (x, prec);
      mpc_set_prec (y, prec);

      /* rounding away from zero: except in the cases x=0 or y=0 (processed
         above), sin x and cos y are never exact, so rounding away from 0 is
         rounding towards 0 and adding one ulp to the absolute value */
      mpc_sin_cos (x, y, op, MPC_RNDZZ, MPC_RNDZZ);
      MPFR_ADD_ONE_ULP (mpc_realref (x));
      MPFR_ADD_ONE_ULP (mpc_imagref (x));
      MPFR_ADD_ONE_ULP (mpc_realref (y));
      MPFR_ADD_ONE_ULP (mpc_imagref (y));
      MPC_ASSERT (mpfr_zero_p (mpc_realref (x)) == 0);

      if (   mpfr_inf_p (mpc_realref (x)) || mpfr_inf_p (mpc_imagref (x))
          || mpfr_inf_p (mpc_realref (y)) || mpfr_inf_p (mpc_imagref (y))) {
         /* If the real or imaginary part of x is infinite, it means that
            Im(op) was large, in which case the result is
            sign(tan(Re(op)))*0 + sign(Im(op))*I,
            where sign(tan(Re(op))) = sign(Re(x))*sign(Re(y)). */
          int inex_re, inex_im;
          mpfr_set_ui (mpc_realref (rop), 0, GMP_RNDN);
          if (mpfr_sgn (mpc_realref (x)) * mpfr_sgn (mpc_realref (y)) < 0)
            {
              mpfr_neg (mpc_realref (rop), mpc_realref (rop), GMP_RNDN);
              inex_re = 1;
            }
          else
            inex_re = -1; /* +0 is rounded down */
          if (mpfr_sgn (mpc_imagref (op)) > 0)
            {
              mpfr_set_ui (mpc_imagref (rop), 1, GMP_RNDN);
              inex_im = 1;
            }
          else
            {
              mpfr_set_si (mpc_imagref (rop), -1, GMP_RNDN);
              inex_im = -1;
            }
          inex = MPC_INEX(inex_re, inex_im);
          goto end;
        }

      exr = mpfr_get_exp (mpc_realref (x));
      eyr = mpfr_get_exp (mpc_realref (y));
      eyi = mpfr_get_exp (mpc_imagref (y));

      /* some parts of the quotient may be exact */
      inex = mpc_div (x, x, y, MPC_RNDZZ);
      /* OP is no pure real nor pure imaginary, so in theory the real and
         imaginary parts of its tangent cannot be null. However due to
         rouding errors this might happen. Consider for example
         tan(1+14*I) = 1.26e-10 + 1.00*I. For small precision sin(op) and
         cos(op) differ only by a factor I, thus after mpc_div x = I and
         its real part is zero. */
      if (mpfr_zero_p (mpc_realref (x)) || mpfr_zero_p (mpc_imagref (x)))
        {
          err = prec; /* double precision */
          continue;
        }
      if (MPC_INEX_RE (inex))
         MPFR_ADD_ONE_ULP (mpc_realref (x));
      if (MPC_INEX_IM (inex))
         MPFR_ADD_ONE_ULP (mpc_imagref (x));
      MPC_ASSERT (mpfr_zero_p (mpc_realref (x)) == 0);
      ezr = mpfr_get_exp (mpc_realref (x));

      /* FIXME: compute
         k = Exp(Re(x))+Exp(Re(y))-2min{Exp(Re(y)), Exp(Im(y))}-Exp(Re(x/y))
         avoiding overflow */
      k = exr - ezr + MPC_MAX(-eyr, eyr - 2 * eyi);
      err = k < 2 ? 7 : (k == 2 ? 8 : (5 + k));

      /* Can the real part be rounded? */
      ok = (!mpfr_number_p (mpc_realref (x)))
           || mpfr_can_round (mpc_realref(x), prec - err, GMP_RNDN, GMP_RNDZ,
                      MPC_PREC_RE(rop) + (MPC_RND_RE(rnd) == GMP_RNDN));

      if (ok)
        {
          /* Can the imaginary part be rounded? */
          ok = (!mpfr_number_p (mpc_imagref (x)))
               || mpfr_can_round (mpc_imagref(x), prec - 6, GMP_RNDN, GMP_RNDZ,
                      MPC_PREC_IM(rop) + (MPC_RND_IM(rnd) == GMP_RNDN));
        }
    }
  while (ok == 0);

  inex = mpc_set (rop, x, rnd);

 end:
  mpc_clear (x);
  mpc_clear (y);

  return inex;
}
