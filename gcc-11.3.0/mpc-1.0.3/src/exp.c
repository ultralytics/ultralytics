/* mpc_exp -- exponential of a complex number.

Copyright (C) 2002, 2009, 2010, 2011 INRIA

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
mpc_exp (mpc_ptr rop, mpc_srcptr op, mpc_rnd_t rnd)
{
  mpfr_t x, y, z;
  mpfr_prec_t prec;
  int ok = 0;
  int inex_re, inex_im;
  int saved_underflow, saved_overflow;

  /* special values */
  if (mpfr_nan_p (mpc_realref (op)) || mpfr_nan_p (mpc_imagref (op)))
    /* NaNs
       exp(nan +i*y) = nan -i*0   if y = -0,
                       nan +i*0   if y = +0,
                       nan +i*nan otherwise
       exp(x+i*nan) =   +/-0 +/-i*0 if x=-inf,
                      +/-inf +i*nan if x=+inf,
                         nan +i*nan otherwise */
    {
      if (mpfr_zero_p (mpc_imagref (op)))
        return mpc_set (rop, op, MPC_RNDNN);

      if (mpfr_inf_p (mpc_realref (op)))
        {
          if (mpfr_signbit (mpc_realref (op)))
            return mpc_set_ui_ui (rop, 0, 0, MPC_RNDNN);
          else
            {
              mpfr_set_inf (mpc_realref (rop), +1);
              mpfr_set_nan (mpc_imagref (rop));
              return MPC_INEX(0, 0); /* Inf/NaN are exact */
            }
        }
      mpfr_set_nan (mpc_realref (rop));
      mpfr_set_nan (mpc_imagref (rop));
      return MPC_INEX(0, 0); /* NaN is exact */
    }


  if (mpfr_zero_p (mpc_imagref(op)))
    /* special case when the input is real
       exp(x-i*0) = exp(x) -i*0, even if x is NaN
       exp(x+i*0) = exp(x) +i*0, even if x is NaN */
    {
      inex_re = mpfr_exp (mpc_realref(rop), mpc_realref(op), MPC_RND_RE(rnd));
      inex_im = mpfr_set (mpc_imagref(rop), mpc_imagref(op), MPC_RND_IM(rnd));
      return MPC_INEX(inex_re, inex_im);
    }

  if (mpfr_zero_p (mpc_realref (op)))
    /* special case when the input is imaginary  */
    {
      inex_re = mpfr_cos (mpc_realref (rop), mpc_imagref (op), MPC_RND_RE(rnd));
      inex_im = mpfr_sin (mpc_imagref (rop), mpc_imagref (op), MPC_RND_IM(rnd));
      return MPC_INEX(inex_re, inex_im);
    }


  if (mpfr_inf_p (mpc_realref (op)))
    /* real part is an infinity,
       exp(-inf +i*y) = 0*(cos y +i*sin y)
       exp(+inf +i*y) = +/-inf +i*nan         if y = +/-inf
                        +inf*(cos y +i*sin y) if 0 < |y| < inf */
    {
      mpfr_t n;

      mpfr_init2 (n, 2);
      if (mpfr_signbit (mpc_realref (op)))
        mpfr_set_ui (n, 0, GMP_RNDN);
      else
        mpfr_set_inf (n, +1);

      if (mpfr_inf_p (mpc_imagref (op)))
        {
          inex_re = mpfr_set (mpc_realref (rop), n, GMP_RNDN);
          if (mpfr_signbit (mpc_realref (op)))
            inex_im = mpfr_set (mpc_imagref (rop), n, GMP_RNDN);
          else
            {
              mpfr_set_nan (mpc_imagref (rop));
              inex_im = 0; /* NaN is exact */
            }
        }
      else
        {
          mpfr_t c, s;
          mpfr_init2 (c, 2);
          mpfr_init2 (s, 2);

          mpfr_sin_cos (s, c, mpc_imagref (op), GMP_RNDN);
          inex_re = mpfr_copysign (mpc_realref (rop), n, c, GMP_RNDN);
          inex_im = mpfr_copysign (mpc_imagref (rop), n, s, GMP_RNDN);

          mpfr_clear (s);
          mpfr_clear (c);
        }

      mpfr_clear (n);
      return MPC_INEX(inex_re, inex_im);
    }

  if (mpfr_inf_p (mpc_imagref (op)))
    /* real part is finite non-zero number, imaginary part is an infinity */
    {
      mpfr_set_nan (mpc_realref (rop));
      mpfr_set_nan (mpc_imagref (rop));
      return MPC_INEX(0, 0); /* NaN is exact */
    }


  /* from now on, both parts of op are regular numbers */

  prec = MPC_MAX_PREC(rop)
         + MPC_MAX (MPC_MAX (-mpfr_get_exp (mpc_realref (op)), 0),
                   -mpfr_get_exp (mpc_imagref (op)));
    /* When op is close to 0, then exp is close to 1+Re(op), while
       cos is close to 1-Im(op); to decide on the ternary value of exp*cos,
       we need a high enough precision so that none of exp or cos is
       computed as 1. */
  mpfr_init2 (x, 2);
  mpfr_init2 (y, 2);
  mpfr_init2 (z, 2);

  /* save the underflow or overflow flags from MPFR */
  saved_underflow = mpfr_underflow_p ();
  saved_overflow = mpfr_overflow_p ();

  do
    {
      prec += mpc_ceil_log2 (prec) + 5;

      mpfr_set_prec (x, prec);
      mpfr_set_prec (y, prec);
      mpfr_set_prec (z, prec);

      /* FIXME: x may overflow so x.y does overflow too, while Re(exp(op))
         could be represented in the precision of rop. */
      mpfr_clear_overflow ();
      mpfr_clear_underflow ();
      mpfr_exp (x, mpc_realref(op), GMP_RNDN); /* error <= 0.5ulp */
      mpfr_sin_cos (z, y, mpc_imagref(op), GMP_RNDN); /* errors <= 0.5ulp */
      mpfr_mul (y, y, x, GMP_RNDN); /* error <= 2ulp */
      ok = mpfr_overflow_p () || mpfr_zero_p (x)
        || mpfr_can_round (y, prec - 2, GMP_RNDN, GMP_RNDZ,
                       MPC_PREC_RE(rop) + (MPC_RND_RE(rnd) == GMP_RNDN));
      if (ok) /* compute imaginary part */
        {
          mpfr_mul (z, z, x, GMP_RNDN);
          ok = mpfr_overflow_p () || mpfr_zero_p (x)
            || mpfr_can_round (z, prec - 2, GMP_RNDN, GMP_RNDZ,
                       MPC_PREC_IM(rop) + (MPC_RND_IM(rnd) == GMP_RNDN));
        }
    }
  while (ok == 0);

  inex_re = mpfr_set (mpc_realref(rop), y, MPC_RND_RE(rnd));
  inex_im = mpfr_set (mpc_imagref(rop), z, MPC_RND_IM(rnd));
  if (mpfr_overflow_p ()) {
    /* overflow in real exponential, inex is sign of infinite result */
    inex_re = mpfr_sgn (y);
    inex_im = mpfr_sgn (z);
  }
  else if (mpfr_underflow_p ()) {
    /* underflow in real exponential, inex is opposite of sign of 0 result */
    inex_re = (mpfr_signbit (y) ? +1 : -1);
    inex_im = (mpfr_signbit (z) ? +1 : -1);
  }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);

  /* restore underflow and overflow flags from MPFR */
  if (saved_underflow)
    mpfr_set_underflow ();
  if (saved_overflow)
    mpfr_set_overflow ();

  return MPC_INEX(inex_re, inex_im);
}
