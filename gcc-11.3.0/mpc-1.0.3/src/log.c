/* mpc_log -- Take the logarithm of a complex number.

Copyright (C) 2008, 2009, 2010, 2011, 2012 INRIA

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

#include <stdio.h> /* for MPC_ASSERT */
#include "mpc-impl.h"

int
mpc_log (mpc_ptr rop, mpc_srcptr op, mpc_rnd_t rnd){
   int ok, underflow = 0;
   mpfr_srcptr x, y;
   mpfr_t v, w;
   mpfr_prec_t prec;
   int loops;
   int re_cmp, im_cmp;
   int inex_re, inex_im;
   int err;
   mpfr_exp_t expw;
   int sgnw;

   /* special values: NaN and infinities */
   if (!mpc_fin_p (op)) {
      if (mpfr_nan_p (mpc_realref (op))) {
         if (mpfr_inf_p (mpc_imagref (op)))
            mpfr_set_inf (mpc_realref (rop), +1);
         else
            mpfr_set_nan (mpc_realref (rop));
         mpfr_set_nan (mpc_imagref (rop));
         inex_im = 0; /* Inf/NaN is exact */
      }
      else if (mpfr_nan_p (mpc_imagref (op))) {
         if (mpfr_inf_p (mpc_realref (op)))
            mpfr_set_inf (mpc_realref (rop), +1);
         else
            mpfr_set_nan (mpc_realref (rop));
         mpfr_set_nan (mpc_imagref (rop));
         inex_im = 0; /* Inf/NaN is exact */
      }
      else /* We have an infinity in at least one part. */ {
         inex_im = mpfr_atan2 (mpc_imagref (rop), mpc_imagref (op), mpc_realref (op),
                               MPC_RND_IM (rnd));
         mpfr_set_inf (mpc_realref (rop), +1);
      }
      return MPC_INEX(0, inex_im);
   }

   /* special cases: real and purely imaginary numbers */
   re_cmp = mpfr_cmp_ui (mpc_realref (op), 0);
   im_cmp = mpfr_cmp_ui (mpc_imagref (op), 0);
   if (im_cmp == 0) {
      if (re_cmp == 0) {
         inex_im = mpfr_atan2 (mpc_imagref (rop), mpc_imagref (op), mpc_realref (op),
                               MPC_RND_IM (rnd));
         mpfr_set_inf (mpc_realref (rop), -1);
         inex_re = 0; /* -Inf is exact */
      }
      else if (re_cmp > 0) {
         inex_re = mpfr_log (mpc_realref (rop), mpc_realref (op), MPC_RND_RE (rnd));
         inex_im = mpfr_set (mpc_imagref (rop), mpc_imagref (op), MPC_RND_IM (rnd));
      }
      else {
         /* op = x + 0*y; let w = -x = |x| */
         int negative_zero;
         mpfr_rnd_t rnd_im;

         negative_zero = mpfr_signbit (mpc_imagref (op));
         if (negative_zero)
            rnd_im = INV_RND (MPC_RND_IM (rnd));
         else
            rnd_im = MPC_RND_IM (rnd);
         w [0] = *mpc_realref (op);
         MPFR_CHANGE_SIGN (w);
         inex_re = mpfr_log (mpc_realref (rop), w, MPC_RND_RE (rnd));
         inex_im = mpfr_const_pi (mpc_imagref (rop), rnd_im);
         if (negative_zero) {
            mpc_conj (rop, rop, MPC_RNDNN);
            inex_im = -inex_im;
         }
      }
      return MPC_INEX(inex_re, inex_im);
   }
   else if (re_cmp == 0) {
      if (im_cmp > 0) {
         inex_re = mpfr_log (mpc_realref (rop), mpc_imagref (op), MPC_RND_RE (rnd));
         inex_im = mpfr_const_pi (mpc_imagref (rop), MPC_RND_IM (rnd));
         /* division by 2 does not change the ternary flag */
         mpfr_div_2ui (mpc_imagref (rop), mpc_imagref (rop), 1, GMP_RNDN);
      }
      else {
         w [0] = *mpc_imagref (op);
         MPFR_CHANGE_SIGN (w);
         inex_re = mpfr_log (mpc_realref (rop), w, MPC_RND_RE (rnd));
         inex_im = mpfr_const_pi (mpc_imagref (rop), INV_RND (MPC_RND_IM (rnd)));
         /* division by 2 does not change the ternary flag */
         mpfr_div_2ui (mpc_imagref (rop), mpc_imagref (rop), 1, GMP_RNDN);
         mpfr_neg (mpc_imagref (rop), mpc_imagref (rop), GMP_RNDN);
         inex_im = -inex_im; /* negate the ternary flag */
      }
      return MPC_INEX(inex_re, inex_im);
   }

   prec = MPC_PREC_RE(rop);
   mpfr_init2 (w, 2);
   /* let op = x + iy; log = 1/2 log (x^2 + y^2) + i atan2 (y, x)   */
   /* loop for the real part: 1/2 log (x^2 + y^2), fast, but unsafe */
   /* implementation                                                */
   ok = 0;
   for (loops = 1; !ok && loops <= 2; loops++) {
      prec += mpc_ceil_log2 (prec) + 4;
      mpfr_set_prec (w, prec);

      mpc_abs (w, op, GMP_RNDN);
         /* error 0.5 ulp */
      if (mpfr_inf_p (w))
         /* intermediate overflow; the logarithm may be representable.
            Intermediate underflow is impossible.                      */
         break;

      mpfr_log (w, w, GMP_RNDN);
         /* generic error of log: (2^(- exp(w)) + 0.5) ulp */

      if (mpfr_zero_p (w))
         /* impossible to round, switch to second algorithm */
         break;

      err = MPC_MAX (-mpfr_get_exp (w), 0) + 1;
         /* number of lost digits */
      ok = mpfr_can_round (w, prec - err, GMP_RNDN, GMP_RNDZ,
         mpfr_get_prec (mpc_realref (rop)) + (MPC_RND_RE (rnd) == GMP_RNDN));
   }

   if (!ok) {
      prec = MPC_PREC_RE(rop);
      mpfr_init2 (v, 2);
      /* compute 1/2 log (x^2 + y^2) = log |x| + 1/2 * log (1 + (y/x)^2)
            if |x| >= |y|; otherwise, exchange x and y                   */
      if (mpfr_cmpabs (mpc_realref (op), mpc_imagref (op)) >= 0) {
         x = mpc_realref (op);
         y = mpc_imagref (op);
      }
      else {
         x = mpc_imagref (op);
         y = mpc_realref (op);
      }

      do {
         prec += mpc_ceil_log2 (prec) + 4;
         mpfr_set_prec (v, prec);
         mpfr_set_prec (w, prec);

         mpfr_div (v, y, x, GMP_RNDD); /* error 1 ulp */
         mpfr_sqr (v, v, GMP_RNDD);
            /* generic error of multiplication:
               1 + 2*1*(2+1*2^(1-prec)) <= 5.0625 since prec >= 6 */
         mpfr_log1p (v, v, GMP_RNDD);
            /* error 1 + 4*5.0625 = 21.25 , see algorithms.tex */
         mpfr_div_2ui (v, v, 1, GMP_RNDD);
            /* If the result is 0, then there has been an underflow somewhere. */

         mpfr_abs (w, x, GMP_RNDN); /* exact */
         mpfr_log (w, w, GMP_RNDN); /* error 0.5 ulp */
         expw = mpfr_get_exp (w);
         sgnw = mpfr_signbit (w);

         mpfr_add (w, w, v, GMP_RNDN);
         if (!sgnw) /* v is positive, so no cancellation;
                       error 22.25 ulp; error counts lost bits */
            err = 5;
         else
            err =   MPC_MAX (5 + mpfr_get_exp (v),
                  /* 21.25 ulp (v) rewritten in ulp (result, now in w) */
                           -1 + expw             - mpfr_get_exp (w)
                  /* 0.5 ulp (previous w), rewritten in ulp (result) */
                  ) + 2;

         /* handle one special case: |x|=1, and (y/x)^2 underflows;
            then 1/2*log(x^2+y^2) \approx 1/2*y^2 also underflows.  */
         if (   (mpfr_cmp_si (x, -1) == 0 || mpfr_cmp_ui (x, 1) == 0)
             && mpfr_zero_p (w))
            underflow = 1;

      } while (!underflow &&
               !mpfr_can_round (w, prec - err, GMP_RNDN, GMP_RNDZ,
               mpfr_get_prec (mpc_realref (rop)) + (MPC_RND_RE (rnd) == GMP_RNDN)));
      mpfr_clear (v);
   }

   /* imaginary part */
   inex_im = mpfr_atan2 (mpc_imagref (rop), mpc_imagref (op), mpc_realref (op),
                         MPC_RND_IM (rnd));

   /* set the real part; cannot be done before if rop==op */
   if (underflow)
      /* create underflow in result */
      inex_re = mpfr_set_ui_2exp (mpc_realref (rop), 1,
                                  mpfr_get_emin_min () - 2, MPC_RND_RE (rnd));
   else
      inex_re = mpfr_set (mpc_realref (rop), w, MPC_RND_RE (rnd));
   mpfr_clear (w);
   return MPC_INEX(inex_re, inex_im);
}
