/* mpc_norm -- Square of the norm of a complex number.

Copyright (C) 2002, 2005, 2008, 2009, 2010, 2011 INRIA

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

/* a <- norm(b) = b * conj(b)
   (the rounding mode is mpfr_rnd_t here since we return an mpfr number) */
int
mpc_norm (mpfr_ptr a, mpc_srcptr b, mpfr_rnd_t rnd)
{
   int inexact;
   int saved_underflow, saved_overflow;

   /* handling of special values; consistent with abs in that
      norm = abs^2; so norm (+-inf, xxx) = norm (xxx, +-inf) = +inf */
   if (!mpc_fin_p (b))
         return mpc_abs (a, b, rnd);
   else if (mpfr_zero_p (mpc_realref (b))) {
      if (mpfr_zero_p (mpc_imagref (b)))
         return mpfr_set_ui (a, 0, rnd); /* +0 */
      else
         return mpfr_sqr (a, mpc_imagref (b), rnd);
   }
   else if (mpfr_zero_p (mpc_imagref (b)))
     return mpfr_sqr (a, mpc_realref (b), rnd); /* Re(b) <> 0 */

   else /* everything finite and non-zero */ {
      mpfr_t u, v, res;
      mpfr_prec_t prec, prec_u, prec_v;
      int loops;
      const int max_loops = 2;
         /* switch to exact squarings when loops==max_loops */

      prec = mpfr_get_prec (a);

      mpfr_init (u);
      mpfr_init (v);
      mpfr_init (res);

      /* save the underflow or overflow flags from MPFR */
      saved_underflow = mpfr_underflow_p ();
      saved_overflow = mpfr_overflow_p ();

      loops = 0;
      mpfr_clear_underflow ();
      mpfr_clear_overflow ();
      do {
         loops++;
         prec += mpc_ceil_log2 (prec) + 3;
         if (loops >= max_loops) {
            prec_u = 2 * MPC_PREC_RE (b);
            prec_v = 2 * MPC_PREC_IM (b);
         }
         else {
            prec_u = MPC_MIN (prec, 2 * MPC_PREC_RE (b));
            prec_v = MPC_MIN (prec, 2 * MPC_PREC_IM (b));
         }

         mpfr_set_prec (u, prec_u);
         mpfr_set_prec (v, prec_v);

         inexact  = mpfr_sqr (u, mpc_realref(b), GMP_RNDD); /* err <= 1 ulp in prec */
         inexact |= mpfr_sqr (v, mpc_imagref(b), GMP_RNDD); /* err <= 1 ulp in prec */

         /* If loops = max_loops, inexact should be 0 here, except in case
               of underflow or overflow.
            If loops < max_loops and inexact is zero, we can exit the
            while-loop since it only remains to add u and v into a. */
         if (inexact) {
             mpfr_set_prec (res, prec);
             mpfr_add (res, u, v, GMP_RNDD); /* err <= 3 ulp in prec */
         }

      } while (loops < max_loops && inexact != 0
               && !mpfr_can_round (res, prec - 2, GMP_RNDD, GMP_RNDU,
                                   mpfr_get_prec (a) + (rnd == GMP_RNDN)));

      if (!inexact)
         /* squarings were exact, neither underflow nor overflow */
         inexact = mpfr_add (a, u, v, rnd);
      /* if there was an overflow in Re(b)^2 or Im(b)^2 or their sum,
         since the norm is larger, there is an overflow for the norm */
      else if (mpfr_overflow_p ()) {
         /* replace by "correctly rounded overflow" */
         mpfr_set_ui (a, 1ul, GMP_RNDN);
         inexact = mpfr_mul_2ui (a, a, mpfr_get_emax (), rnd);
      }
      else if (mpfr_underflow_p ()) {
         /* necessarily one of the squarings did underflow (otherwise their
            sum could not underflow), thus one of u, v is zero. */
         mpfr_exp_t emin = mpfr_get_emin ();

         /* Now either both u and v are zero, or u is zero and v exact,
            or v is zero and u exact.
            In the latter case, Im(b)^2 < 2^(emin-1).
            If ulp(u) >= 2^(emin+1) and norm(b) is not exactly
            representable at the target precision, then rounding u+Im(b)^2
            is equivalent to rounding u+2^(emin-1).
            For instance, if exp(u)>0 and the target precision is smaller
            than about |emin|, the norm is not representable. To make the
            scaling in the "else" case work without underflow, we test
            whether exp(u) is larger than a small negative number instead.
            The second case is handled analogously.                        */
         if (!mpfr_zero_p (u)
             && mpfr_get_exp (u) - 2 * (mpfr_exp_t) prec_u > emin
             && mpfr_get_exp (u) > -10) {
               mpfr_set_prec (v, MPFR_PREC_MIN);
               mpfr_set_ui_2exp (v, 1, emin - 1, GMP_RNDZ);
               inexact = mpfr_add (a, u, v, rnd);
         }
         else if (!mpfr_zero_p (v)
             && mpfr_get_exp (v) - 2 * (mpfr_exp_t) prec_v > emin
             && mpfr_get_exp (v) > -10) {
               mpfr_set_prec (u, MPFR_PREC_MIN);
               mpfr_set_ui_2exp (u, 1, emin - 1, GMP_RNDZ);
               inexact = mpfr_add (a, u, v, rnd);
         }
         else {
            unsigned long int scale, exp_re, exp_im;
            int inex_underflow;

            /* scale the input to an average exponent close to 0 */
            exp_re = (unsigned long int) (-mpfr_get_exp (mpc_realref (b)));
            exp_im = (unsigned long int) (-mpfr_get_exp (mpc_imagref (b)));
            scale = exp_re / 2 + exp_im / 2 + (exp_re % 2 + exp_im % 2) / 2;
               /* (exp_re + exp_im) / 2, computed in a way avoiding
                  integer overflow                                  */
            if (mpfr_zero_p (u)) {
               /* recompute the scaled value exactly */
               mpfr_mul_2ui (u, mpc_realref (b), scale, GMP_RNDN);
               mpfr_sqr (u, u, GMP_RNDN);
            }
            else /* just scale */
               mpfr_mul_2ui (u, u, 2*scale, GMP_RNDN);
            if (mpfr_zero_p (v)) {
               mpfr_mul_2ui (v, mpc_imagref (b), scale, GMP_RNDN);
               mpfr_sqr (v, v, GMP_RNDN);
            }
            else
               mpfr_mul_2ui (v, v, 2*scale, GMP_RNDN);

            inexact = mpfr_add (a, u, v, rnd);
            mpfr_clear_underflow ();
            inex_underflow = mpfr_div_2ui (a, a, 2*scale, rnd);
            if (mpfr_underflow_p ())
               inexact = inex_underflow;
         }
      }
      else /* no problems, ternary value due to mpfr_can_round trick */
         inexact = mpfr_set (a, res, rnd);

      /* restore underflow and overflow flags from MPFR */
      if (saved_underflow)
        mpfr_set_underflow ();
      if (saved_overflow)
        mpfr_set_overflow ();

      mpfr_clear (u);
      mpfr_clear (v);
      mpfr_clear (res);
   }

   return inexact;
}
