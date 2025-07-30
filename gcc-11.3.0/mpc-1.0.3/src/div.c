/* mpc_div -- Divide two complex numbers.

Copyright (C) 2002, 2003, 2004, 2005, 2008, 2009, 2010, 2011, 2012 INRIA

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

/* this routine deals with the case where w is zero */
static int
mpc_div_zero (mpc_ptr a, mpc_srcptr z, mpc_srcptr w, mpc_rnd_t rnd)
/* Assumes w==0, implementation according to C99 G.5.1.8 */
{
   int sign = MPFR_SIGNBIT (mpc_realref (w));
   mpfr_t infty;

   mpfr_init2 (infty, MPFR_PREC_MIN);
   mpfr_set_inf (infty, sign);
   mpfr_mul (mpc_realref (a), infty, mpc_realref (z), MPC_RND_RE (rnd));
   mpfr_mul (mpc_imagref (a), infty, mpc_imagref (z), MPC_RND_IM (rnd));
   mpfr_clear (infty);
   return MPC_INEX (0, 0); /* exact */
}

/* this routine deals with the case where z is infinite and w finite */
static int
mpc_div_inf_fin (mpc_ptr rop, mpc_srcptr z, mpc_srcptr w)
/* Assumes w finite and non-zero and z infinite; implementation
   according to C99 G.5.1.8                                     */
{
   int a, b, x, y;

   a = (mpfr_inf_p (mpc_realref (z)) ? MPFR_SIGNBIT (mpc_realref (z)) : 0);
   b = (mpfr_inf_p (mpc_imagref (z)) ? MPFR_SIGNBIT (mpc_imagref (z)) : 0);

   /* a is -1 if Re(z) = -Inf, 1 if Re(z) = +Inf, 0 if Re(z) is finite
      b is -1 if Im(z) = -Inf, 1 if Im(z) = +Inf, 0 if Im(z) is finite */

   /* x = MPC_MPFR_SIGN (a * mpc_realref (w) + b * mpc_imagref (w)) */
   /* y = MPC_MPFR_SIGN (b * mpc_realref (w) - a * mpc_imagref (w)) */
   if (a == 0 || b == 0) {
     /* only one of a or b can be zero, since z is infinite */
      x = a * MPC_MPFR_SIGN (mpc_realref (w)) + b * MPC_MPFR_SIGN (mpc_imagref (w));
      y = b * MPC_MPFR_SIGN (mpc_realref (w)) - a * MPC_MPFR_SIGN (mpc_imagref (w));
   }
   else {
      /* Both parts of z are infinite; x could be determined by sign
         considerations and comparisons. Since operations with non-finite
         numbers are not considered time-critical, we let mpfr do the work. */
      mpfr_t sign;

      mpfr_init2 (sign, 2);
      /* This is enough to determine the sign of sums and differences. */

      if (a == 1)
         if (b == 1) {
            mpfr_add (sign, mpc_realref (w), mpc_imagref (w), GMP_RNDN);
            x = MPC_MPFR_SIGN (sign);
            mpfr_sub (sign, mpc_realref (w), mpc_imagref (w), GMP_RNDN);
            y = MPC_MPFR_SIGN (sign);
         }
         else { /* b == -1 */
            mpfr_sub (sign, mpc_realref (w), mpc_imagref (w), GMP_RNDN);
            x = MPC_MPFR_SIGN (sign);
            mpfr_add (sign, mpc_realref (w), mpc_imagref (w), GMP_RNDN);
            y = -MPC_MPFR_SIGN (sign);
         }
      else /* a == -1 */
         if (b == 1) {
            mpfr_sub (sign, mpc_imagref (w), mpc_realref (w), GMP_RNDN);
            x = MPC_MPFR_SIGN (sign);
            mpfr_add (sign, mpc_realref (w), mpc_imagref (w), GMP_RNDN);
            y = MPC_MPFR_SIGN (sign);
         }
         else { /* b == -1 */
            mpfr_add (sign, mpc_realref (w), mpc_imagref (w), GMP_RNDN);
            x = -MPC_MPFR_SIGN (sign);
            mpfr_sub (sign, mpc_imagref (w), mpc_realref (w), GMP_RNDN);
            y = MPC_MPFR_SIGN (sign);
         }
      mpfr_clear (sign);
   }

   if (x == 0)
      mpfr_set_nan (mpc_realref (rop));
   else
      mpfr_set_inf (mpc_realref (rop), x);
   if (y == 0)
      mpfr_set_nan (mpc_imagref (rop));
   else
      mpfr_set_inf (mpc_imagref (rop), y);

   return MPC_INEX (0, 0); /* exact */
}


/* this routine deals with the case where z if finite and w infinite */
static int
mpc_div_fin_inf (mpc_ptr rop, mpc_srcptr z, mpc_srcptr w)
/* Assumes z finite and w infinite; implementation according to
   C99 G.5.1.8                                                  */
{
   mpfr_t c, d, a, b, x, y, zero;

   mpfr_init2 (c, 2); /* needed to hold a signed zero, +1 or -1 */
   mpfr_init2 (d, 2);
   mpfr_init2 (x, 2);
   mpfr_init2 (y, 2);
   mpfr_init2 (zero, 2);
   mpfr_set_ui (zero, 0ul, GMP_RNDN);
   mpfr_init2 (a, mpfr_get_prec (mpc_realref (z)));
   mpfr_init2 (b, mpfr_get_prec (mpc_imagref (z)));

   mpfr_set_ui (c, (mpfr_inf_p (mpc_realref (w)) ? 1 : 0), GMP_RNDN);
   MPFR_COPYSIGN (c, c, mpc_realref (w), GMP_RNDN);
   mpfr_set_ui (d, (mpfr_inf_p (mpc_imagref (w)) ? 1 : 0), GMP_RNDN);
   MPFR_COPYSIGN (d, d, mpc_imagref (w), GMP_RNDN);

   mpfr_mul (a, mpc_realref (z), c, GMP_RNDN); /* exact */
   mpfr_mul (b, mpc_imagref (z), d, GMP_RNDN);
   mpfr_add (x, a, b, GMP_RNDN);

   mpfr_mul (b, mpc_imagref (z), c, GMP_RNDN);
   mpfr_mul (a, mpc_realref (z), d, GMP_RNDN);
   mpfr_sub (y, b, a, GMP_RNDN);

   MPFR_COPYSIGN (mpc_realref (rop), zero, x, GMP_RNDN);
   MPFR_COPYSIGN (mpc_imagref (rop), zero, y, GMP_RNDN);

   mpfr_clear (c);
   mpfr_clear (d);
   mpfr_clear (x);
   mpfr_clear (y);
   mpfr_clear (zero);
   mpfr_clear (a);
   mpfr_clear (b);

   return MPC_INEX (0, 0); /* exact */
}


static int
mpc_div_real (mpc_ptr rop, mpc_srcptr z, mpc_srcptr w, mpc_rnd_t rnd)
/* Assumes z finite and w finite and non-zero, with imaginary part
   of w a signed zero.                                             */
{
   int inex_re, inex_im;
   /* save signs of operands in case there are overlaps */
   int zrs = MPFR_SIGNBIT (mpc_realref (z));
   int zis = MPFR_SIGNBIT (mpc_imagref (z));
   int wrs = MPFR_SIGNBIT (mpc_realref (w));
   int wis = MPFR_SIGNBIT (mpc_imagref (w));

   /* warning: rop may overlap with z,w so treat the imaginary part first */
   inex_im = mpfr_div (mpc_imagref(rop), mpc_imagref(z), mpc_realref(w), MPC_RND_IM(rnd));
   inex_re = mpfr_div (mpc_realref(rop), mpc_realref(z), mpc_realref(w), MPC_RND_RE(rnd));

   /* correct signs of zeroes if necessary, which does not affect the
      inexact flags                                                    */
   if (mpfr_zero_p (mpc_realref (rop)))
      mpfr_setsign (mpc_realref (rop), mpc_realref (rop), (zrs != wrs && zis != wis),
         GMP_RNDN); /* exact */
   if (mpfr_zero_p (mpc_imagref (rop)))
      mpfr_setsign (mpc_imagref (rop), mpc_imagref (rop), (zis != wrs && zrs == wis),
         GMP_RNDN);

   return MPC_INEX(inex_re, inex_im);
}


static int
mpc_div_imag (mpc_ptr rop, mpc_srcptr z, mpc_srcptr w, mpc_rnd_t rnd)
/* Assumes z finite and w finite and non-zero, with real part
   of w a signed zero.                                        */
{
   int inex_re, inex_im;
   int overlap = (rop == z) || (rop == w);
   int imag_z = mpfr_zero_p (mpc_realref (z));
   mpfr_t wloc;
   mpc_t tmprop;
   mpc_ptr dest = (overlap) ? tmprop : rop;
   /* save signs of operands in case there are overlaps */
   int zrs = MPFR_SIGNBIT (mpc_realref (z));
   int zis = MPFR_SIGNBIT (mpc_imagref (z));
   int wrs = MPFR_SIGNBIT (mpc_realref (w));
   int wis = MPFR_SIGNBIT (mpc_imagref (w));

   if (overlap)
      mpc_init3 (tmprop, MPC_PREC_RE (rop), MPC_PREC_IM (rop));

   wloc[0] = mpc_imagref(w)[0]; /* copies mpfr struct IM(w) into wloc */
   inex_re = mpfr_div (mpc_realref(dest), mpc_imagref(z), wloc, MPC_RND_RE(rnd));
   mpfr_neg (wloc, wloc, GMP_RNDN);
   /* changes the sign only in wloc, not in w; no need to correct later */
   inex_im = mpfr_div (mpc_imagref(dest), mpc_realref(z), wloc, MPC_RND_IM(rnd));

   if (overlap) {
      /* Note: we could use mpc_swap here, but this might cause problems
         if rop and tmprop have been allocated using different methods, since
         it will swap the significands of rop and tmprop. See
         http://lists.gforge.inria.fr/pipermail/mpc-discuss/2009-August/000504.html */
      mpc_set (rop, tmprop, MPC_RNDNN); /* exact */
      mpc_clear (tmprop);
   }

   /* correct signs of zeroes if necessary, which does not affect the
      inexact flags                                                    */
   if (mpfr_zero_p (mpc_realref (rop)))
      mpfr_setsign (mpc_realref (rop), mpc_realref (rop), (zrs != wrs && zis != wis),
         GMP_RNDN); /* exact */
   if (imag_z)
      mpfr_setsign (mpc_imagref (rop), mpc_imagref (rop), (zis != wrs && zrs == wis),
         GMP_RNDN);

   return MPC_INEX(inex_re, inex_im);
}


int
mpc_div (mpc_ptr a, mpc_srcptr b, mpc_srcptr c, mpc_rnd_t rnd)
{
   int ok_re = 0, ok_im = 0;
   mpc_t res, c_conj;
   mpfr_t q;
   mpfr_prec_t prec;
   int inex, inexact_prod, inexact_norm, inexact_re, inexact_im, loops = 0;
   int underflow_norm, overflow_norm, underflow_prod, overflow_prod;
   int underflow_re = 0, overflow_re = 0, underflow_im = 0, overflow_im = 0;
   mpfr_rnd_t rnd_re = MPC_RND_RE (rnd), rnd_im = MPC_RND_IM (rnd);
   int saved_underflow, saved_overflow;
   int tmpsgn;

   /* According to the C standard G.3, there are three types of numbers:   */
   /* finite (both parts are usual real numbers; contains 0), infinite     */
   /* (at least one part is a real infinity) and all others; the latter    */
   /* are numbers containing a nan, but no infinity, and could reasonably  */
   /* be called nan.                                                       */
   /* By G.5.1.4, infinite/finite=infinite; finite/infinite=0;             */
   /* all other divisions that are not finite/finite return nan+i*nan.     */
   /* Division by 0 could be handled by the following case of division by  */
   /* a real; we handle it separately instead.                             */
   if (mpc_zero_p (c))
      return mpc_div_zero (a, b, c, rnd);
   else if (mpc_inf_p (b) && mpc_fin_p (c))
         return mpc_div_inf_fin (a, b, c);
   else if (mpc_fin_p (b) && mpc_inf_p (c))
         return mpc_div_fin_inf (a, b, c);
   else if (!mpc_fin_p (b) || !mpc_fin_p (c)) {
      mpc_set_nan (a);
      return MPC_INEX (0, 0);
   }
   else if (mpfr_zero_p(mpc_imagref(c)))
      return mpc_div_real (a, b, c, rnd);
   else if (mpfr_zero_p(mpc_realref(c)))
      return mpc_div_imag (a, b, c, rnd);
      
   prec = MPC_MAX_PREC(a);

   mpc_init2 (res, 2);
   mpfr_init (q);

   /* create the conjugate of c in c_conj without allocating new memory */
   mpc_realref (c_conj)[0] = mpc_realref (c)[0];
   mpc_imagref (c_conj)[0] = mpc_imagref (c)[0];
   MPFR_CHANGE_SIGN (mpc_imagref (c_conj));

   /* save the underflow or overflow flags from MPFR */
   saved_underflow = mpfr_underflow_p ();
   saved_overflow = mpfr_overflow_p ();

   do {
      loops ++;
      prec += loops <= 2 ? mpc_ceil_log2 (prec) + 5 : prec / 2;

      mpc_set_prec (res, prec);
      mpfr_set_prec (q, prec);

      /* first compute norm(c) */
      mpfr_clear_underflow ();
      mpfr_clear_overflow ();
      inexact_norm = mpc_norm (q, c, GMP_RNDU);
      underflow_norm = mpfr_underflow_p ();
      overflow_norm = mpfr_overflow_p ();
      if (underflow_norm)
         mpfr_set_ui (q, 0ul, GMP_RNDN);
         /* to obtain divisions by 0 later on */

      /* now compute b*conjugate(c) */
      mpfr_clear_underflow ();
      mpfr_clear_overflow ();
      inexact_prod = mpc_mul (res, b, c_conj, MPC_RNDZZ);
      inexact_re = MPC_INEX_RE (inexact_prod);
      inexact_im = MPC_INEX_IM (inexact_prod);
      underflow_prod = mpfr_underflow_p ();
      overflow_prod = mpfr_overflow_p ();
         /* unfortunately, does not distinguish between under-/overflow
            in real or imaginary parts
            hopefully, the side-effects of mpc_mul do indeed raise the
            mpfr exceptions */
      if (overflow_prod) {
         int isinf = 0;
         tmpsgn = mpfr_sgn (mpc_realref(res));
         if (tmpsgn > 0)
           {
             mpfr_nextabove (mpc_realref(res));
             isinf = mpfr_inf_p (mpc_realref(res));
             mpfr_nextbelow (mpc_realref(res));
           }
         else if (tmpsgn < 0)
           {
             mpfr_nextbelow (mpc_realref(res));
             isinf = mpfr_inf_p (mpc_realref(res));
             mpfr_nextabove (mpc_realref(res));
           }
         if (isinf)
           {
             mpfr_set_inf (mpc_realref(res), tmpsgn);
             overflow_re = 1;
           }
         tmpsgn = mpfr_sgn (mpc_imagref(res));
         isinf = 0;
         if (tmpsgn > 0)
           {
             mpfr_nextabove (mpc_imagref(res));
             isinf = mpfr_inf_p (mpc_imagref(res));
             mpfr_nextbelow (mpc_imagref(res));
           }
         else if (tmpsgn < 0)
           {
             mpfr_nextbelow (mpc_imagref(res));
             isinf = mpfr_inf_p (mpc_imagref(res));
             mpfr_nextabove (mpc_imagref(res));
           }
         if (isinf)
           {
             mpfr_set_inf (mpc_imagref(res), tmpsgn);
             overflow_im = 1;
           }
         mpc_set (a, res, rnd);
         goto end;
      }

      /* divide the product by the norm */
      if (inexact_norm == 0 && (inexact_re == 0 || inexact_im == 0)) {
         /* The division has good chances to be exact in at least one part.  */
         /* Since this can cause problems when not rounding to the nearest,  */
         /* we use the division code of mpfr, which handles the situation.   */
         mpfr_clear_underflow ();
         mpfr_clear_overflow ();
         inexact_re |= mpfr_div (mpc_realref (res), mpc_realref (res), q, GMP_RNDZ);
         underflow_re = mpfr_underflow_p ();
         overflow_re = mpfr_overflow_p ();
         ok_re = !inexact_re || underflow_re || overflow_re
                 || mpfr_can_round (mpc_realref (res), prec - 4, GMP_RNDN,
                    GMP_RNDZ, MPC_PREC_RE(a) + (rnd_re == GMP_RNDN));

         if (ok_re) /* compute imaginary part */ {
            mpfr_clear_underflow ();
            mpfr_clear_overflow ();
            inexact_im |= mpfr_div (mpc_imagref (res), mpc_imagref (res), q, GMP_RNDZ);
            underflow_im = mpfr_underflow_p ();
            overflow_im = mpfr_overflow_p ();
            ok_im = !inexact_im || underflow_im || overflow_im
                    || mpfr_can_round (mpc_imagref (res), prec - 4, GMP_RNDN,
                       GMP_RNDZ, MPC_PREC_IM(a) + (rnd_im == GMP_RNDN));
         }
      }
      else {
         /* The division is inexact, so for efficiency reasons we invert q */
         /* only once and multiply by the inverse. */
         if (mpfr_ui_div (q, 1ul, q, GMP_RNDZ) || inexact_norm) {
             /* if 1/q is inexact, the approximations of the real and
                imaginary part below will be inexact, unless RE(res)
                or IM(res) is zero */
             inexact_re |= ~mpfr_zero_p (mpc_realref (res));
             inexact_im |= ~mpfr_zero_p (mpc_imagref (res));
         }
         mpfr_clear_underflow ();
         mpfr_clear_overflow ();
         inexact_re |= mpfr_mul (mpc_realref (res), mpc_realref (res), q, GMP_RNDZ);
         underflow_re = mpfr_underflow_p ();
         overflow_re = mpfr_overflow_p ();
         ok_re = !inexact_re || underflow_re || overflow_re
                 || mpfr_can_round (mpc_realref (res), prec - 4, GMP_RNDN,
                    GMP_RNDZ, MPC_PREC_RE(a) + (rnd_re == GMP_RNDN));

         if (ok_re) /* compute imaginary part */ {
            mpfr_clear_underflow ();
            mpfr_clear_overflow ();
            inexact_im |= mpfr_mul (mpc_imagref (res), mpc_imagref (res), q, GMP_RNDZ);
            underflow_im = mpfr_underflow_p ();
            overflow_im = mpfr_overflow_p ();
            ok_im = !inexact_im || underflow_im || overflow_im
                    || mpfr_can_round (mpc_imagref (res), prec - 4, GMP_RNDN,
                       GMP_RNDZ, MPC_PREC_IM(a) + (rnd_im == GMP_RNDN));
         }
      }
   } while ((!ok_re || !ok_im) && !underflow_norm && !overflow_norm
                               && !underflow_prod && !overflow_prod);

   inex = mpc_set (a, res, rnd);
   inexact_re = MPC_INEX_RE (inex);
   inexact_im = MPC_INEX_IM (inex);

 end:
   /* fix values and inexact flags in case of overflow/underflow */
   /* FIXME: heuristic, certainly does not cover all cases */
   if (overflow_re || (underflow_norm && !underflow_prod)) {
      mpfr_set_inf (mpc_realref (a), mpfr_sgn (mpc_realref (res)));
      inexact_re = mpfr_sgn (mpc_realref (res));
   }
   else if (underflow_re || (overflow_norm && !overflow_prod)) {
      inexact_re = mpfr_signbit (mpc_realref (res)) ? 1 : -1;
      mpfr_set_zero (mpc_realref (a), -inexact_re);
   }
   if (overflow_im || (underflow_norm && !underflow_prod)) {
      mpfr_set_inf (mpc_imagref (a), mpfr_sgn (mpc_imagref (res)));
      inexact_im = mpfr_sgn (mpc_imagref (res));
   }
   else if (underflow_im || (overflow_norm && !overflow_prod)) {
      inexact_im = mpfr_signbit (mpc_imagref (res)) ? 1 : -1;
      mpfr_set_zero (mpc_imagref (a), -inexact_im);
   }

   mpc_clear (res);
   mpfr_clear (q);

   /* restore underflow and overflow flags from MPFR */
   if (saved_underflow)
     mpfr_set_underflow ();
   if (saved_overflow)
     mpfr_set_overflow ();

   return MPC_INEX (inexact_re, inexact_im);
}
