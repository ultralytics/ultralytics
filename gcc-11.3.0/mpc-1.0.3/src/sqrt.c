/* mpc_sqrt -- Take the square root of a complex number.

Copyright (C) 2002, 2008, 2009, 2010, 2011, 2012 INRIA

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

#if MPFR_VERSION_MAJOR < 3
#define mpfr_min_prec(x) \
   ( ((prec + BITS_PER_MP_LIMB - 1) / BITS_PER_MP_LIMB) * BITS_PER_MP_LIMB \
     - mpn_scan1 (x->_mpfr_d, 0))
#endif


int
mpc_sqrt (mpc_ptr a, mpc_srcptr b, mpc_rnd_t rnd)
{
  int ok_w, ok_t = 0;
  mpfr_t    w, t;
  mpfr_rnd_t  rnd_w, rnd_t;
  mpfr_prec_t prec_w, prec_t;
  /* the rounding mode and the precision required for w and t, which can */
  /* be either the real or the imaginary part of a */
  mpfr_prec_t prec;
  int inex_w, inex_t = 1, inex_re, inex_im, loops = 0;
  const int re_cmp = mpfr_cmp_ui (mpc_realref (b), 0),
            im_cmp = mpfr_cmp_ui (mpc_imagref (b), 0);
     /* comparison of the real/imaginary part of b with 0 */
  int repr_w, repr_t = 0 /* to avoid gcc warning */ ;
     /* flag indicating whether the computed value is already representable
        at the target precision */
  const int im_sgn = mpfr_signbit (mpc_imagref (b)) == 0 ? 0 : -1;
     /* we need to know the sign of Im(b) when it is +/-0 */
  const mpfr_rnd_t r = im_sgn ? GMP_RNDD : GMP_RNDU;
     /* rounding mode used when computing t */

  /* special values */
  if (!mpc_fin_p (b)) {
   /* sqrt(x +i*Inf) = +Inf +I*Inf, even if x = NaN */
   /* sqrt(x -i*Inf) = +Inf -I*Inf, even if x = NaN */
   if (mpfr_inf_p (mpc_imagref (b)))
      {
         mpfr_set_inf (mpc_realref (a), +1);
         mpfr_set_inf (mpc_imagref (a), im_sgn);
         return MPC_INEX (0, 0);
      }

   if (mpfr_inf_p (mpc_realref (b)))
      {
         if (mpfr_signbit (mpc_realref (b)))
         {
            if (mpfr_number_p (mpc_imagref (b)))
               {
               /* sqrt(-Inf +i*y) = +0 +i*Inf, when y positive */
               /* sqrt(-Inf +i*y) = +0 -i*Inf, when y positive */
               mpfr_set_ui (mpc_realref (a), 0, GMP_RNDN);
               mpfr_set_inf (mpc_imagref (a), im_sgn);
               return MPC_INEX (0, 0);
               }
            else
               {
               /* sqrt(-Inf +i*NaN) = NaN +/-i*Inf */
               mpfr_set_nan (mpc_realref (a));
               mpfr_set_inf (mpc_imagref (a), im_sgn);
               return MPC_INEX (0, 0);
               }
         }
         else
         {
            if (mpfr_number_p (mpc_imagref (b)))
               {
               /* sqrt(+Inf +i*y) = +Inf +i*0, when y positive */
               /* sqrt(+Inf +i*y) = +Inf -i*0, when y positive */
               mpfr_set_inf (mpc_realref (a), +1);
               mpfr_set_ui (mpc_imagref (a), 0, GMP_RNDN);
               if (im_sgn)
                  mpc_conj (a, a, MPC_RNDNN);
               return MPC_INEX (0, 0);
               }
            else
               {
               /* sqrt(+Inf -i*Inf) = +Inf -i*Inf */
               /* sqrt(+Inf +i*Inf) = +Inf +i*Inf */
               /* sqrt(+Inf +i*NaN) = +Inf +i*NaN */
               return mpc_set (a, b, rnd);
               }
         }
      }

   /* sqrt(x +i*NaN) = NaN +i*NaN, if x is not infinite */
   /* sqrt(NaN +i*y) = NaN +i*NaN, if y is not infinite */
   if (mpfr_nan_p (mpc_realref (b)) || mpfr_nan_p (mpc_imagref (b)))
      {
         mpfr_set_nan (mpc_realref (a));
         mpfr_set_nan (mpc_imagref (a));
         return MPC_INEX (0, 0);
      }
  }

  /* purely real */
  if (im_cmp == 0)
    {
      if (re_cmp == 0)
        {
          mpc_set_ui_ui (a, 0, 0, MPC_RNDNN);
          if (im_sgn)
            mpc_conj (a, a, MPC_RNDNN);
          return MPC_INEX (0, 0);
        }
      else if (re_cmp > 0)
        {
          inex_w = mpfr_sqrt (mpc_realref (a), mpc_realref (b), MPC_RND_RE (rnd));
          mpfr_set_ui (mpc_imagref (a), 0, GMP_RNDN);
          if (im_sgn)
            mpc_conj (a, a, MPC_RNDNN);
          return MPC_INEX (inex_w, 0);
        }
      else
        {
          mpfr_init2 (w, MPC_PREC_RE (b));
          mpfr_neg (w, mpc_realref (b), GMP_RNDN);
          if (im_sgn)
            {
              inex_w = -mpfr_sqrt (mpc_imagref (a), w, INV_RND (MPC_RND_IM (rnd)));
              mpfr_neg (mpc_imagref (a), mpc_imagref (a), GMP_RNDN);
            }
          else
            inex_w = mpfr_sqrt (mpc_imagref (a), w, MPC_RND_IM (rnd));

          mpfr_set_ui (mpc_realref (a), 0, GMP_RNDN);
          mpfr_clear (w);
          return MPC_INEX (0, inex_w);
        }
    }

  /* purely imaginary */
  if (re_cmp == 0)
    {
      mpfr_t y;

      y[0] = mpc_imagref (b)[0];
      /* If y/2 underflows, so does sqrt(y/2) */
      mpfr_div_2ui (y, y, 1, GMP_RNDN);
      if (im_cmp > 0)
        {
          inex_w = mpfr_sqrt (mpc_realref (a), y, MPC_RND_RE (rnd));
          inex_t = mpfr_sqrt (mpc_imagref (a), y, MPC_RND_IM (rnd));
        }
      else
        {
          mpfr_neg (y, y, GMP_RNDN);
          inex_w = mpfr_sqrt (mpc_realref (a), y, MPC_RND_RE (rnd));
          inex_t = -mpfr_sqrt (mpc_imagref (a), y, INV_RND (MPC_RND_IM (rnd)));
          mpfr_neg (mpc_imagref (a), mpc_imagref (a), GMP_RNDN);
        }
      return MPC_INEX (inex_w, inex_t);
    }

  prec = MPC_MAX_PREC(a);

  mpfr_init (w);
  mpfr_init (t);

   if (re_cmp > 0) {
      rnd_w = MPC_RND_RE (rnd);
      prec_w = MPC_PREC_RE (a);
      rnd_t = MPC_RND_IM(rnd);
      if (rnd_t == GMP_RNDZ)
         /* force GMP_RNDD or GMP_RNDUP, using sign(t) = sign(y) */
         rnd_t = (im_cmp > 0 ? GMP_RNDD : GMP_RNDU);
      prec_t = MPC_PREC_IM (a);
   }
   else {
      prec_w = MPC_PREC_IM (a);
      prec_t = MPC_PREC_RE (a);
      if (im_cmp > 0) {
         rnd_w = MPC_RND_IM(rnd);
         rnd_t = MPC_RND_RE(rnd);
         if (rnd_t == GMP_RNDZ)
            rnd_t = GMP_RNDD;
      }
      else {
         rnd_w = INV_RND(MPC_RND_IM (rnd));
         rnd_t = INV_RND(MPC_RND_RE (rnd));
         if (rnd_t == GMP_RNDZ)
            rnd_t = GMP_RNDU;
      }
   }

  do
    {
      loops ++;
      prec += (loops <= 2) ? mpc_ceil_log2 (prec) + 4 : prec / 2;
      mpfr_set_prec (w, prec);
      mpfr_set_prec (t, prec);
      /* let b = x + iy */
      /* w = sqrt ((|x| + sqrt (x^2 + y^2)) / 2), rounded down */
      /* total error bounded by 3 ulps */
      inex_w = mpc_abs (w, b, GMP_RNDD);
      if (re_cmp < 0)
        inex_w |= mpfr_sub (w, w, mpc_realref (b), GMP_RNDD);
      else
        inex_w |= mpfr_add (w, w, mpc_realref (b), GMP_RNDD);
      inex_w |= mpfr_div_2ui (w, w, 1, GMP_RNDD);
      inex_w |= mpfr_sqrt (w, w, GMP_RNDD);

      repr_w = mpfr_min_prec (w) <= prec_w;
      if (!repr_w)
         /* use the usual trick for obtaining the ternary value */
         ok_w = mpfr_can_round (w, prec - 2, GMP_RNDD, GMP_RNDU,
                                prec_w + (rnd_w == GMP_RNDN));
      else {
            /* w is representable in the target precision and thus cannot be
               rounded up */
         if (rnd_w == GMP_RNDN)
            /* If w can be rounded to nearest, then actually no rounding
               occurs, and the ternary value is known from inex_w. */
            ok_w = mpfr_can_round (w, prec - 2, GMP_RNDD, GMP_RNDN, prec_w);
         else
            /* If w can be rounded down, then any direct rounding and the
               ternary flag can be determined from inex_w. */
            ok_w = mpfr_can_round (w, prec - 2, GMP_RNDD, GMP_RNDD, prec_w);
      }

      if (!inex_w || ok_w) {
         /* t = y / 2w, rounded away */
         /* total error bounded by 7 ulps */
         inex_t = mpfr_div (t, mpc_imagref (b), w, r);
         if (!inex_t && inex_w)
            /* The division was exact, but w was not. */
            inex_t = im_sgn ? -1 : 1;
         inex_t |= mpfr_div_2ui (t, t, 1, r);
         repr_t = mpfr_min_prec (t) <= prec_t;
         if (!repr_t)
             /* As for w; since t was rounded away, we check whether rounding to 0
                is possible. */
            ok_t = mpfr_can_round (t, prec - 3, r, GMP_RNDZ,
                                   prec_t + (rnd_t == GMP_RNDN));
         else {
            if (rnd_t == GMP_RNDN)
               ok_t = mpfr_can_round (t, prec - 3, r, GMP_RNDN, prec_t);
            else
               ok_t = mpfr_can_round (t, prec - 3, r, r, prec_t);
         }
      }
    }
    while ((inex_w && !ok_w) || (inex_t && !ok_t));

   if (re_cmp > 0) {
         inex_re = mpfr_set (mpc_realref (a), w, MPC_RND_RE(rnd));
         inex_im = mpfr_set (mpc_imagref (a), t, MPC_RND_IM(rnd));
   }
   else if (im_cmp > 0) {
      inex_re = mpfr_set (mpc_realref(a), t, MPC_RND_RE(rnd));
      inex_im = mpfr_set (mpc_imagref(a), w, MPC_RND_IM(rnd));
   }
   else {
      inex_re = mpfr_neg (mpc_realref (a), t, MPC_RND_RE(rnd));
      inex_im = mpfr_neg (mpc_imagref (a), w, MPC_RND_IM(rnd));
   }

   if (repr_w && inex_w) {
      if (rnd_w == GMP_RNDN) {
         /* w has not been rounded with mpfr_set/mpfr_neg, determine ternary
            value from inex_w instead */
         if (re_cmp > 0)
            inex_re = inex_w;
         else if (im_cmp > 0)
            inex_im = inex_w;
         else
            inex_im = -inex_w;
      }
      else {
         /* determine ternary value, but also potentially add 1 ulp; can only
            be done now when we are in the target precision */
         if (re_cmp > 0) {
            if (rnd_w == GMP_RNDU) {
               MPFR_ADD_ONE_ULP (mpc_realref (a));
               inex_re = +1;
            }
            else
               inex_re = -1;
         }
         else if (im_cmp > 0) {
            if (rnd_w == GMP_RNDU) {
               MPFR_ADD_ONE_ULP (mpc_imagref (a));
               inex_im = +1;
            }
            else
               inex_im = -1;
         }
         else {
            if (rnd_w == GMP_RNDU) {
               MPFR_ADD_ONE_ULP (mpc_imagref (a));
               inex_im = -1;
            }
            else
               inex_im = +1;
         }
      }
   }
   if (repr_t && inex_t) {
      if (rnd_t == GMP_RNDN) {
         if (re_cmp > 0)
            inex_im = inex_t;
         else if (im_cmp > 0)
            inex_re = inex_t;
         else
            inex_re = -inex_t;
      }
      else {
         if (re_cmp > 0) {
            if (rnd_t == r)
               inex_im = inex_t;
            else {
               inex_im = -inex_t;
               /* im_cmp > 0 implies that Im(b) > 0, thus im_sgn = 0
                  and r = GMP_RNDU.
                  im_cmp < 0 implies that Im(b) < 0, thus im_sgn = -1
                  and r = GMP_RNDD. */
               MPFR_SUB_ONE_ULP (mpc_imagref (a));
            }
         }
         else if (im_cmp > 0) {
            if (rnd_t == r)
               inex_re = inex_t;
            else {
               inex_re = -inex_t;
               /* im_cmp > 0 implies r = GMP_RNDU (see above) */
               MPFR_SUB_ONE_ULP (mpc_realref (a));
            }
         }
         else { /* im_cmp < 0 */
            if (rnd_t == r)
               inex_re = -inex_t;
            else {
               inex_re = inex_t;
               /* im_cmp < 0 implies r = GMP_RNDD (see above) */
               MPFR_SUB_ONE_ULP (mpc_realref (a));
            }
         }
      }
   }

  mpfr_clear (w);
  mpfr_clear (t);

  return MPC_INEX (inex_re, inex_im);
}
