/* mpc_pow_ui -- Raise a complex number to an integer power.

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

#include <limits.h> /* for CHAR_BIT */
#include "mpc-impl.h"

static int
mpc_pow_usi_naive (mpc_ptr z, mpc_srcptr x, unsigned long y, int sign,
   mpc_rnd_t rnd)
{
   int inex;
   mpc_t t;

   mpc_init3 (t, sizeof (unsigned long) * CHAR_BIT, MPFR_PREC_MIN);
   if (sign > 0)
      mpc_set_ui (t, y, MPC_RNDNN); /* exact */
   else
      mpc_set_si (t, - (signed long) y, MPC_RNDNN);
   inex = mpc_pow (z, x, t, rnd);
   mpc_clear (t);

   return inex;
}


int
mpc_pow_usi (mpc_ptr z, mpc_srcptr x, unsigned long y, int sign,
   mpc_rnd_t rnd)
   /* computes z = x^(sign*y) */
{
   int inex;
   mpc_t t, x3;
   mpfr_prec_t p, l, l0;
   long unsigned int u;
   int has3; /* non-zero if y has '11' in its binary representation */
   int loop, done;

   /* let mpc_pow deal with special values */
   if (!mpc_fin_p (x) || mpfr_zero_p (mpc_realref (x)) || mpfr_zero_p (mpc_imagref(x))
       || y == 0)
      return mpc_pow_usi_naive (z, x, y, sign, rnd);
   /* easy special cases */
   else if (y == 1) {
      if (sign > 0)
         return mpc_set (z, x, rnd);
      else
         return mpc_ui_div (z, 1ul, x, rnd);
   }
   else if (y == 2 && sign > 0)
      return mpc_sqr (z, x, rnd);
   /* let mpc_pow treat potential over- and underflows */
   else {
      mpfr_exp_t exp_r = mpfr_get_exp (mpc_realref (x)),
                 exp_i = mpfr_get_exp (mpc_imagref (x));
      if (   MPC_MAX (exp_r, exp_i) > mpfr_get_emax () / (mpfr_exp_t) y
             /* heuristic for overflow */
          || MPC_MAX (-exp_r, -exp_i) > (-mpfr_get_emin ()) / (mpfr_exp_t) y
             /* heuristic for underflow */
         )
         return mpc_pow_usi_naive (z, x, y, sign, rnd);
   }

   has3 = (y & (y >> 1)) != 0;
   for (l = 0, u = y; u > 3; l ++, u >>= 1);
   /* l>0 is the number of bits of y, minus 2, thus y has bits:
      y_{l+1} y_l y_{l-1} ... y_1 y_0 */
   l0 = l + 2;
   p = MPC_MAX_PREC(z) + l0 + 32; /* l0 ensures that y*2^{-p} <= 1 below */
   mpc_init2 (t, p);
   if (has3)
      mpc_init2 (x3, p);

   loop = 0;
   done = 0;
   while (!done) {
      loop++;

      mpc_sqr (t, x, MPC_RNDNN);
      if (has3) {
         mpc_mul (x3, t, x, MPC_RNDNN);
         if ((y >> l) & 1) /* y starts with 11... */
            mpc_set (t, x3, MPC_RNDNN);
      }
      while (l-- > 0) {
         mpc_sqr (t, t, MPC_RNDNN);
         if ((y >> l) & 1) {
            if ((l > 0) && ((y >> (l-1)) & 1)) /* implies has3 <> 0 */ {
               l--;
               mpc_sqr (t, t, MPC_RNDNN);
               mpc_mul (t, t, x3, MPC_RNDNN);
            }
            else
               mpc_mul (t, t, x, MPC_RNDNN);
         }
      }
      if (sign < 0)
         mpc_ui_div (t, 1ul, t, MPC_RNDNN);

      if (mpfr_zero_p (mpc_realref(t)) || mpfr_zero_p (mpc_imagref(t))) {
         inex = mpc_pow_usi_naive (z, x, y, sign, rnd);
            /* since mpfr_get_exp() is not defined for zero */
         done = 1;
      }
      else {
         /* see error bound in algorithms.tex; we use y<2^l0 instead of y-1
            also when sign>0                                                */
         mpfr_exp_t diff;
         mpfr_prec_t er, ei;

         diff = mpfr_get_exp (mpc_realref(t)) - mpfr_get_exp (mpc_imagref(t));
         /* the factor on the real part is 2+2^(-diff+2) <= 4 for diff >= 1
            and < 2^(-diff+3) for diff <= 0 */
         er = (diff >= 1) ? l0 + 3 : l0 + (-diff) + 3;
         /* the factor on the imaginary part is 2+2^(diff+2) <= 4 for diff <= -1
            and < 2^(diff+3) for diff >= 0 */
         ei = (diff <= -1) ? l0 + 3 : l0 + diff + 3;
         if (mpfr_can_round (mpc_realref(t), p - er, GMP_RNDN, GMP_RNDZ,
                              MPC_PREC_RE(z) + (MPC_RND_RE(rnd) == GMP_RNDN))
               && mpfr_can_round (mpc_imagref(t), p - ei, GMP_RNDN, GMP_RNDZ,
                              MPC_PREC_IM(z) + (MPC_RND_IM(rnd) == GMP_RNDN))) {
            inex = mpc_set (z, t, rnd);
            done = 1;
         }
         else if (loop == 1 && SAFE_ABS(mpfr_prec_t, diff) < MPC_MAX_PREC(z)) {
            /* common case, make a second trial at higher precision */
            p += MPC_MAX_PREC(x);
            mpc_set_prec (t, p);
            if (has3)
               mpc_set_prec (x3, p);
            l = l0 - 2;
         }
         else {
            /* stop the loop and use mpc_pow */
            inex = mpc_pow_usi_naive (z, x, y, sign, rnd);
            done = 1;
         }
      }
   }

   mpc_clear (t);
   if (has3)
      mpc_clear (x3);

   return inex;
}


int
mpc_pow_ui (mpc_ptr z, mpc_srcptr x, unsigned long y, mpc_rnd_t rnd)
{
  return mpc_pow_usi (z, x, y, 1, rnd);
}
