/* mpc_fma -- Fused multiply-add of three complex numbers

Copyright (C) 2011, 2012 INRIA

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

/* return a bound on the precision needed to add or subtract x and y exactly */
static mpfr_prec_t
bound_prec_addsub (mpfr_srcptr x, mpfr_srcptr y)
{
  if (!mpfr_regular_p (x))
    return mpfr_get_prec (y);
  else if (!mpfr_regular_p (y))
    return mpfr_get_prec (x);
  else /* neither x nor y are NaN, Inf or zero */
    {
      mpfr_exp_t ex = mpfr_get_exp (x);
      mpfr_exp_t ey = mpfr_get_exp (y);
      mpfr_exp_t ulpx = ex - mpfr_get_prec (x);
      mpfr_exp_t ulpy = ey - mpfr_get_prec (y);
      return ((ex >= ey) ? ex : ey) + 1 - ((ulpx <= ulpy) ? ulpx : ulpy);
    }
}

/* r <- a*b+c */
int
mpc_fma_naive (mpc_ptr r, mpc_srcptr a, mpc_srcptr b, mpc_srcptr c, mpc_rnd_t rnd)
{
  mpfr_t rea_reb, rea_imb, ima_reb, ima_imb, tmp;
  mpfr_prec_t pre12, pre13, pre23, pim12, pim13, pim23;
  int inex_re, inex_im;

  mpfr_init2 (rea_reb, mpfr_get_prec (mpc_realref(a)) + mpfr_get_prec (mpc_realref(b)));
  mpfr_init2 (rea_imb, mpfr_get_prec (mpc_realref(a)) + mpfr_get_prec (mpc_imagref(b)));
  mpfr_init2 (ima_reb, mpfr_get_prec (mpc_imagref(a)) + mpfr_get_prec (mpc_realref(b)));
  mpfr_init2 (ima_imb, mpfr_get_prec (mpc_imagref(a)) + mpfr_get_prec (mpc_imagref(b)));

  mpfr_mul (rea_reb, mpc_realref(a), mpc_realref(b), GMP_RNDZ); /* exact */
  mpfr_mul (rea_imb, mpc_realref(a), mpc_imagref(b), GMP_RNDZ); /* exact */
  mpfr_mul (ima_reb, mpc_imagref(a), mpc_realref(b), GMP_RNDZ); /* exact */
  mpfr_mul (ima_imb, mpc_imagref(a), mpc_imagref(b), GMP_RNDZ); /* exact */

  /* Re(r) <- rea_reb - ima_imb + Re(c) */

  pre12 = bound_prec_addsub (rea_reb, ima_imb); /* bound on exact precision for
						   rea_reb - ima_imb */
  pre13 = bound_prec_addsub (rea_reb, mpc_realref(c));
  /* bound for rea_reb + Re(c) */
  pre23 = bound_prec_addsub (ima_imb, mpc_realref(c));
  /* bound for ima_imb - Re(c) */
  if (pre12 <= pre13 && pre12 <= pre23) /* (rea_reb - ima_imb) + Re(c) */
    {
      mpfr_init2 (tmp, pre12);
      mpfr_sub (tmp, rea_reb, ima_imb, GMP_RNDZ); /* exact */
      inex_re = mpfr_add (mpc_realref(r), tmp, mpc_realref(c), MPC_RND_RE(rnd));
      /* the only possible bad overlap is between r and c, but since we are
	 only touching the real part of both, it is ok */
    }
  else if (pre13 <= pre23) /* (rea_reb + Re(c)) - ima_imb */
    {
      mpfr_init2 (tmp, pre13);
      mpfr_add (tmp, rea_reb, mpc_realref(c), GMP_RNDZ); /* exact */
      inex_re = mpfr_sub (mpc_realref(r), tmp, ima_imb, MPC_RND_RE(rnd));
      /* the only possible bad overlap is between r and c, but since we are
	 only touching the real part of both, it is ok */
    }
  else /* rea_reb + (Re(c) - ima_imb) */
    {
      mpfr_init2 (tmp, pre23);
      mpfr_sub (tmp, mpc_realref(c), ima_imb, GMP_RNDZ); /* exact */
      inex_re = mpfr_add (mpc_realref(r), tmp, rea_reb, MPC_RND_RE(rnd));
      /* the only possible bad overlap is between r and c, but since we are
	 only touching the real part of both, it is ok */
    }

  /* Im(r) <- rea_imb + ima_reb + Im(c) */
  pim12 = bound_prec_addsub (rea_imb, ima_reb); /* bound on exact precision for
						   rea_imb + ima_reb */
  pim13 = bound_prec_addsub (rea_imb, mpc_imagref(c));
  /* bound for rea_imb + Im(c) */
  pim23 = bound_prec_addsub (ima_reb, mpc_imagref(c));
  /* bound for ima_reb + Im(c) */
  if (pim12 <= pim13 && pim12 <= pim23) /* (rea_imb + ima_reb) + Im(c) */
    {
      mpfr_set_prec (tmp, pim12);
      mpfr_add (tmp, rea_imb, ima_reb, GMP_RNDZ); /* exact */
      inex_im = mpfr_add (mpc_imagref(r), tmp, mpc_imagref(c), MPC_RND_IM(rnd));
      /* the only possible bad overlap is between r and c, but since we are
	 only touching the imaginary part of both, it is ok */
    }
  else if (pim13 <= pim23) /* (rea_imb + Im(c)) + ima_reb */
    {
      mpfr_set_prec (tmp, pim13);
      mpfr_add (tmp, rea_imb, mpc_imagref(c), GMP_RNDZ); /* exact */
      inex_im = mpfr_add (mpc_imagref(r), tmp, ima_reb, MPC_RND_IM(rnd));
      /* the only possible bad overlap is between r and c, but since we are
	 only touching the imaginary part of both, it is ok */
    }
  else /* rea_imb + (Im(c) + ima_reb) */
    {
      mpfr_set_prec (tmp, pre23);
      mpfr_add (tmp, mpc_imagref(c), ima_reb, GMP_RNDZ); /* exact */
      inex_im = mpfr_add (mpc_imagref(r), tmp, rea_imb, MPC_RND_IM(rnd));
      /* the only possible bad overlap is between r and c, but since we are
	 only touching the imaginary part of both, it is ok */
    }

  mpfr_clear (rea_reb);
  mpfr_clear (rea_imb);
  mpfr_clear (ima_reb);
  mpfr_clear (ima_imb);
  mpfr_clear (tmp);

  return MPC_INEX(inex_re, inex_im);
}

/* The algorithm is as follows:
   - in a first pass, we use the target precision + some extra bits
   - if it fails, we add the number of cancelled bits when adding
     Re(a*b) and Re(c) [similarly for the imaginary part]
   - it is fails again, we call the mpc_fma_naive function, which also
     deals with the special cases */
int
mpc_fma (mpc_ptr r, mpc_srcptr a, mpc_srcptr b, mpc_srcptr c, mpc_rnd_t rnd)
{
  mpc_t ab;
  mpfr_prec_t pre, pim, wpre, wpim;
  mpfr_exp_t diffre, diffim;
  int i, inex = 0, okre = 0, okim = 0;

  if (mpc_fin_p (a) == 0 || mpc_fin_p (b) == 0 || mpc_fin_p (c) == 0)
    return mpc_fma_naive (r, a, b, c, rnd);

  pre = mpfr_get_prec (mpc_realref(r));
  pim = mpfr_get_prec (mpc_imagref(r));
  wpre = pre + mpc_ceil_log2 (pre) + 10;
  wpim = pim + mpc_ceil_log2 (pim) + 10;
  mpc_init3 (ab, wpre, wpim);
  for (i = 0; i < 2; ++i)
    {
      mpc_mul (ab, a, b, MPC_RNDZZ);
      if (mpfr_zero_p (mpc_realref(ab)) || mpfr_zero_p (mpc_imagref(ab)))
        break;
      diffre = mpfr_get_exp (mpc_realref(ab));
      diffim = mpfr_get_exp (mpc_imagref(ab));
      mpc_add (ab, ab, c, MPC_RNDZZ);
      if (mpfr_zero_p (mpc_realref(ab)) || mpfr_zero_p (mpc_imagref(ab)))
        break;
      diffre -= mpfr_get_exp (mpc_realref(ab));
      diffim -= mpfr_get_exp (mpc_imagref(ab));
      diffre = (diffre > 0 ? diffre + 1 : 1);
      diffim = (diffim > 0 ? diffim + 1 : 1);
      okre = diffre > (mpfr_exp_t) wpre ? 0 : mpfr_can_round (mpc_realref(ab),
                                 wpre - diffre, GMP_RNDN, GMP_RNDZ,
                                 pre + (MPC_RND_RE (rnd) == GMP_RNDN));
      okim = diffim > (mpfr_exp_t) wpim ? 0 : mpfr_can_round (mpc_imagref(ab),
                                 wpim - diffim, GMP_RNDN, GMP_RNDZ,
                                 pim + (MPC_RND_IM (rnd) == GMP_RNDN));
      if (okre && okim)
        {
          inex = mpc_set (r, ab, rnd);
          break;
        }
      if (i == 1)
        break;
      if (okre == 0 && diffre > 1)
        wpre += diffre;
      if (okim == 0 && diffim > 1)
        wpim += diffim;
      mpfr_set_prec (mpc_realref(ab), wpre);
      mpfr_set_prec (mpc_imagref(ab), wpim);
    }
  mpc_clear (ab);
  return okre && okim ? inex : mpc_fma_naive (r, a, b, c, rnd);
}
