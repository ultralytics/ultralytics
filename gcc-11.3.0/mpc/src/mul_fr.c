/* mpc_mul_fr -- Multiply a complex number by a floating-point number.

Copyright (C) 2002, 2008, 2009, 2010, 2011 INRIA

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
mpc_mul_fr (mpc_ptr a, mpc_srcptr b, mpfr_srcptr c, mpc_rnd_t rnd)
{
  int inex_re, inex_im;
  mpfr_t real;

  if (c == mpc_realref (a))
    /* We have to use a temporary variable. */
    mpfr_init2 (real, MPC_PREC_RE (a));
  else
    real [0] = mpc_realref (a) [0];

  inex_re = mpfr_mul (real, mpc_realref(b), c, MPC_RND_RE(rnd));
  inex_im = mpfr_mul (mpc_imagref(a), mpc_imagref(b), c, MPC_RND_IM(rnd));
  mpfr_set (mpc_realref (a), real, GMP_RNDN); /* exact */

  if (c == mpc_realref (a))
    mpfr_clear (real);

  return MPC_INEX(inex_re, inex_im);
}
