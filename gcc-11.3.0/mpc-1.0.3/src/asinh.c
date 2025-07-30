/* mpc_asinh -- inverse hyperbolic sine of a complex number.

Copyright (C) 2009, 2011, 2012 INRIA

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
mpc_asinh (mpc_ptr rop, mpc_srcptr op, mpc_rnd_t rnd)
{
  /* asinh(op) = -i*asin(i*op) */
  int inex;
  mpc_t z, a;
  mpfr_t tmp;

  /* z = i*op */
  mpc_realref (z)[0] = mpc_imagref (op)[0];
  mpc_imagref (z)[0] = mpc_realref (op)[0];
  MPFR_CHANGE_SIGN (mpc_realref (z));

  /* Note reversal of precisions due to later multiplication by -i */
  mpc_init3 (a, MPC_PREC_IM(rop), MPC_PREC_RE(rop));

  inex = mpc_asin (a, z,
                   MPC_RND (INV_RND (MPC_RND_IM (rnd)), MPC_RND_RE (rnd)));

  /* if a = asin(i*op) = x+i*y, and we want y-i*x */

  /* change a to -i*a */
  tmp[0] = mpc_realref (a)[0];
  mpc_realref (a)[0] = mpc_imagref (a)[0];
  mpc_imagref (a)[0] = tmp[0];
  MPFR_CHANGE_SIGN (mpc_imagref (a));

  mpc_set (rop, a, MPC_RNDNN);   /* exact */

  mpc_clear (a);

  return MPC_INEX (MPC_INEX_IM (inex), -MPC_INEX_RE (inex));
}
