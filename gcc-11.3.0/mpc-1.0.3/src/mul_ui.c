/* mpc_mul_ui -- Multiply a complex number by a nonnegative integer.

Copyright (C) 2002, 2009, 2011 INRIA

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
mpc_mul_ui (mpc_ptr a, mpc_srcptr b, unsigned long int c, mpc_rnd_t rnd)
{
  int inex_re, inex_im;

  inex_re = mpfr_mul_ui (mpc_realref(a), mpc_realref(b), c, MPC_RND_RE(rnd));
  inex_im = mpfr_mul_ui (mpc_imagref(a), mpc_imagref(b), c, MPC_RND_IM(rnd));

  return MPC_INEX(inex_re, inex_im);
}
