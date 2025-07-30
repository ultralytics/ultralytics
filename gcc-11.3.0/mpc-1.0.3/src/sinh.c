/* mpc_sinh -- hyperbolic sine of a complex number.

Copyright (C)2008, 2009, 2011, 2012 INRIA

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
mpc_sinh (mpc_ptr rop, mpc_srcptr op, mpc_rnd_t rnd)
{
  /* sinh(op) = -i*sin(i*op) = conj(-i*sin(conj(-i*op))) */
  mpc_t z;
  mpc_t sin_z;
  int inex;

  /* z := conj(-i * op) and rop = conj(-i * sin(z)), in other words, we have
     to switch real and imaginary parts. Let us set them without copying
     significands. */
  mpc_realref (z)[0] = mpc_imagref (op)[0];
  mpc_imagref (z)[0] = mpc_realref (op)[0];
  mpc_realref (sin_z)[0] = mpc_imagref (rop)[0];
  mpc_imagref (sin_z)[0] = mpc_realref (rop)[0];

  inex = mpc_sin (sin_z, z, MPC_RND (MPC_RND_IM (rnd), MPC_RND_RE (rnd)));

  /* sin_z and rop parts share the same significands, copy the rest now. */
  mpc_realref (rop)[0] = mpc_imagref (sin_z)[0];
  mpc_imagref (rop)[0] = mpc_realref (sin_z)[0];

  /* swap inexact flags for real and imaginary parts */
  return MPC_INEX (MPC_INEX_IM (inex), MPC_INEX_RE (inex));
}
