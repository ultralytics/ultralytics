/* mpc_swap -- Swap two complex numbers.

Copyright (C) 2009, 2011 INRIA

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

void
mpc_swap (mpc_ptr a, mpc_ptr b)
{
  /* assumes real and imaginary parts do not overlap */
  mpfr_swap (mpc_realref(a), mpc_realref(b));
  mpfr_swap (mpc_imagref(a), mpc_imagref(b));
}
