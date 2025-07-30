/* mpc_proj -- projection of a complex number onto the Riemann sphere.

Copyright (C) 2008, 2009, 2011 INRIA

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
mpc_proj (mpc_ptr a, mpc_srcptr b, mpc_rnd_t rnd)
{
   if (mpc_inf_p (b)) {
      /* infinities project to +Inf +i* copysign(0.0, cimag(z)) */
      mpfr_set_inf (mpc_realref (a), +1);
      mpfr_set_zero (mpc_imagref (a), (mpfr_signbit (mpc_imagref (b)) ? -1 : 1));
      return MPC_INEX (0, 0);
   }
   else
      return mpc_set (a, b, rnd);
}
