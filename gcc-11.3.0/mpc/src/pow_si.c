/* mpc_pow_si -- Raise a complex number to an integer power.

Copyright (C) 2009, 2010 INRIA

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
mpc_pow_si (mpc_ptr z, mpc_srcptr x, long y, mpc_rnd_t rnd)
{
   if (y >= 0)
     return mpc_pow_usi (z, x, (unsigned long) y, 1, rnd);
   else
     return mpc_pow_usi (z, x, (unsigned long) (-y), -1, rnd);
}
