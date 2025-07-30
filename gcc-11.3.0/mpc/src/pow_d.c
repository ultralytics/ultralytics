/* mpc_pow_d -- Raise a complex number to a double-precision power.

Copyright (C) 2009 INRIA

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

#include <stdio.h>    /* for MPC_ASSERT */
#include <float.h>    /* for DBL_MANT_DIG */
#include "mpc-impl.h"

int
mpc_pow_d (mpc_ptr z, mpc_srcptr x, double y, mpc_rnd_t rnd)
{
  mpc_t yy;
  int inex;
  
  MPC_ASSERT(FLT_RADIX == 2);
  mpc_init3 (yy, DBL_MANT_DIG, MPFR_PREC_MIN);
  mpc_set_d (yy, y, MPC_RNDNN);   /* exact */
  inex = mpc_pow (z, x, yy, rnd);
  mpc_clear (yy);
  return inex;
}

