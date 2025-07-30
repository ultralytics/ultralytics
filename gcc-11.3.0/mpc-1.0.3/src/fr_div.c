/* mpc_fr_div -- Divide a floating-point number by a complex number.

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
mpc_fr_div (mpc_ptr a, mpfr_srcptr b, mpc_srcptr c, mpc_rnd_t rnd)
{
   mpc_t bc;
   int inexact;

   mpc_realref (bc)[0] = b [0];
   mpfr_init (mpc_imagref (bc));
   /* we consider the operand b to have imaginary part +0 */
   mpfr_set_ui (mpc_imagref (bc), 0, GMP_RNDN);

   inexact = mpc_div (a, bc, c, rnd);

   mpfr_clear (mpc_imagref (bc));

   return inexact;
}
