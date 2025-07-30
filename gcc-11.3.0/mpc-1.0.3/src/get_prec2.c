/* mpc_get_prec2 -- returns the precisions of the real and of the imaginary
   part through the first two arguments

Copyright (C) 2007, 2009, 2010 INRIA

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
mpc_get_prec2 (mpfr_prec_t *pr, mpfr_prec_t *pi, mpc_srcptr x)
{
   *pr = MPC_PREC_RE (x);
   *pi = MPC_PREC_IM (x);
}
